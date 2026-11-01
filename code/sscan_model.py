#!/usr/bin/env python3
"""Capacity-and-reuse-distance model of shared-scan vs request-level prefix caching.

A discrete-event model of one GPU serving semantic-filter requests:
  * service: requests are served FCFS; a request costs alpha * computed_prefill_tokens + beta
    (+ gamma * decode tokens for the extract family);
  * cache: GPU KV behaves as an LRU cache of per-document prompt prefixes with usable capacity
    kappa * C tokens; a request whose document prefix is resident computes only its question tail;
  * policies release requests exactly as the live harness does (eager, bounded window W, lazy
    circular block scan with F blocks of stage-1 work in flight);
  * stage-2 requests are spawned with the stage-1 pass rate estimated on calibration runs (the
    model never sees a held-out run's labels).
Parameters (alpha, beta, kappa, pass rate) are fitted on calibration runs only; the model then
predicts cached fraction, makespan and p95 latency for held-out configurations.
"""
import argparse
import heapq
import itertools
import json
import math
import random
from collections import OrderedDict, deque


class BlockLRU:
    """Prefix-block KV cache in the style of vLLM APC: 16-token blocks, LRU eviction, and a
    block can only hit when every earlier block of the same prompt hit."""

    def __init__(self, cap_blocks, enabled=True):
        self.cap, self.enabled, self.d = cap_blocks, enabled, OrderedDict()

    def lookup_insert(self, keys):
        hits = 0
        if self.enabled:
            for k in keys:
                if k in self.d:
                    self.d.move_to_end(k)
                    hits += 1
                else:
                    break
        if self.enabled:
            for k in keys[hits:]:
                self.d[k] = True
                self.d.move_to_end(k)
            while len(self.d) > self.cap:
                self.d.popitem(last=False)
        return hits


def docs_of(q, n):
    return [(q["lo"] + k) % n for k in range(q["n"])]


BLK = 16


def simulate(cfg, trace, policy, params, rng):
    """Event-driven model of one GPU. cfg: n, doc_tokens, shared_prefix, preamble, kv_tokens,
    block, family, inflight, window (window policy), apc. Returns cached fraction, makespan,
    p95 and mean latency."""
    a, b, kap, p_pass = params["alpha"], params["beta"], params["kappa"], params["pass_rate"]
    gam = params.get("gamma", 0.0)
    n, L, SP, pre = cfg["n"], cfg["doc_tokens"], cfg["shared_prefix"], cfg["preamble"]
    cache = BlockLRU(int(kap * cfg["kv_tokens"] / BLK), enabled=cfg.get("apc", True))
    pre_blocks = pre // BLK
    stats = dict(computed=0.0, prompt=0.0, requests=0)
    q_by = {q["qid"]: dict(q) for q in trace}
    for q in q_by.values():
        q["_docs"] = set(docs_of(q, n))
    pending = {qid: 0 for qid in q_by}
    released_all = {qid: False for qid in q_by}
    done_t = {}
    queue = deque()
    heap, seq = [], itertools.count()
    for q in q_by.values():
        heapq.heappush(heap, (q["arrival"], next(seq), "arrive", q["qid"]))
    if policy == "window":
        W = cfg["window"]
        last = max(q["arrival"] for q in q_by.values())
        k = 1
        while (k - 1) * W <= last:
            heapq.heappush(heap, (k * W, next(seq), "window", None))
            k += 1
    nb = math.ceil(n / cfg["block"])
    need, active, s1_inflight, window_buf = {}, [], {}, []
    pos = [0]
    busy = [False]

    def keys_for(d):
        nblk = SP[d] // BLK
        return [("pre", i) for i in range(pre_blocks)] + [(d, i) for i in range(pre_blocks, nblk)]

    def service_time(req):
        qid, d, stage, blk = req
        prompt = L[d] + (10 if (stage == 2 and cfg["family"] == "extract") else 0)
        hits = cache.lookup_insert(keys_for(d))
        computed = prompt - BLK * hits
        stats["computed"] += computed
        stats["prompt"] += prompt
        stats["requests"] += 1
        dec = 16 if (stage == 2 and cfg["family"] == "extract") else 0
        return a * computed + b + gam * dec

    def enqueue(req):
        queue.append(req)
        pending[req[0]] += 1

    def try_release():
        while len(s1_inflight) < cfg["inflight"]:
            dm = {}
            for qid in active:
                for bk in need[qid]:
                    if bk not in s1_inflight:
                        dm[bk] = dm.get(bk, 0) + 1
            if not dm:
                return
            bsel = next((pos[0] + k) % nb for k in range(nb) if (pos[0] + k) % nb in dm)
            pos[0] = (bsel + 1) % nb
            docs = range(bsel * cfg["block"], min(n, (bsel + 1) * cfg["block"]))
            qs = [qid for qid in active if bsel in need[qid]]
            cnt = 0
            for qid in qs:  # query-major within a block, as in the harness
                for d in docs:
                    if d in q_by[qid]["_docs"]:
                        enqueue((qid, d, 1, bsel))
                        cnt += 1
            for qid in qs:
                need[qid].discard(bsel)
            if cnt:
                s1_inflight[bsel] = cnt

    def check_done(qid, t):
        if qid in done_t or pending[qid] > 0:
            return
        if policy == "lazy":
            if qid not in need or need[qid]:
                return
        elif not released_all[qid]:
            return
        done_t[qid] = t
        if policy == "lazy" and qid in active:
            active.remove(qid)

    def start_next(t):
        if not busy[0] and queue:
            req = queue.popleft()
            busy[0] = True
            heapq.heappush(heap, (t + service_time(req), next(seq), "done", req))

    while heap:
        t, _, kind, payload = heapq.heappop(heap)
        if kind == "arrive":
            q = q_by[payload]
            if policy == "eager":
                for d in docs_of(q, n):
                    enqueue((payload, d, 1, None))
                released_all[payload] = True
            elif policy == "window":
                window_buf.append(payload)
            else:
                active.append(payload)
                need[payload] = {d // cfg["block"] for d in q["_docs"]}
                try_release()
        elif kind == "window":
            if window_buf:
                batch = list(window_buf)
                window_buf.clear()
                for d, qid in sorted((d, qid) for qid in batch for d in q_by[qid]["_docs"]):
                    enqueue((qid, d, 1, None))
                for qid in batch:
                    released_all[qid] = True
        else:  # a request completed
            busy[0] = False
            qid, d, stage, blk = payload
            pending[qid] -= 1
            if stage == 1 and q_by[qid]["p2"] is not None and rng.random() < p_pass:
                enqueue((qid, d, 2, blk))
            if policy == "lazy" and stage == 1:
                s1_inflight[blk] -= 1
                if s1_inflight[blk] == 0:
                    del s1_inflight[blk]
                    try_release()
            check_done(qid, t)
        for qid in q_by:
            check_done(qid, t)
        start_next(t)
    lat = sorted(done_t[qid] - q_by[qid]["arrival"] for qid in q_by)
    p95 = lat[min(len(lat) - 1, int(math.ceil(0.95 * len(lat))) - 1)]
    return dict(computed=stats["computed"], prompt=stats["prompt"], requests=stats["requests"],
                cached_frac=1 - stats["computed"] / stats["prompt"], makespan=max(done_t.values()),
                p95=p95, mean_latency=sum(lat) / len(lat))


POLICY_OF = {"P1_eager": "eager", "P1_eager_rerun": "eager", "P2_window30": "window", "P2_window10": "window",
             "P6_lazyscan": "lazy"}


def cfg_from_run(R):
    return dict(n=R["n_docs"], doc_tokens=R["doc_tokens"], shared_prefix=R["shared_prefix_tokens"],
                preamble=R["preamble_tokens"], kv_tokens=R["kv_tokens"], block=R["block"],
                family=R["family"], inflight=2, apc=R.get("apc", True))


def predict_run(cfg, trace, name, params, reps=3):
    pol = POLICY_OF[name]
    c = dict(cfg)
    if pol == "window":
        c["window"] = 30.0 if name.endswith("30") else 10.0
    outs = [simulate(c, [dict(q) for q in trace], pol, params, random.Random(r)) for r in range(reps)]
    return {k: sum(o[k] for o in outs) / reps for k in ("cached_frac", "makespan", "p95", "mean_latency", "computed")}


def fit(cal_runs):
    """alpha, beta: least squares of makespan on (computed tokens, requests) over the *eager*
    calibration runs only (saturated FCFS; no release waiting). Then kappa and a scale on alpha
    are chosen on a small grid by simulating every calibration policy with its own release
    schedule (makespan relative error + cached-fraction error). The pass rate is the
    calibration stage-1 pass rate."""
    rows = [(x["computed_prefill"], x["requests"], x["makespan"]) for R in cal_runs for x in R["runs"]
            if POLICY_OF.get(x["policy"]) == "eager"]
    sxx = sum(c * c for c, _, _ in rows); sxy = sum(c * r for c, r, _ in rows); syy = sum(r * r for _, r, _ in rows)
    sxt = sum(c * t for c, _, t in rows); syt = sum(r * t for _, r, t in rows)
    det = sxx * syy - sxy * sxy
    alpha0 = (sxt * syy - syt * sxy) / det if det else sxt / sxx
    beta0 = (syt * sxx - sxt * sxy) / det if det else 0.0
    if beta0 < 0 or alpha0 <= 0:  # collinear eager runs: non-negative fit through the origin
        beta0, alpha0 = 0.0, sxt / sxx
    pr = cal_runs[0]["_pass_rate"]
    best = None
    for kappa in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for sc in (0.85, 1.0, 1.15):
            params = dict(alpha=alpha0 * sc, beta=beta0, kappa=kappa, pass_rate=pr)
            err = 0.0
            for R in cal_runs:
                for run in R["runs"]:
                    p = predict_run(cfg_from_run(R), run["trace"], run["policy"], params, reps=1)
                    obs_c = run["cached_tokens"] / run["prompt_tokens"]
                    err += ((p["makespan"] - run["makespan"]) / run["makespan"]) ** 2 + (p["cached_frac"] - obs_c) ** 2
            if best is None or err < best[0]:
                best = (err, params)
    # Calibration diagnostic: predicted vs observed for every calibration policy.
    for R in cal_runs:
        for run in R["runs"]:
            p = predict_run(cfg_from_run(R), run["trace"], run["policy"], best[1], reps=1)
            print("calib", run["policy"], "makespan obs %.1f pred %.1f | cached obs %.3f pred %.3f | p95 obs %.1f pred %.1f" % (
                run["makespan"], p["makespan"], run["cached_tokens"] / run["prompt_tokens"], p["cached_frac"], run["p95"], p["p95"]),
                flush=True)
    return best[1], best[0]


def isolated_t1(cfg, params):
    """Model-predicted isolated single-stage, full-corpus query time (cold cache), as measured in the harness."""
    q = [dict(qid=0, arrival=0.0, p1=0, p2=None, lo=0, n=cfg["n"])]
    return simulate(dict(cfg), q, "eager", params, random.Random(0))["makespan"]


def make_trace(n_docs, m, seed, rate, overlap, family, n_preds=16, n_extract=4):
    """Identical to shared_scan_live.make_trace (same RNG calls in the same order)."""
    rng = random.Random(seed)
    t, qs = 0.0, []
    lo_f, hi_f = {"high": (0.5, 1.0), "mid": (0.3, 1.0), "low": (0.05, 0.2)}[overlap]
    for i in range(m):
        if i > 0:
            t += rng.expovariate(rate)
        frac = rng.uniform(lo_f, hi_f)
        lo = rng.randrange(n_docs)
        p1 = rng.randrange(n_preds)
        if family == "extract":
            p2 = rng.randrange(n_extract)
        else:
            p2 = rng.choice([p for p in range(n_preds) if p != p1]) if rng.random() < 0.5 else None
        qs.append(dict(qid=i, arrival=t, p1=p1, p2=p2, lo=lo, n=int(frac * n_docs)))
    return qs


def cell_cfg(src, n, blocks, family, block=32):
    """Token statistics for a held-out cell from a source run (calibration or smoke); documents
    are resampled cyclically when the cell has more documents than the source."""
    k = len(src["doc_tokens"])
    idx = [i % k for i in range(n)]
    return dict(n=n, doc_tokens=[src["doc_tokens"][i] for i in idx],
                shared_prefix=[src["shared_prefix_tokens"][i] for i in idx], preamble=src["preamble_tokens"],
                kv_tokens=blocks * 16, block=block, family=family, inflight=2, apc=True)


def predict_cell(cell, params, src, policies=("P1_eager", "P2_window30", "P6_lazyscan")):
    cfg = cell_cfg(src, cell["n"], cell["blocks"], cell["family"])
    cfg["apc"] = not cell.get("no_apc", False)
    # Predicted isolated single-query time sets the arrival rate, as in the harness.
    t1 = isolated_t1(cfg, params)
    mean_frac = {"high": 0.75, "mid": 0.65, "low": 0.125}[cell["overlap"]]
    out = {}
    for seed in cell["seeds"]:
        trace = make_trace(cell["n"], cell["m"], 1000 + seed, cell["load"] / (t1 * mean_frac), cell["overlap"], cell["family"])
        out[seed] = {nm: predict_run(cfg, trace, nm, params) for nm in policies}
    agg = {nm: {k: sum(out[s][nm][k] for s in out) / len(out) for k in ("cached_frac", "makespan", "p95")} for nm in policies}
    base = [nm for nm in policies if nm != "P6_lazyscan"]
    best_mk = min(base, key=lambda nm: agg[nm]["makespan"])
    best_p95 = min(base, key=lambda nm: agg[nm]["p95"])
    return dict(cell=cell, t1_pred=t1, per_policy=agg,
                makespan_gain=agg[best_mk]["makespan"] / agg["P6_lazyscan"]["makespan"],
                p95_gain=agg[best_p95]["p95"] / agg["P6_lazyscan"]["p95"],
                best_baseline_makespan=best_mk, best_baseline_p95=best_p95)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--calib", nargs="+", required=True)
    f.add_argument("--out", required=True)
    pr = sub.add_parser("predict")
    pr.add_argument("--params", required=True)
    pr.add_argument("--cells", required=True, help="JSON list of cells with a 'src' run path")
    pr.add_argument("--out", required=True)
    pr.add_argument("--results", required=True, help="directory holding the allowed source runs")
    a = ap.parse_args()
    if a.cmd == "fit":
        cal = []
        for fpath in a.calib:
            R = json.load(open(fpath))
            r0 = R["runs"][0]
            two = [q for q in r0["trace"] if q["p2"] is not None]
            s1 = sum(q["n"] for q in r0["trace"])
            R["_pass_rate"] = max(0.0, r0["requests"] - s1) / max(1, sum(q["n"] for q in two))
            cal.append(R)
        params, err = fit(cal)
        json.dump(dict(params=params, fit_error=err, calib=a.calib), open(a.out, "w"), indent=1)
        print(json.dumps(dict(params=params, fit_error=err)))
    else:
        params = json.load(open(a.params))["params"]
        cells = json.load(open(a.cells))
        preds = []
        import os
        allowed = {"arxiv": os.path.join(a.results, "C1_s0.json"), "scifact": os.path.join(a.results, "smoke_vllm.json")}
        for c in cells:
            assert os.path.abspath(c["src"]) == os.path.abspath(allowed[c["corpus"]]), \
                f"cell {c['name']}: token statistics must come from {allowed[c['corpus']]}"
            src = json.load(open(c["src"]))
            p = predict_cell(c, params, src)
            preds.append(p)
            print(c["name"], "makespan_gain", round(p["makespan_gain"], 2), "p95_gain", round(p["p95_gain"], 2),
                  {nm: round(v["cached_frac"], 3) for nm, v in p["per_policy"].items()}, flush=True)
        import datetime
        from zoneinfo import ZoneInfo  # fails closed if America/Detroit tz data is unavailable
        stamp = datetime.datetime.now(ZoneInfo("America/Detroit")).isoformat(timespec="seconds")
        json.dump(dict(params=params, predictions=preds, written_at=stamp,
                       sources=sorted({c["src"] for c in cells})), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
