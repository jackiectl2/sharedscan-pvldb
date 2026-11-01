#!/usr/bin/env python3
"""Freeze the inputs and model predictions of block B6 (refine-logs/EXPERIMENT_PLAN.md, B6) before any of its
jobs is submitted.

B6 repeats the main comparison with contiguous, non-wrapping key ranges. The confirmatory traces draw a range start
uniformly over the collection and wrap around its end (two key intervals for most queries). B6 copies each frozen
<corpus>_k1600_l3 trace query by query (arrival time, range length, predicates) and redraws only the range start,
uniformly among the positions at which the whole range fits (independent RNG, seed 5000+s), so load is unchanged and
the comparison is paired.
Qwen3-8B, 1,600 KV blocks, load 3.0, seeds 10-12. Predictions of P (primary) and M1-M3 are frozen here.

  <corpus>_k1600_l3_nowrap: P1_eager, [P1_eager_rerun on arXiv: noise floor for result-set agreement], P2_window30,
       P2_adaptive (no model prediction), P6_lazyscan_b16, P6_lazyscan (b = 32).
"""
import argparse
import datetime
import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from zoneinfo import ZoneInfo

import confirm_freeze as CF
import sweep_live_b5 as H

SEEDS = [10, 11, 12]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def nowrap_trace(frozen, n_docs, seed):
    """Copy of a frozen confirmatory trace (qid, arrival, n, p1, p2 unchanged) with each range start redrawn uniformly
    among the positions at which the whole range fits, from an independent RNG, so every range is one contiguous key
    interval and the comparison with the wrapped trace is paired query by query."""
    rng = random.Random(seed)
    out = []
    for q in frozen:
        r = dict(q)
        r["lo"] = rng.randrange(n_docs - q["n"] + 1)
        out.append(r)
    return out


def sim_policy(name):
    """Map a policy name to (model policy, window seconds, block size or None, blocks in flight or None)."""
    if name.startswith("P1_eager"):
        return "eager", None, None, None
    if name.startswith("P2_window"):
        return "window", float(name[len("P2_window"):]), None, None
    if name.startswith("P6_lazyscan"):
        bs, fl = H.p6_params(name, 32)
        return "lazy", None, bs, fl
    raise ValueError(name)


def _predict(task):
    model, cell, seed, policy, cfg, trace, params = task
    sim, extra = CF.MODELS[model]
    pol, w, blk, fl = sim_policy(policy)
    c = dict(cfg)
    if w is not None:
        c["window"] = w
    if blk is not None:
        c["block"] = blk
    if fl is not None:
        c["inflight"] = fl
    outs = [sim(c, [dict(q) for q in trace], pol, dict(params, **extra), random.Random(r)) for r in range(3)]
    return model, cell, seed, policy, {k: sum(o[k] for o in outs) / 3 for k in ("cached_frac", "makespan", "p95", "mean_latency")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-freeze", required=True)
    ap.add_argument("--confirm-sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    assert sha(a.confirm_freeze) == a.confirm_sha, "confirmatory freeze file does not match its recorded sha256"
    F = json.load(open(a.confirm_freeze))
    params = F["params_v1"]
    cells, traces = {}, {}
    for corp in F["corpora"]:
        base = F["cells"][f"{corp}_k1600_l3"]
        name = f"{corp}_k1600_l3_nowrap"
        pols = ["P1_eager"] + (["P1_eager_rerun"] if corp == "arxiv" else []) + \
               ["P2_window30", "P2_adaptive", "P6_lazyscan_b16", "P6_lazyscan"]
        cells[name] = dict(base, policies=pols, seeds=SEEDS, trace=name, ranges="contiguous", block_id="B6")
        traces[name] = {str(s): nowrap_trace(F["traces"][base["trace"]][str(s)], base["n"], 5000 + s) for s in SEEDS}
        for s in SEEDS:
            src, new = F["traces"][base["trace"]][str(s)], traces[name][str(s)]
            assert len(src) == len(new) and all(q["lo"] + q["n"] <= base["n"] for q in new), "a range wraps"
            assert all({k: v for k, v in a.items() if k != "lo"} == {k: v for k, v in b.items() if k != "lo"}
                       for a, b in zip(src, new)), "B6 trace is not paired with the confirmatory trace"
    tasks = []
    for name, cell in cells.items():
        cfg = CF.model_cfg(F["corpora"][cell["corpus"]], cell["blocks"])
        cfg["apc"] = cell.get("apc", True)
        for s in cell["seeds"]:
            for pol in cell["policies"]:
                if pol == "P2_adaptive":  # no model prediction for adaptive batching (as in B2)
                    continue
                for m in CF.MODELS:
                    tasks.append((m, name, s, pol, cfg, traces[cell["trace"]][str(s)], params))
    raw = {m: {} for m in CF.MODELS}
    with ProcessPoolExecutor(a.workers) as ex:
        for m, name, s, pol, p in ex.map(_predict, tasks):
            raw[m].setdefault(name, {}).setdefault(str(s), {})[pol] = p
    preds = {}
    for m in CF.MODELS:
        agg = {name: {pol: {k: sum(by_seed[str(s)][pol][k] for s in cells[name]["seeds"]) / len(cells[name]["seeds"])
                            for k in ("cached_frac", "makespan", "p95", "mean_latency")}
                      for pol in cells[name]["policies"] if pol != "P2_adaptive"}
               for name, by_seed in raw[m].items()}
        summary = {}
        for name, A in agg.items():
            base = [p for p in A if not p.startswith("P6")]
            summary[name] = {}
            for var in [p for p in A if p.startswith("P6")]:
                g = min(A[x]["makespan"] for x in base) / A[var]["makespan"]
                summary[name][var] = dict(makespan_gain_vs_best_predicted_baseline=g, cls=CF.classify(g),
                                          p95_gain=min(A[x]["p95"] for x in base) / A[var]["p95"], p6_cached_frac=A[var]["cached_frac"])
                print(m, name, var, "gain %.3f %s p6 cached %.3f" % (g, CF.classify(g), A[var]["cached_frac"]), flush=True)
        preds[m] = dict(per_seed=raw[m], per_cell=agg, summary=summary)
    # Runtime estimate only (walltime): adaptive batching is estimated by the 30 s window's prediction.
    est = {name: [sum(max(raw[m][name][str(s)]["P2_window30" if pol == "P2_adaptive" else pol]["makespan"] for m in CF.MODELS)
                      for pol in cell["policies"]) for s in cell["seeds"]] for name, cell in cells.items()}
    here = os.path.dirname(os.path.abspath(__file__))
    stamp = datetime.datetime.now(ZoneInfo("America/Detroit")).isoformat(timespec="seconds")
    out = dict(protocol="refine-logs/EXPERIMENT_PLAN.md (B6); prospective predictions, descriptive",
               written_at=stamp, confirm_freeze_sha256=a.confirm_sha,
               code_sha256={s: sha(os.path.join(here, s)) for s in
                            ["sweep_live_b5.py", "b6_freeze.py", "confirm_freeze.py", "sscan_model.py", "sscan_model_v2.py"]},
               params_v1=params, models=F["models"], primary_model=F["primary_model"], model=F["model"],
               model_revision=F["model_revision"], tie_band=F["tie_band"], mean_frac=F["mean_frac"],
               corpora=F["corpora"], cells=cells, traces=traces, predictions=preds,
               predicted_policy_seconds_per_seed=est)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print("written_at", stamp, "sha256", sha(a.out), flush=True)


if __name__ == "__main__":
    main()
