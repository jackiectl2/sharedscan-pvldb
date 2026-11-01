#!/usr/bin/env python3
"""Freeze the inputs and model predictions of block B8 (refine-logs/EXPERIMENT_PLAN.md, B8) before any of its jobs runs.

All cells use Qwen3-8B, 1,600 KV blocks, nominal load 3, contiguous ranges (start redrawn with RNG seed 5000+s as in B6),
and the Qwen3-8B parameters of the confirmatory freeze. Descriptive; no pre-registered pass/fail test depends on B8.
  <corpus>_k1600_l3_dist     every query gets its own predicates (none repeats within a trace)
  <corpus>_k1600_l3_burst    balanced two-phase hyperexponential interarrivals, CV 3, same mean rate
  arxiv_k1600_l3_extract     stage 2 is a short extraction with decode, all queries (PubMed extraction prompts do not
                             share the filter prefix in the harness, so arXiv only); P is run without a decode cost, so
                             its values only size walltime and are not predictions to report
  <corpus>_k1600_l3_f1       B6 contiguous traces unchanged, adaptive against SharedScan with F=1 and F=2 at b=16 and
                             F=1 at b=32 (seeds 10-12); predicted baseline is eager only, as these cells have no window
  scale file: <corpus>_n<N>_k<K>_l3, arXiv N=5,000 and PubMed N=3,000, K=1,600 and K=10,000 (the capacity an L40S has
                             after the model weights): documents selected by the frozen rule with N in place of 1,000
                             (superset of the 1,000-document collections). Arrival rate follows the model's t1 for N
                             documents, so per-query work grows and the arrival rate falls relative to the 1,000-document
                             cells. Separate file because the harness keys corpora by name.
The extract cell is a new workload (new ranges and arrivals, every query has stage 2), not a decode-only ablation.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from zoneinfo import ZoneInfo

import b8_predicates as PX
import confirm_freeze as CF
import sscan_model as v1
import sweep_live_b5 as H
from b5_freeze import _predict, sha

SEEDS = [10, 11]
MEAN_FRAC = 0.75
POL = ["P1_eager", "P2_window30", "P2_adaptive", "P6_lazyscan_F1", "P6_lazyscan"]
POL_F1 = ["P1_eager", "P2_adaptive", "P6_lazyscan_b16", "P6_lazyscan_b16_F1", "P6_lazyscan_F1"]
MODELS = ["P", "M1"]
SCALE = {"arxiv": 5000, "pubmed": 3000}
SCALE_BLOCKS = [1600, 10000]
ARXIV_PREDICATES = list(H.PREDICATES)  # setup_corpus("arxiv") keeps whatever list is current


def setup(corpus):
    H.PREDICATES = list(ARXIV_PREDICATES)
    H.setup_corpus(corpus)


def contiguous_trace(n, m, s, rate, family="filter"):
    H.CFG["family"] = family
    tr = H.make_trace(n, m, 1000 + s, rate, "high")
    rng = random.Random(5000 + s)
    for q in tr:
        q["lo"] = rng.randrange(n - q["n"] + 1)
    assert all(q["lo"] + q["n"] <= n for q in tr)
    return tr


def distinct(tr, s, n_preds):
    order = list(range(n_preds))
    random.Random(7000 + s).shuffle(order)
    it = iter(order)
    for q in tr:
        q["p1"] = next(it)
        if q["p2"] is not None:
            q["p2"] = next(it)
    used = [q["p1"] for q in tr] + [q["p2"] for q in tr if q["p2"] is not None]
    assert len(used) == len(set(used))
    return tr


def bursty(tr, s, rate, cv=3.0):
    p = 0.5 * (1 + math.sqrt((cv * cv - 1) / (cv * cv + 1)))
    rng = random.Random(6000 + s)
    t = 0.0
    for i, q in enumerate(tr):
        if i > 0:
            t += rng.expovariate(2 * p * rate) if rng.random() < p else rng.expovariate(2 * (1 - p) * rate)
        q["arrival"] = t
    return tr


def run_predictions(cells, traces, corpora, params, workers):
    tasks = []
    for name, cell in cells.items():
        cfg = dict(CF.model_cfg(corpora[cell["corpus"]], cell["blocks"]), n=cell["n"], family=cell["family"])
        for s in cell["seeds"]:
            for pol in cell["policies"]:
                if pol == "P2_adaptive":
                    continue
                for m in (["P"] if cell["family"] == "extract" else MODELS):
                    tasks.append((m, name, s, pol, cfg, traces[cell["trace"]][str(s)], params))
    raw = {m: {} for m in MODELS}
    with ProcessPoolExecutor(workers) as ex:
        for m, name, s, pol, p in ex.map(_predict, tasks):
            raw[m].setdefault(name, {}).setdefault(str(s), {})[pol] = p
    preds = {}
    for m in MODELS:
        agg = {name: {pol: {k: sum(by[str(s)][pol][k] for s in cells[name]["seeds"]) / len(cells[name]["seeds"])
                            for k in ("cached_frac", "makespan", "p95", "mean_latency")}
                      for pol in by[str(cells[name]["seeds"][0])]} for name, by in raw[m].items()}
        summary = {}
        for name, A in agg.items():
            base = [p for p in A if not p.startswith("P6")]
            summary[name] = {}
            for var in [p for p in A if p.startswith("P6")]:
                g = min(A[x]["makespan"] for x in base) / A[var]["makespan"]
                summary[name][var] = dict(makespan_gain_vs_best_predicted_baseline=g, cls=CF.classify(g),
                                          p6_cached_frac=A[var]["cached_frac"])
                print(m, name, var, "gain %.3f cached %.3f" % (g, A[var]["cached_frac"]), flush=True)
        preds[m] = dict(per_seed=raw[m], per_cell=agg, summary=summary)
    est = {}
    for name, cell in cells.items():
        est[name] = []
        for s in cell["seeds"]:
            mods = [m for m in MODELS if name in raw[m]]
            est[name].append(sum(max(raw[m][name][str(s)]["P2_window30" if (pol == "P2_adaptive" and "P2_window30" in cell["policies"])
                                                          else ("P6_lazyscan_F1" if pol == "P2_adaptive" else pol)]["makespan"] for m in mods)
                                 for pol in cell["policies"]))
    return preds, est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-freeze", required=True)
    ap.add_argument("--confirm-sha", required=True)
    ap.add_argument("--b6-freeze", required=True)
    ap.add_argument("--b6-sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-scale", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scale-only", action="store_true", help="write only the scale file (the main file was frozen earlier)")
    a = ap.parse_args()
    assert sha(a.confirm_freeze) == a.confirm_sha
    assert sha(a.b6_freeze) == a.b6_sha
    F = json.load(open(a.confirm_freeze))
    B6 = json.load(open(a.b6_freeze))
    params = F["params_v1"]
    cells, traces = {}, {}
    n_extra = {"arxiv": len(PX.ARXIV_EXTRA), "pubmed": len(PX.PUBMED_EXTRA)}
    for c in F["corpora"]:
        base = F["cells"][f"{c}_k1600_l3"]
        rate = base["rate"]
        n, m = base["n"], base["m"]
        setup(c)
        common = dict(base, policies=POL, seeds=SEEDS, ranges="contiguous", block_id="B8")
        d, b = f"{c}_k1600_l3_dist", f"{c}_k1600_l3_burst"
        cells[d] = dict(common, trace=d, predicates="distinct",
                        predicate_list=list(H.PREDICATES) + {"arxiv": PX.ARXIV_EXTRA, "pubmed": PX.PUBMED_EXTRA}[c])
        cells[b] = dict(common, trace=b, arrivals="hyperexponential cv=3")
        traces[d] = {str(s): distinct(contiguous_trace(n, m, s, rate), s, 16 + n_extra[c]) for s in SEEDS}
        traces[b] = {str(s): bursty(contiguous_trace(n, m, s, rate), s, rate) for s in SEEDS}
        f1 = f"{c}_k1600_l3_f1"
        src = B6["cells"][f"{c}_k1600_l3_nowrap"]
        cells[f1] = dict(src, policies=POL_F1, trace=f1, block_id="B8", predicted_baseline="eager only")
        traces[f1] = B6["traces"][f"{c}_k1600_l3_nowrap"]
        if c == "arxiv":
            x = "arxiv_k1600_l3_extract"
            cells[x] = dict(common, trace=x, family="extract", prediction_status="walltime sizing only: P has no decode cost; not a reportable prediction")
            traces[x] = {str(s): contiguous_trace(n, m, s, rate, "extract") for s in SEEDS}
    H.CFG["family"] = "filter"
    if not a.scale_only:
        preds, est = run_predictions(cells, traces, F["corpora"], params, a.workers)
    here = os.path.dirname(os.path.abspath(__file__))
    code = {s: sha(os.path.join(here, s)) for s in ["sweep_live_b5.py", "b8_live.py", "b8_predicates.py", "b8_freeze.py", "b5_freeze.py", "confirm_freeze.py", "sscan_model.py"]}
    stamp = lambda: datetime.datetime.now(ZoneInfo("America/Detroit")).isoformat(timespec="seconds")
    meta = dict(b6_freeze_sha256=a.b6_sha, protocol="refine-logs/EXPERIMENT_PLAN.md (B8); prospective predictions, descriptive", confirm_freeze_sha256=a.confirm_sha,
                code_sha256=code, params_v1=params, models=F["models"], primary_model=F["primary_model"], model=F["model"],
                model_revision=F["model_revision"], tie_band=F["tie_band"], mean_frac=F["mean_frac"])
    if not a.scale_only:
        out = dict(meta, written_at=stamp(), corpora=F["corpora"], cells=cells, traces=traces, predictions=preds, predicted_policy_seconds_per_seed=est)
        json.dump(out, open(a.out, "w"))
        print("written", a.out, sha(a.out), flush=True)
    # Scale cells: same selection rule, first N documents; the arrival rate follows the model's t1 for N documents.
    import huggingface_hub
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    huggingface_hub.list_repo_files = lambda *x, **k: ["document/test-00000-of-00001.parquet"]  # offline: the cached file load_pubmed_docs lists
    tok = AutoTokenizer.from_pretrained(snapshot_download(F["model"], revision=F["model_revision"]))
    corpora, cells, traces = {}, {}, {}
    for c, N in SCALE.items():
        setup(c)
        docs = {"arxiv": H.load_docs, "pubmed": H.load_pubmed_docs}[c](N, 0)
        assert len(docs) == N
        small = {d["id"]: d for d in F["corpora"][c]["docs"]}
        byid = {d["id"]: d for d in docs}
        assert all(byid.get(i) == d for i, d in small.items()), "the 1,000-document collection is not a subset with equal content"
        pr = H.Prompts(tok, docs)
        doc_tok, shared, pre = H.token_stats(pr, N)
        src = F["corpora"][c]
        assert max(doc_tok) < F["cells"][f"{c}_k1600_l3"]["max_len"]
        corpora[c] = dict(src, docs=docs, doc_tokens=doc_tok, shared_prefix_tokens=shared, preamble_tokens=pre, corpus_prefix_tokens=sum(doc_tok),
                          selection_rule=f"as in the confirmatory freeze, with {N} documents in place of 1,000; contains those 1,000 unchanged (asserted)",
                          docs_sha256=hashlib.sha256(json.dumps(docs, sort_keys=True).encode()).hexdigest())
        base = F["cells"][f"{c}_k1600_l3"]
        for K in SCALE_BLOCKS:
            cfg = dict(CF.model_cfg(corpora[c], K), n=N)
            t1 = v1.isolated_t1(cfg, params)
            rate = base["load"] / (t1 * MEAN_FRAC)
            name = f"{c}_n{N}_k{K}_l3"
            cells[name] = dict(base, n=N, blocks=K, t1_hat=t1, rate=rate, policies=POL, seeds=SEEDS, trace=name, ranges="contiguous", block_id="B8")
            traces[name] = {str(s): contiguous_trace(N, base["m"], s, rate) for s in SEEDS}
    preds, est = run_predictions(cells, traces, corpora, params, a.workers)
    out = dict(meta, written_at=stamp(), corpora=corpora, cells=cells, traces=traces, predictions=preds, predicted_policy_seconds_per_seed=est)
    json.dump(out, open(a.out_scale, "w"))
    print("written", a.out_scale, sha(a.out_scale), flush=True)


if __name__ == "__main__":
    main()
