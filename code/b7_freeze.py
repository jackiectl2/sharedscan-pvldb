#!/usr/bin/env python3
"""Freeze the inputs and model predictions of block B7 step 2 (refine-logs/EXPERIMENT_PLAN.md, B7): a second model,
Qwen3-4B-Instruct-2507, before any of its jobs is submitted.

Parameters come from the B7 calibration session (results/b7/calib_q4b_arxiv_s0.json), measured exactly as the
Qwen3-8B parameters were: alpha is the least-squares makespan per computed prompt token of the two eager runs, through
the origin when the fit with a per-request term is collinear or negative (beta = 0); the pass rate is the first run's
stage-2 requests per document of two-predicate queries (sscan_model.fit). Unlike sscan_model.fit, no scale on alpha is
searched (for Qwen3-8B the search selected 1.0); kappa = 0.7 is carried over unchanged for the diagnostic models M1/M2.

Corpora are the frozen confirmatory documents, re-tokenized with the Qwen3-4B tokenizer and chat template (the harness
asserts these statistics at run time). Cells <corpus>_k1600_l3_q4b: 1,600 KV blocks, load 3.0 relative to Qwen3-4B
(rate = 3 / (t1_hat * 0.75), t1_hat from the model with the new parameters), seeds 10-11; traces from the harness's
make_trace (seed 1000+s) with each range start redrawn so that every range is one contiguous interval (as in B6,
independent RNG seed 5000+s). Policies: P1_eager, P2_window30, P2_adaptive (no prediction), P6_lazyscan_b16, P6_lazyscan.
Descriptive; no pre-registered pass/fail test depends on B7.
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
import sscan_model as v1
import sweep_live_b5 as H
from b5_freeze import _predict, sha

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SEEDS = [10, 11]
MEAN_FRAC = 0.75
POLICIES = ["P1_eager", "P2_window30", "P2_adaptive", "P6_lazyscan_b16", "P6_lazyscan"]


def measured_params(cal, kappa):
    """alpha, beta from the eager runs of the calibration session (as sscan_model.fit, without the scale search);
    pass rate as sscan_model's fit command computes it."""
    rows = [(x["computed_prefill"], x["requests"], x["makespan"]) for x in cal["runs"] if x["policy"].startswith("P1_eager")]
    assert len(rows) == 2, "expected the two eager runs of the calibration session"
    sxx = sum(c * c for c, _, _ in rows); sxy = sum(c * r for c, r, _ in rows); syy = sum(r * r for _, r, _ in rows)
    sxt = sum(c * t for c, _, t in rows); syt = sum(r * t for _, r, t in rows)
    det = sxx * syy - sxy * sxy
    alpha = (sxt * syy - syt * sxy) / det if det else sxt / sxx
    beta = (syt * sxx - sxt * sxy) / det if det else 0.0
    if beta < 0 or alpha <= 0:
        beta, alpha = 0.0, sxt / sxx
    r0 = cal["runs"][0]
    assert r0["policy"] == "P1_eager"
    two = [q for q in r0["trace"] if q["p2"] is not None]
    s1 = sum(q["n"] for q in r0["trace"])
    pass_rate = max(0.0, r0["requests"] - s1) / max(1, sum(q["n"] for q in two))
    return dict(alpha=alpha, beta=beta, kappa=kappa, pass_rate=pass_rate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-freeze", required=True)
    ap.add_argument("--confirm-sha", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    assert sha(a.confirm_freeze) == a.confirm_sha, "confirmatory freeze file does not match its recorded sha256"
    F = json.load(open(a.confirm_freeze))
    cal = json.load(open(a.calib))
    assert cal["model"] == MODEL and cal["corpus"] == "arxiv" and cal["apc"], "unexpected calibration session"
    params = measured_params(cal, F["params_v1"]["kappa"])
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path = snapshot_download(MODEL)
    revision = os.path.basename(path.rstrip("/"))
    assert revision == cal["model_revision"], "calibration ran on a different snapshot"
    tok = AutoTokenizer.from_pretrained(path)
    corpora = {}
    for c, corp in F["corpora"].items():
        H.setup_corpus(c)
        pr = H.Prompts(tok, corp["docs"])
        doc_tok, shared, pre = H.token_stats(pr, len(corp["docs"]))
        if c == "arxiv":
            assert (doc_tok, shared, pre) == (cal["doc_tokens"], cal["shared_prefix_tokens"], cal["preamble_tokens"]), \
                "arXiv token statistics differ from the calibration session"
        corpora[c] = dict(corp, doc_tokens=doc_tok, shared_prefix_tokens=shared, preamble_tokens=pre,
                          corpus_prefix_tokens=sum(doc_tok), example_prompt=tok.decode(pr.single(0, 0)))
        print(c, "prompt tokens mean %.1f (Qwen3-8B %.1f)" % (sum(doc_tok) / len(doc_tok), sum(corp["doc_tokens"]) / len(doc_tok)),
              "shared prefix mean %.1f" % (sum(shared) / len(shared)), flush=True)
    cells, traces = {}, {}
    for c in F["corpora"]:
        base = F["cells"][f"{c}_k1600_l3"]
        name = f"{c}_k1600_l3_q4b"
        assert max(corpora[c]["doc_tokens"]) < base["max_len"]
        cfg = CF.model_cfg(corpora[c], base["blocks"])
        t1_hat = v1.isolated_t1(cfg, params)
        rate = base["load"] / (t1_hat * MEAN_FRAC)
        cells[name] = dict(base, t1_hat=t1_hat, rate=rate, policies=POLICIES, seeds=SEEDS, trace=name,
                           ranges="contiguous", block_id="B7")
        H.setup_corpus(c)
        H.CFG["family"] = base["family"]
        rng = {s: random.Random(5000 + s) for s in SEEDS}
        traces[name] = {}
        for s in SEEDS:
            tr = H.make_trace(base["n"], base["m"], 1000 + s, rate, "high")
            for q in tr:
                q["lo"] = rng[s].randrange(base["n"] - q["n"] + 1)
            assert all(q["lo"] + q["n"] <= base["n"] for q in tr)
            traces[name][str(s)] = tr
    tasks = []
    for name, cell in cells.items():
        cfg = CF.model_cfg(corpora[cell["corpus"]], cell["blocks"])
        for s in SEEDS:
            for pol in POLICIES:
                if pol == "P2_adaptive":
                    continue
                for m in CF.MODELS:
                    tasks.append((m, name, s, pol, cfg, traces[name][str(s)], params))
    raw = {m: {} for m in CF.MODELS}
    with ProcessPoolExecutor(a.workers) as ex:
        for m, name, s, pol, p in ex.map(_predict, tasks):
            raw[m].setdefault(name, {}).setdefault(str(s), {})[pol] = p
    preds = {}
    for m in CF.MODELS:
        agg = {name: {pol: {k: sum(by_seed[str(s)][pol][k] for s in SEEDS) / len(SEEDS)
                            for k in ("cached_frac", "makespan", "p95", "mean_latency")}
                      for pol in POLICIES if pol != "P2_adaptive"} for name, by_seed in raw[m].items()}
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
    est = {name: [sum(max(raw[m][name][str(s)]["P2_window30" if pol == "P2_adaptive" else pol]["makespan"] for m in CF.MODELS)
                      for pol in POLICIES) for s in SEEDS] for name in cells}
    here = os.path.dirname(os.path.abspath(__file__))
    stamp = datetime.datetime.now(ZoneInfo("America/Detroit")).isoformat(timespec="seconds")
    out = dict(protocol="refine-logs/EXPERIMENT_PLAN.md (B7 step 2); prospective predictions, descriptive",
               written_at=stamp, confirm_freeze_sha256=a.confirm_sha, calibration_file=os.path.abspath(a.calib),
               calibration_sha256=sha(a.calib),
               code_sha256={s: sha(os.path.join(here, s)) for s in
                            ["sweep_live_b5.py", "b7_freeze.py", "b5_freeze.py", "confirm_freeze.py", "sscan_model.py", "sscan_model_v2.py"]},
               params_v1=params, params_note="Qwen3-4B parameters measured from the B7 calibration session",
               models=F["models"], primary_model=F["primary_model"], model=MODEL, model_revision=revision,
               tie_band=F["tie_band"], mean_frac=F["mean_frac"], corpora=corpora, cells=cells, traces=traces,
               predictions=preds, predicted_policy_seconds_per_seed=est)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print("params", json.dumps(params), "written_at", stamp, "sha256", sha(a.out), flush=True)


if __name__ == "__main__":
    main()
