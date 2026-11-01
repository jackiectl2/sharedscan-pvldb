#!/usr/bin/env python3
"""Freeze the inputs and model predictions of follow-up blocks B2 and B3 (refine-logs/EXPERIMENT_PLAN.md)
before any of their jobs is submitted.

The confirmatory freeze file is read (sha256 asserted) and its corpora, model revision and traces are
carried over unchanged: B2 and B3 reuse the frozen load-3.0 traces of <corpus>_k1600_l3, so their runs pair
with the confirmatory ones. Predictions use the frozen models (P primary, M1-M3 diagnostics) through
confirm_freeze.MODELS. B2 and B3 are reported descriptively; no pre-registered test depends on them.

  B2 (batching baselines): <corpus>_batch_k1600_l3, policies P1_eager, P2_adaptive, P2_window10,
     P2_window60, P6_lazyscan (b = 32); seeds 10-12. P2_adaptive has no model prediction.
  B3 (boundary cells): pubmed_k<C>_l3_b<b> for (C, b) in (1200, 32), (1200, 16), (1600, 48), (3200, 64);
     policies P1_eager, P2_window30, P6_lazyscan_b<b>; seeds 10-11.
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

B3_CELLS = [(1200, 32), (1200, 16), (1600, 48), (3200, 64)]
B2_POLICIES = ["P1_eager", "P2_adaptive", "P2_window10", "P2_window60", "P6_lazyscan"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sim_policy(name):
    """Map a policy name to (model policy, window seconds, block size or None)."""
    if name.startswith("P1_eager"):
        return "eager", None, None
    if name.startswith("P2_window"):
        return "window", float(name[len("P2_window"):]), None
    if name == "P6_lazyscan":
        return "lazy", None, 32
    if name.startswith("P6_lazyscan_b"):
        return "lazy", None, int(name[len("P6_lazyscan_b"):])
    raise ValueError(name)


def _predict(task):
    model, cell, seed, policy, cfg, trace, params = task
    sim, extra = CF.MODELS[model]
    pol, w, blk = sim_policy(policy)
    c = dict(cfg)
    if w is not None:
        c["window"] = w
    if blk is not None:
        c["block"] = blk
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
    cells = {}
    for c in F["corpora"]:
        base = dict(F["cells"][f"{c}_k1600_l3"])
        cells[f"{c}_batch_k1600_l3"] = dict(base, policies=B2_POLICIES, seeds=[10, 11, 12], trace=f"{c}_k1600_l3", block_id="B2")
    for C, b in B3_CELLS:
        base = dict(F["cells"]["pubmed_k1600_l3"])
        cells[f"pubmed_k{C}_l3_b{b}"] = dict(base, blocks=C, p6_block=b, policies=["P1_eager", "P2_window30", f"P6_lazyscan_b{b}"],
                                            seeds=[10, 11], trace="pubmed_k1600_l3", block_id="B3")
    traces = {f"{c}_k1600_l3": F["traces"][f"{c}_k1600_l3"] for c in F["corpora"]}
    tasks = []
    for name, cell in cells.items():
        cfg = CF.model_cfg(F["corpora"][cell["corpus"]], cell["blocks"])
        for s in cell["seeds"]:
            for pol in cell["policies"]:
                if pol == "P2_adaptive":
                    continue
                for m in CF.MODELS:
                    tasks.append((m, name, s, pol, cfg, traces[cell["trace"]][str(s)], params))
    raw = {m: {} for m in CF.MODELS}
    with ProcessPoolExecutor(a.workers) as ex:
        for m, name, s, pol, p in ex.map(_predict, tasks):
            raw[m].setdefault(name, {}).setdefault(str(s), {})[pol] = p
    preds = {}
    for m in CF.MODELS:
        agg = {}
        for name, by_seed in raw[m].items():
            seeds = cells[name]["seeds"]
            pols = list(next(iter(by_seed.values())))
            agg[name] = {pol: {k: sum(by_seed[str(s)][pol][k] for s in seeds) / len(seeds)
                               for k in ("cached_frac", "makespan", "p95", "mean_latency")} for pol in pols}
        summary = {}
        for name, A in agg.items():
            var = [p for p in cells[name]["policies"] if p.startswith("P6")][0]
            base = [p for p in A if not p.startswith("P6")]
            g = min(A[x]["makespan"] for x in base) / A[var]["makespan"]
            summary[name] = dict(variant=var, makespan_gain_vs_best_predicted_baseline=g, cls=CF.classify(g),
                                 p95_gain=min(A[x]["p95"] for x in base) / A[var]["p95"], p6_cached_frac=A[var]["cached_frac"])
            print(m, name, var, "gain %.3f %s p6 cached %.3f" % (g, CF.classify(g), A[var]["cached_frac"]), flush=True)
        preds[m] = dict(per_seed=raw[m], per_cell=agg, summary=summary)
    est = {}
    for name, cell in cells.items():
        per = []
        for s in cell["seeds"]:
            tot = 0.0
            for pol in cell["policies"]:
                pp = "P2_window10" if pol == "P2_adaptive" else pol  # estimate only: adaptive ~ short window
                tot += max(raw[m][name][str(s)][pp]["makespan"] for m in CF.MODELS)
            per.append(tot)
        est[name] = per
    here = os.path.dirname(os.path.abspath(__file__))
    stamp = datetime.datetime.now(ZoneInfo("America/Detroit")).isoformat(timespec="seconds")
    out = dict(protocol="refine-logs/EXPERIMENT_PLAN.md (B2, B3); descriptive, no pre-registered test",
               written_at=stamp, confirm_freeze_sha256=a.confirm_sha,
               code_sha256={s: sha(os.path.join(here, s)) for s in
                            ["sweep_live.py", "sweep_freeze.py", "confirm_freeze.py", "sscan_model.py", "sscan_model_v2.py"]},
               params_v1=params, models=F["models"], primary_model=F["primary_model"], model=F["model"],
               model_revision=F["model_revision"], tie_band=F["tie_band"], mean_frac=F["mean_frac"],
               corpora=F["corpora"], cells=cells, traces=traces, predictions=preds,
               predicted_policy_seconds_per_seed=est)
    json.dump(out, open(a.out, "w"))
    print("written_at", stamp, "sha256", sha(a.out), flush=True)


if __name__ == "__main__":
    main()
