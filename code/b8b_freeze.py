#!/usr/bin/env python3
"""Freeze block B8b before any of its jobs runs: seeds 12 and 13 of the B8 distinct-predicate, bursty and arXiv extraction
cells (refine-logs/EXPERIMENT_PLAN.md, B8b). Cell definitions, trace construction, parameters and models are those of
b8_freeze.py, imported rather than copied; only the seeds differ, and b8_freeze.json stays as frozen. Seeds are the next two
integers, chosen before their traces were generated; no seed is dropped or chosen by its realized CV or load.
Usage: b8b_freeze.py --b8-freeze <b8_freeze.json> --b8-sha <sha> --out <b8b_freeze.json> [--workers N]"""
import argparse
import datetime
import json
import os
from zoneinfo import ZoneInfo

import b8_freeze as B

SEEDS = [12, 13]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b8-freeze", required=True)
    ap.add_argument("--b8-sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    assert B.sha(a.b8_freeze) == a.b8_sha
    assert not os.path.exists(a.out), f"{a.out} exists; refusing to overwrite"
    F = json.load(open(a.b8_freeze))
    params = F["params_v1"]
    cells, traces = {}, {}
    n_extra = {"arxiv": len(B.PX.ARXIV_EXTRA), "pubmed": len(B.PX.PUBMED_EXTRA)}
    for c in F["corpora"]:
        B.setup(c)
        for kind in ("dist", "burst", "extract"):
            name = f"{c}_k1600_l3_{kind}"
            if name not in F["cells"]:
                continue
            cell = dict(F["cells"][name], seeds=SEEDS, block_id="B8b")
            n, m, rate = cell["n"], cell["m"], cell["rate"]
            cells[name] = cell
            if kind == "dist":
                traces[name] = {str(s): B.distinct(B.contiguous_trace(n, m, s, rate), s, 16 + n_extra[c]) for s in SEEDS}
            elif kind == "burst":
                traces[name] = {str(s): B.bursty(B.contiguous_trace(n, m, s, rate), s, rate) for s in SEEDS}
            else:
                traces[name] = {str(s): B.contiguous_trace(n, m, s, rate, "extract") for s in SEEDS}
    B.H.CFG["family"] = "filter"
    preds, est = B.run_predictions(cells, traces, F["corpora"], params, a.workers)
    here = os.path.dirname(os.path.abspath(__file__))
    code = dict(F["code_sha256"], **{"b8b_freeze.py": B.sha(os.path.join(here, "b8b_freeze.py")), "b8_freeze.py": B.sha(os.path.join(here, "b8_freeze.py"))})
    meta = {k: F[k] for k in ("confirm_freeze_sha256", "b6_freeze_sha256", "params_v1", "models", "primary_model", "model", "model_revision", "tie_band", "mean_frac")}
    out = dict(meta, b8_freeze_sha256=a.b8_sha, protocol="refine-logs/EXPERIMENT_PLAN.md (B8b); prospective predictions, descriptive", code_sha256=code,
               written_at=datetime.datetime.now(ZoneInfo("America/Detroit")).isoformat(timespec="seconds"),
               corpora=F["corpora"], cells=cells, traces=traces, predictions=preds, predicted_policy_seconds_per_seed=est)
    json.dump(out, open(a.out, "w"))
    print("written", a.out, B.sha(a.out), flush=True)


if __name__ == "__main__":
    main()
