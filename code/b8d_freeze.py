#!/usr/bin/env python3
"""Freeze block B8d before any of its jobs runs: seeds 16 and 17 of the arXiv distinct-predicate, bursty and extraction cells,
written to a new file. Definitions are those of b8_freeze.json; earlier traces are regenerated and must match. Chosen because
these jobs finish before the deadline; seeds are the next integers, none dropped or chosen by realized CV or load.
Usage: b8d_freeze.py --b8 <json> --b8-sha <sha> --b8b <json> --b8b-sha <sha> --out <json> [--workers N]"""
import argparse
import datetime
import json
import os
from zoneinfo import ZoneInfo

import b8_freeze as B
from b8c_freeze import META, canon, main_traces

SEEDS = [16, 17]


def main():
    ap = argparse.ArgumentParser()
    for name in ("b8", "b8b"):
        ap.add_argument(f"--{name}", required=True)
        ap.add_argument(f"--{name}-sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    for path, want in ((a.b8, a.b8_sha), (a.b8b, a.b8b_sha)):
        assert B.sha(path) == want, path
    assert not os.path.exists(a.out), f"{a.out} exists; refusing to overwrite"
    F, Fb = json.load(open(a.b8)), json.load(open(a.b8b))
    here = os.path.dirname(os.path.abspath(__file__))
    n_extra = len(B.PX.ARXIV_EXTRA)
    B.setup("arxiv")
    cells, traces = {}, {}
    for kind in ("dist", "burst", "extract"):
        name = f"arxiv_k1600_l3_{kind}"
        cell = dict(F["cells"][name], seeds=SEEDS, block_id="B8d")
        cells[name] = cell
        for frozen, seeds in ((F, [10, 11]), (Fb, [12, 13])):
            assert canon(main_traces(kind, cell, "arxiv", seeds, n_extra)) == frozen["traces"][name], (name, seeds, "generator changed")
        traces[name] = main_traces(kind, cell, "arxiv", SEEDS, n_extra)
    B.H.CFG["family"] = "filter"
    code = dict(F["code_sha256"], **{"b8d_freeze.py": B.sha(os.path.join(here, "b8d_freeze.py")), "b8_freeze.py": B.sha(os.path.join(here, "b8_freeze.py"))})
    preds, est = B.run_predictions(cells, traces, {"arxiv": F["corpora"]["arxiv"]}, F["params_v1"], a.workers)
    stamp = datetime.datetime.now(ZoneInfo("America/Detroit")).isoformat(timespec="seconds")
    out = dict({k: F[k] for k in META}, b8_freeze_sha256=a.b8_sha, protocol="refine-logs/EXPERIMENT_PLAN.md (B8d); prospective predictions, descriptive",
               code_sha256=code, written_at=stamp, corpora={"arxiv": F["corpora"]["arxiv"]}, cells=cells, traces=traces, predictions=preds, predicted_policy_seconds_per_seed=est)
    json.dump(out, open(a.out, "w"))
    print("written", a.out, B.sha(a.out), flush=True)


if __name__ == "__main__":
    main()
