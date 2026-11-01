#!/usr/bin/env python3
"""Freeze block B8c before any of its jobs runs (refine-logs/EXPERIMENT_PLAN.md, B8c). Two seed extensions, written to two new
files; b8_freeze.json, b8b_freeze.json and b8_scale_freeze.json stay as frozen.
  b8c_scale_freeze.json  seeds 12 and 13 of the arXiv scale cells (N=5,000, K=1,600 and K=10,000). PubMed scale cells take
                         about 6 h per seed and are not extended, a choice made from runtime before these traces exist.
  b8c_freeze.json        seeds 14 and 15 of the distinct-predicate, bursty and arXiv extraction cells.
Cell definitions, arrival rates, trace construction, parameters and models are those of the existing freezes; only the seeds
differ. Seeds are the next integers, chosen before their traces were generated; none is dropped or chosen by realized CV or
load. Before writing, the frozen traces of the earlier seeds are regenerated and must match, so the generator is unchanged.
Usage: b8c_freeze.py --b8 <json> --b8-sha <sha> --b8b <json> --b8b-sha <sha> --scale <json> --scale-sha <sha>
       --out <b8c_freeze.json> --out-scale <b8c_scale_freeze.json> [--workers N]"""
import argparse
import datetime
import json
import os
from zoneinfo import ZoneInfo

import b8_freeze as B

MAIN_SEEDS = [14, 15]
SCALE_SEEDS = [12, 13]
SCALE_CELLS = ["arxiv_n5000_k1600_l3", "arxiv_n5000_k10000_l3"]
META = ("confirm_freeze_sha256", "b6_freeze_sha256", "params_v1", "models", "primary_model", "model", "model_revision", "tie_band", "mean_frac")


def canon(x):
    """The form a trace has after the JSON round trip of a freeze file."""
    return json.loads(json.dumps(x))


def main_traces(kind, cell, c, seeds, n_extra):
    n, m, rate = cell["n"], cell["m"], cell["rate"]
    if kind == "dist":
        return {str(s): B.distinct(B.contiguous_trace(n, m, s, rate), s, 16 + n_extra) for s in seeds}
    if kind == "burst":
        return {str(s): B.bursty(B.contiguous_trace(n, m, s, rate), s, rate) for s in seeds}
    return {str(s): B.contiguous_trace(n, m, s, rate, "extract") for s in seeds}


def main():
    ap = argparse.ArgumentParser()
    for name in ("b8", "b8b", "scale"):
        ap.add_argument(f"--{name}", required=True)
        ap.add_argument(f"--{name}-sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-scale", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    for path, want in ((a.b8, a.b8_sha), (a.b8b, a.b8b_sha), (a.scale, a.scale_sha)):
        assert B.sha(path) == want, path
    for path in (a.out, a.out_scale):
        assert not os.path.exists(path), f"{path} exists; refusing to overwrite"
    F, Fb, S = json.load(open(a.b8)), json.load(open(a.b8b)), json.load(open(a.scale))
    here = os.path.dirname(os.path.abspath(__file__))
    stamp = lambda: datetime.datetime.now(ZoneInfo("America/Detroit")).isoformat(timespec="seconds")
    n_extra = {"arxiv": len(B.PX.ARXIV_EXTRA), "pubmed": len(B.PX.PUBMED_EXTRA)}

    cells, traces = {}, {}
    scale_traces = {}
    for c in F["corpora"]:
        B.setup(c)
        for kind in ("dist", "burst", "extract"):
            name = f"{c}_k1600_l3_{kind}"
            if name not in F["cells"]:
                continue
            cell = dict(F["cells"][name], seeds=MAIN_SEEDS, block_id="B8c")
            cells[name] = cell
            for frozen, seeds in ((F, [10, 11]), (Fb, [12, 13])):
                assert canon(main_traces(kind, cell, c, seeds, n_extra[c])) == frozen["traces"][name], (name, seeds, "generator changed")
            traces[name] = main_traces(kind, cell, c, MAIN_SEEDS, n_extra[c])
        for name in SCALE_CELLS:
            cell = S["cells"][name]
            if cell["corpus"] != c:
                continue
            old = {str(s): B.contiguous_trace(cell["n"], cell["m"], s, cell["rate"]) for s in [10, 11]}
            assert canon(old) == S["traces"][name], (name, "generator changed")
            scale_traces[name] = {str(s): B.contiguous_trace(cell["n"], cell["m"], s, cell["rate"]) for s in SCALE_SEEDS}
    assert set(scale_traces) == set(SCALE_CELLS)

    B.H.CFG["family"] = "filter"
    code = dict(F["code_sha256"], **{"b8c_freeze.py": B.sha(os.path.join(here, "b8c_freeze.py")), "b8_freeze.py": B.sha(os.path.join(here, "b8_freeze.py"))})
    protocol = "refine-logs/EXPERIMENT_PLAN.md (B8c); prospective predictions, descriptive"

    preds, est = B.run_predictions(cells, traces, F["corpora"], F["params_v1"], a.workers)
    out = dict({k: F[k] for k in META}, b8_freeze_sha256=a.b8_sha, protocol=protocol, code_sha256=code, written_at=stamp(),
               corpora=F["corpora"], cells=cells, traces=traces, predictions=preds, predicted_policy_seconds_per_seed=est)
    json.dump(out, open(a.out, "w"))
    print("written", a.out, B.sha(a.out), flush=True)

    scale_cells = {n: dict(S["cells"][n], seeds=SCALE_SEEDS, block_id="B8c") for n in SCALE_CELLS}
    preds, est = B.run_predictions(scale_cells, scale_traces, S["corpora"], S["params_v1"], a.workers)
    out = dict({k: S[k] for k in META}, b8_scale_freeze_sha256=a.scale_sha, protocol=protocol, code_sha256=code, written_at=stamp(),
               corpora=S["corpora"], cells=scale_cells, traces=scale_traces, predictions=preds, predicted_policy_seconds_per_seed=est)
    json.dump(out, open(a.out_scale, "w"))
    print("written", a.out_scale, B.sha(a.out_scale), flush=True)


if __name__ == "__main__":
    main()
