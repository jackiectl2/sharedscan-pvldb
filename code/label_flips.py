#!/usr/bin/env python3
"""Descriptive analysis of label differences behind result-set agreement (reads saved labels only).

A label is per (query, document, stage), but the prompt depends only on (document, predicate) (confirm_live.py), so
labels are grouped by that pair; the stage only selects which of a query's predicates applies. For each 2,400-block
trace with an eager rerun this reports the distinct prompts whose label differs from the reference, for: SharedScan
and the window against B1's eager run; the rerun against its job's first eager run; that first run against B1's.
A prompt requested more than once in a run can get different labels; such prompts are counted separately and
treated as differing from the reference when their label sets differ. It also aggregates agreement ranges for B5-B7.
Usage: label_flips.py <root> [--json out.json]
"""
import glob
import json
import os
import sys
from collections import defaultdict

CELLS = ["arxiv_k2400_l3", "pubmed_k2400_l3"]
SEEDS = [10, 11, 12]


def prompt_labels(run):
    """(doc, predicate) -> set of labels; more than one value means the same prompt got different labels in the run."""
    pred = {q["qid"]: q for q in run["trace"]}
    out = defaultdict(set)
    for qid, doc, stage, v in run["labels"]:
        q = pred[qid]
        out[(doc, q["p1"] if stage == 1 else q["p2"])].add(v)
    return out


def diff(a, b):
    return {k for k in a.keys() & b.keys() if a[k] != b[k]}


def main():
    root = sys.argv[1]
    rows = []
    for c in CELLS:
        for s in SEEDS:
            b1 = {r["policy"]: r for r in json.load(open(f"{root}/results/confirm/{c}_s{s}.json"))["runs"]}
            rr = {r["policy"]: r for r in json.load(open(f"{root}/results/confirm_rerun/{c}_s{s}.json"))["runs"]}
            L = dict(e=prompt_labels(b1["P1_eager"]), w=prompt_labels(b1["P2_window30"]), s=prompt_labels(b1["P6_lazyscan"]),
                     e1=prompt_labels(rr["P1_eager"]), e2=prompt_labels(rr["P1_eager_rerun"]))
            d = dict(scan=diff(L["e"], L["s"]), window=diff(L["e"], L["w"]), rerun=diff(L["e1"], L["e2"]), cross=diff(L["e"], L["e1"]))
            eager_all = d["rerun"] | d["cross"]
            rows.append(dict(cell=c, seed=s, prompts=len(L["e"]), inconsistent_within_run={k: sum(len(v) > 1 for v in x.values()) for k, x in L.items()},
                             n={k: len(v) for k, v in d.items()}, scan_in_eager=len(d["scan"] & eager_all), window_in_eager=len(d["window"] & eager_all),
                             scan=sorted(map(list, d["scan"]))[:12], eager=sorted(map(list, eager_all))[:12]))
    agg = defaultdict(list)
    for blk in ("b5", "b6", "b7"):
        for f in sorted(glob.glob(f"{root}/results/{blk}/*_s1[0-2].json")):
            D = json.load(open(f))
            if "runs" not in D:
                continue
            for r in D["runs"]:
                if r["policy"] == "P1_eager":
                    continue
                fam = "scan" if r["policy"].startswith("P6") else r["policy"]
                agg[fam].append((blk, os.path.basename(f)[:-5], r["policy"], r["stage1_agreement"], r["stage2_agreement"], r["queries_with_identical_result"]))
    rng = {k: dict(n=len(v), s1=[min(x[3] for x in v), max(x[3] for x in v)], s2=[min(x[4] for x in v if x[4] is not None), max(x[4] for x in v if x[4] is not None)],
                   identical=[min(x[5] for x in v), max(x[5] for x in v)], worst=min(v, key=lambda x: x[5])[:3]) for k, v in agg.items()}
    cfg = defaultdict(list)
    for k, v in agg.items():
        for x in v:
            cfg[(k, x[1].rsplit("_s", 1)[0], x[2])].append(x[5])
    means = defaultdict(list)
    for (k, _, _), v in cfg.items():
        means[k].append(sum(v) / len(v))
    for k in rng:
        rng[k]["config_mean"] = [min(means[k]), max(means[k])]
    out = dict(rows=rows, ranges=rng)
    print(json.dumps(out, indent=0)[:9000])
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
