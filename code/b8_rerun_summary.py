#!/usr/bin/env python3
"""Acceptance gate and summary for the B8 eager reruns (results/b8/rerun). The harness publishes its JSON after each policy,
so a crash can leave a valid, incomplete file; a file enters only with both policies in order, all queries completed, the
frozen trace, labels present and the override recorded. Agreement of the rerun is against the first eager run of the same
job; string equality follows the harness (lower-cased, whitespace-collapsed). Usage: b8_rerun_summary.py <root> [--json out]"""
import glob
import json
import os
import sys

POLICIES = ["P1_eager", "P1_eager_rerun"]


def main():
    root = sys.argv[1]
    F = json.load(open(f"{root}/results/b8/b8_freeze.json"))
    out = {}
    for f in sorted(glob.glob(f"{root}/results/b8/rerun/*_s*.json")):
        D = json.load(open(f))
        cell, seed = os.path.basename(f)[:-5].rsplit("_s", 1)
        spec, reasons = F["cells"][cell], []
        trace = F["traces"][spec["trace"]][seed]
        runs = D["runs"]
        if [r["policy"] for r in runs] != POLICIES:
            reasons.append("policies")
        if D.get("policy_override") != POLICIES:
            reasons.append("override not recorded")
        for r in runs:
            if r["completed"] != spec["m"] or str(r["seed"]) != seed or r["trace"] != trace or not r.get("labels"):
                reasons.append(f"{r['policy']} incomplete or wrong trace")
        row = dict(accepted=not reasons, reasons=reasons, code_verified_in_job="code_sha256_verified" in D)
        if not reasons:
            r = runs[1]
            row.update(stage1=r["stage1_agreement"], stage1_pairs=r["stage1_pairs"], stage2=r["stage2_agreement"], stage2_pairs=r["stage2_pairs"],
                       identical=round(r["queries_with_identical_result"] * spec["m"]), m=spec["m"],
                       makespan=[x["makespan"] for x in runs])
        out[f"{cell}_s{seed}"] = row
    print(json.dumps(out, indent=1))
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
