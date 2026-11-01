#!/usr/bin/env python3
"""Summarize the same-configuration eager reruns (results/confirm_rerun) beside B1 (results/confirm).

Acceptance, per cell and seed: the rerun file has exactly the runs P1_eager, P1_eager_rerun, each with
60 completed queries, saved labels and the B1 trace of that cell and seed. Reported per seed:
  primary   within-job agreement of P1_eager_rerun with the rerun job's P1_eager (stage 1 over all
            pre-registered pairs, stage 2 over pairs passing stage 1 in both runs, identical results k/60);
  secondary cross-job agreement of the rerun job's P1_eager with B1's P1_eager (from saved labels);
  B1        the window's and SharedScan's agreement with B1's P1_eager, as recorded in B1.
The agreement function is a copy of confirm_live.agreement (filter family); this script reads JSON only.
Usage: confirm_rerun_summary.py <root> [--json out.json]
"""
import json
import os
import sys

CELLS = ["arxiv_k2400_l3", "pubmed_k2400_l3"]
SEEDS = [10, 11, 12]


def docs_of(q, n_docs):
    return [(q["lo"] + k) % n_docs for k in range(q["n"])]


def agreement(ref, lab, trace, n_docs):
    s1 = [(q["qid"], d, 1) for q in trace for d in docs_of(q, n_docs)]
    s2 = [(q["qid"], d, 2) for q in trace if q["p2"] is not None for d in docs_of(q, n_docs)
          if ref.get((q["qid"], d, 1)) == 1 and lab.get((q["qid"], d, 1)) == 1]

    def final(L, q):
        return {d for d in docs_of(q, n_docs)
                if L.get((q["qid"], d, 1)) == 1 and (q["p2"] is None or L.get((q["qid"], d, 2)) == 1)}
    same = sum(int(final(ref, q) == final(lab, q)) for q in trace)
    return dict(stage1=sum(ref.get(k) == lab.get(k) for k in s1) / len(s1), stage1_pairs=len(s1),
                stage2=(sum(ref.get(k) == lab.get(k) for k in s2) / len(s2)) if s2 else None, stage2_pairs=len(s2),
                identical=same, of=len(trace))


def labels(run):
    return {(a, b, c): v for a, b, c, v in run["labels"]}


def main():
    root = sys.argv[1]
    rows = []
    for cell in CELLS:
        for s in SEEDS:
            b1 = json.load(open(os.path.join(root, "results/confirm", f"{cell}_s{s}.json")))
            p = os.path.join(root, "results/confirm_rerun", f"{cell}_s{s}.json")
            if not os.path.exists(p):
                rows.append(dict(cell=cell, seed=s, status="missing"))
                continue
            rr = json.load(open(p))
            runs = {r["policy"]: r for r in rr["runs"]}
            b1runs = {r["policy"]: r for r in b1["runs"]}
            ok = (sorted(runs) == ["P1_eager", "P1_eager_rerun"] and len(rr["runs"]) == 2
                  and rr.get("policy_override") == ["P1_eager", "P1_eager_rerun"]
                  and all(r["completed"] == 60 and "labels" in r and r["trace"] == b1runs["P1_eager"]["trace"]
                          for r in rr["runs"]))
            if not ok:
                rows.append(dict(cell=cell, seed=s, status="incomplete"))
                continue
            trace, n = b1runs["P1_eager"]["trace"], b1["n_docs"]
            row = dict(cell=cell, seed=s, status="ok",
                       within=agreement(labels(runs["P1_eager"]), labels(runs["P1_eager_rerun"]), trace, n),
                       makespan=[round(runs[k]["makespan"], 1) for k in ("P1_eager", "P1_eager_rerun")])
            row["cross_job"] = (agreement(labels(b1runs["P1_eager"]), labels(runs["P1_eager"]), trace, n)
                                if "labels" in b1runs["P1_eager"] else None)
            row["b1"] = {k: dict(stage1=v["stage1_agreement"], stage2=v["stage2_agreement"],
                                 identical=round(v["queries_with_identical_result"] * 60))
                         for k, v in b1runs.items() if k != "P1_eager"}
            rows.append(row)
    for r in rows:
        print(json.dumps(r))
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
