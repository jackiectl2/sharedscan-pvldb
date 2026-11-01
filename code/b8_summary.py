#!/usr/bin/env python3
"""Summarize block B8 (results/b8) against its frozen predictions. Per cell, over the seeds with a complete run: policy
means (makespan, p95, cached fraction), the gain of each SharedScan variant over the best of eager and the 30 s window
(eager only in the _f1 cells), the ratios to adaptive batching, the prediction of P, realized arrival rate and CV,
realized load (scaled warm-up), first-stage pass rate and agreement with the job's eager run. Reads JSON only.
Usage: b8_summary.py <root> [--json out.json]"""
import glob
import json
import os
import statistics as st
import sys


def rng2(values):
    """Range of stage-2 agreement over seeds; the runner reports None when a run has no comparable pairs."""
    v = [x for x in values if x is not None]
    return [min(v), max(v)] if v else None


def main():
    root = sys.argv[1]
    F = {}
    for fn in ("b8_freeze.json", "b8b_freeze.json", "b8_scale_freeze.json", "b8c_freeze.json", "b8c_scale_freeze.json"):
        p = f"{root}/results/b8/{fn}"
        if os.path.exists(p):
            d = json.load(open(p))
            for c, spec in d["cells"].items():
                e = F.setdefault(c, dict(spec=dict(spec, seeds=[]), traces={}, per_seed={}, batches=[]))
                assert e["spec"]["policies"] == spec["policies"] and e["spec"]["m"] == spec["m"], c
                e["spec"]["seeds"] += spec["seeds"]
                e["batches"].append(list(spec["seeds"]))
                e["traces"].update(d["traces"][spec["trace"]])
                e["per_seed"].update(d["predictions"]["P"]["per_seed"].get(c, {}))
    out, unfinished = {}, {}
    for cell, E in F.items():
        spec, traces, pseed = E["spec"], E["traces"], E["per_seed"]
        runs, skipped = {}, []
        for s in spec["seeds"]:
            f = f"{root}/results/b8/{cell}_s{s}.json"
            if not os.path.exists(f):
                skipped.append(s); continue
            D = json.load(open(f))
            if [r["policy"] for r in D["runs"]] != spec["policies"] or any(r["completed"] != spec["m"] for r in D["runs"]):
                skipped.append(s); continue
            assert D["runs"][0]["trace"] == traces[str(s)], (cell, s, "trace differs from the freeze")
            runs[s] = (D, {r["policy"]: r for r in D["runs"]})
        unfinished[cell] = skipped
        if not runs:
            continue
        m = lambda p, k: st.mean(R[p][k] for _, R in runs.values())
        pm = lambda p, k: st.mean(pseed[str(s)][p][k] for s in runs)
        pol = spec["policies"]
        base = [p for p in pol if p in ("P1_eager", "P2_window30")]
        best = min(base, key=lambda p: m(p, "makespan"))
        row = dict(seeds=sorted(runs), skipped=skipped, best_baseline=best,
                   policies={p: dict(makespan=m(p, "makespan"), p95=m(p, "p95"), mean_latency=m(p, "mean_latency"),
                                     cached=sum(R[p]["cached_tokens"] for _, R in runs.values()) / sum(R[p]["prompt_tokens"] for _, R in runs.values()),
                                     stage1=[min(R[p]["stage1_agreement"] for _, R in runs.values()), max(R[p]["stage1_agreement"] for _, R in runs.values())],
                                     stage2=rng2([R[p]["stage2_agreement"] for _, R in runs.values()]),
                                     stage2_pairs=[R[p]["stage2_pairs"] for _, R in runs.values()],
                                     identical=[round(R[p]["queries_with_identical_result"] * spec["m"]) for _, R in runs.values()]) for p in pol},
                   per_seed_makespan={s: {p: R[p]["makespan"] for p in pol} for s, (_, R) in runs.items()})
        row["gain"] = {}
        for v in [p for p in pol if p.startswith("P6")]:
            g = dict(vs_best=m(best, "makespan") / m(v, "makespan"), vs_eager=m("P1_eager", "makespan") / m(v, "makespan"),
                     p95_vs_best=m(best, "p95") / m(v, "p95"),
                     vs_adaptive=m("P2_adaptive", "makespan") / m(v, "makespan"), p95_vs_adaptive=m("P2_adaptive", "p95") / m(v, "p95"),
                     per_seed_vs_best={s: R[best]["makespan"] / R[v]["makespan"] for s, (_, R) in runs.items()},
                     per_seed_vs_adaptive={s: R["P2_adaptive"]["makespan"] / R[v]["makespan"] for s, (_, R) in runs.items()},
                     per_seed_p95_vs_adaptive={s: R["P2_adaptive"]["p95"] / R[v]["p95"] for s, (_, R) in runs.items()})
            if spec["family"] == "filter":
                # The frozen prediction over the seeds that completed: mean policy makespans, better predicted baseline.
                g.update(pred=min(pm(b, "makespan") for b in base) / pm(v, "makespan"), pred_cached=pm(v, "cached_frac"))
            row["gain"][v] = g
        real = []
        for s, (D, R) in runs.items():
            tr = traces[str(s)]
            gaps = [b["arrival"] - a["arrival"] for a, b in zip(tr, tr[1:])]
            r0 = R["P1_eager"]
            real.append(dict(seed=s, rate=len(gaps) / tr[-1]["arrival"], cv=st.pstdev(gaps) / st.mean(gaps),
                             realized_load=len(gaps) / tr[-1]["arrival"] * D["calibration_single_query_s"] * st.mean(q["n"] for q in tr) / D["n_docs"],
                             pass_rate=(r0["requests"] - sum(q["n"] for q in tr)) / max(1, sum(q["n"] for q in tr if q["p2"] is not None))))
        row["realized"] = real
        # Expansion batches: seeds frozen together, with the ratio of mean makespans and p95 within the batch. Descriptive.
        row["batches"] = {}
        for seeds in E["batches"]:
            done = [s for s in seeds if s in runs]
            if not done:
                continue
            bm = lambda p, k: st.mean(runs[s][1][p][k] for s in done)
            bbest = min(base, key=lambda p: bm(p, "makespan"))
            row["batches"]["-".join(map(str, seeds))] = dict(
                seeds=done, unfinished=[s for s in seeds if s not in runs], best_baseline=bbest,
                policies={p: dict(makespan=bm(p, "makespan"), p95=bm(p, "p95")) for p in pol},
                gain={v: dict(vs_best=bm(bbest, "makespan") / bm(v, "makespan"), vs_adaptive=bm("P2_adaptive", "makespan") / bm(v, "makespan"),
                              p95_vs_adaptive=bm("P2_adaptive", "p95") / bm(v, "p95")) for v in row["gain"]})
        out[cell] = row
    print(json.dumps(out, indent=0))
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        json.dump(out, open(path, "w"), indent=1)
        json.dump({c: s for c, s in unfinished.items() if s}, open(os.path.join(os.path.dirname(path), "unfinished.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
