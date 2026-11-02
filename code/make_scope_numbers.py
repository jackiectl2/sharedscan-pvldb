#!/usr/bin/env python3
"""Numbers and one sentence for the scope prose of paper/sections/evaluation.tex, generated from results/b8/summary.json and
rerun-independent fields so late results need only a re-run. Also checks the qualitative claims the prose makes and prints
CLAIM FAILED for any that no longer holds; those need a change of wording.
Usage: make_scope_numbers.py <summary.json> <out.tex>"""
import json
import sys

CELLS = {"AxDist": "arxiv_k1600_l3_dist", "PmDist": "pubmed_k1600_l3_dist", "AxBurst": "arxiv_k1600_l3_burst", "PmBurst": "pubmed_k1600_l3_burst",
         "AxExtract": "arxiv_k1600_l3_extract", "AxSmall": "arxiv_n5000_k1600_l3", "AxLarge": "arxiv_n5000_k10000_l3",
         "PmSmall": "pubmed_n3000_k1600_l3", "PmLarge": "pubmed_n3000_k10000_l3"}
F2, F1 = "P6_lazyscan", "P6_lazyscan_F1"
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def rng(vals, fmt="{:.2f}"):
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return lo if lo == hi else f"{lo}--{hi}"


def main():
    S = json.load(open(sys.argv[1]))
    M = {}
    for k, c in CELLS.items():
        if c not in S:
            continue
        r, g = S[c], S[c]["gain"][F2]
        M[k + "Tr"] = WORDS[len(r["seeds"])]
        M[k + "Gain"] = f"{g['vs_best']:.2f}"
        M[k + "Ad"] = f"{g['vs_adaptive']:.2f}"
        M[k + "PAd"] = f"{g['p95_vs_adaptive']:.2f}"
        if "pred" in g:
            M[k + "Pred"] = f"{g['pred']:.2f}"
        M[k + "GainR"] = rng(g["per_seed_vs_best"].values())
        M[k + "AdR"] = rng(g["per_seed_vs_adaptive"].values())
        M[k + "PAdR"] = rng(g["per_seed_p95_vs_adaptive"].values())
        M[k + "PF"] = f"{r['policies'][F2]['p95']:.0f}"
        M[k + "PO"] = f"{r['policies'][F1]['p95']:.0f}"
        M[k + "Pass"] = rng([x["pass_rate"] * 100 for x in r["realized"]], "{:.0f}")
        M[k + "Cached"] = f"{r['policies'][F2]['cached'] * 100:.0f}"
    b = [x for c in ("AxBurst", "PmBurst") for x in S[CELLS[c]]["realized"]]
    M["BurstCV"] = rng([x["cv"] for x in b])
    M["BurstLoad"] = rng([x["realized_load"] for x in b], "{:.1f}")
    x = S[CELLS["AxExtract"]]["policies"]
    for k, p in (("Sys", F2), ("One", F1), ("Ad", "P2_adaptive"), ("Win", "P2_window30")):
        M["Ext" + k] = rng([v * 100 for v in x[p]["stage2"]], "{:.1f}")
        M["Ext" + k + "Id"] = rng(x[p]["identical"], "{:d}")
    M["ExtPairs"] = rng([v for p in (F2, F1, "P2_adaptive", "P2_window30") for v in x[p]["stage2_pairs"]], "{:,d}")
    sc = [S[CELLS[k]]["policies"] for k in ("AxSmall", "AxLarge", "PmSmall", "PmLarge") if CELLS[k] in S]
    for k, p in (("Sys", F2), ("Ad", "P2_adaptive"), ("Win", "P2_window30")):
        M["Scale" + k + "Id"] = rng([v for q in sc for v in q[p]["identical"]], "{:d}")
    cells1k = ["arxiv_k1600_l3_dist", "pubmed_k1600_l3_dist", "arxiv_k1600_l3_burst", "pubmed_k1600_l3_burst", "arxiv_k1600_l3_extract"]
    pct = [(S[c]["policies"][F1]["makespan"] / S[c]["policies"][F2]["makespan"] - 1) * 100 for c in cells1k if c in S]
    for c in ("arxiv_k1600_l3_f1", "pubmed_k1600_l3_f1"):
        q = S[c]["policies"]
        pct.append((q["P6_lazyscan_b16_F1"]["makespan"] / q["P6_lazyscan_b16"]["makespan"] - 1) * 100)
    M["OneLo"], M["OneHi"] = f"{min(pct):.0f}", f"{max(pct):.0f}"

    s = []
    if CELLS["PmLarge"] in S:
        parts = []
        for k, blocks in (("PmSmall", "1,600"), ("PmLarge", "10,000")):
            if CELLS[k] in S:
                parts.append((k, blocks))
        s.append("On PubMed, with " + " and ".join(f"{M[k + 'Tr']} trace{'' if M[k + 'Tr'] == 'one' else 's'} at {b}" for k, b in parts) + " blocks, \\sys lowers makespan relative to the window by "
                 + " and ".join(f"{M[k + 'Gain']}$\\times$ (predicted {M[k + 'Pred']}$\\times$)" for k, _ in parts)
                 + ". Relative to adaptive batching, makespan is " + " and ".join(f"{M[k + 'Ad']}$\\times$" for k, _ in parts)
                 + " lower and p95 latency " + " and ".join(f"{M[k + 'PAd']}$\\times$" for k, _ in parts) + " lower.")
    if CELLS["PmSmall"] not in S:
        s.append("The PubMed runs at 1,600 blocks did not finish in time.")
    M["PmScaleSentence"] = " ".join(s)

    M["AxSeedHi"] = str(max(int(x) for c in ("AxDist", "AxBurst", "AxExtract") for x in S[CELLS[c]]["seeds"]))
    # Budgets as a share of the reusable prefix footprint (Eq. ratio, shared prefix tokens / 16): arXiv 76,513 KV blocks.
    M["KvLo"] = "1" if CELLS["PmSmall"] in S else "2"
    M["KvText"] = "2--13\\% of the arXiv prefixes and " + ("1--8\\%" if CELLS["PmSmall"] in S else "8\\%") + " of the PubMed prefixes"

    def check(ok, msg):
        if not ok:
            print("CLAIM FAILED:", msg)
    lower = [c for c in cells1k if S[c]["policies"][F1]["p95"] < S[c]["policies"][F2]["p95"]]
    check(len(lower) == len(cells1k), f"F=1 lowers p95 in all five b=32 1,000-document cells: {lower}")
    for c in cells1k:
        more = sum(S[c]["policies"][F1]["identical"]) > sum(S[c]["policies"][F2]["identical"])
        check(more == (c == "arxiv_k1600_l3_dist"), f"F=1 matches eager on more queries than F=2 only for distinct arXiv predicates; {c}")
    for k, c in CELLS.items():
        if c in S:
            g = S[c]["gain"][F2]
            check(min(g["per_seed_vs_adaptive"].values()) > 1, f"SharedScan beats adaptive makespan on every trace of {c}")
            check(min(g["per_seed_p95_vs_adaptive"].values()) > 1, f"SharedScan beats adaptive p95 on every trace of {c}")
            check(min(g["per_seed_vs_best"].values()) > 1, f"SharedScan beats the best baseline on every trace of {c}")
    x2 = S[CELLS["AxExtract"]]["policies"]
    check(min(x2[F2]["stage2"]) < min(x2["P2_adaptive"]["stage2"]) and max(x2[F2]["stage2"]) > max(x2["P2_adaptive"]["stage2"]), "SharedScan extraction agreement straddles adaptive")
    with open(sys.argv[2], "w") as f:
        f.write("% Generated by figures/make_scope_numbers.py; do not edit by hand.\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\Sc{k}}}{{{v}}}\n")


if __name__ == "__main__":
    main()
