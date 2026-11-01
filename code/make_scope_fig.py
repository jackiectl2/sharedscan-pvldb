#!/usr/bin/env python3
"""Scope figure: every trace of the scope cells, SharedScan (b=32, F=2) against the better of eager and the 30 s window and
against adaptive batching, marked by expansion batch; bars mark the pooled mean ratio. Reads results/b8/summary.json. A small
derived file with the plotted values is written next to the PDF. Usage: make_scope_fig.py <summary.json> <out.pdf>"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
ROWS = [("arxiv_k1600_l3_dist", "arXiv, distinct"), ("pubmed_k1600_l3_dist", "PubMed, distinct"), ("arxiv_k1600_l3_burst", "arXiv, hyperexp."),
        ("pubmed_k1600_l3_burst", "PubMed, hyperexp."), ("arxiv_k1600_l3_extract", "arXiv, extraction"), ("arxiv_n5000_k1600_l3", "arXiv 5k, K1.6k"),
        ("arxiv_n5000_k10000_l3", "arXiv 5k, K10k"), ("pubmed_n3000_k1600_l3", "PubMed 3k, K1.6k"), ("pubmed_n3000_k10000_l3", "PubMed 3k, K10k")]
# Secondary encoding by marker shape; first batch of each cell is seeds 10-11.
BATCH = {"10-11": ("#0072B2", "o", "Seeds 10, 11"), "12-13": ("#D55E00", "s", "Seeds 12, 13"), "14-15": ("#009E73", "^", "Seeds 14, 15")}
PANELS = [("per_seed_vs_best", "vs_best", "Makespan vs. better baseline"), ("per_seed_vs_adaptive", "vs_adaptive", "Makespan vs. adaptive"),
          ("per_seed_p95_vs_adaptive", "p95_vs_adaptive", "p95 vs. adaptive")]


def main():
    S = json.load(open(sys.argv[1]))
    rows = [(c, l) for c, l in ROWS if c in S]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 0.19 * len(rows) + 0.75), sharey=True)
    data = {}
    for ax, (key, mean_key, title) in zip(axes, PANELS):
        ax.axvline(1, color="#888888", lw=0.7, zorder=1)
        for y, (c, _) in enumerate(rows):
            g = S[c]["gain"]["P6_lazyscan"]
            ax.plot([g[mean_key]] * 2, [y - 0.32, y + 0.32], color="#222222", lw=1.2, zorder=2)
            for label, b in S[c]["batches"].items():
                col, mk, _ = BATCH[label]
                xs = [g[key][str(s)] for s in b["seeds"]]
                ax.scatter(xs, [y] * len(xs), s=13, color=col, marker=mk, edgecolors="white", linewidths=0.4, zorder=3)
                data.setdefault(c, {}).setdefault(key, {})[label] = dict(zip(map(str, b["seeds"]), xs))
        ax.set_title(title, loc="left")
        ax.set_xlabel("Baseline / SharedScan")
        ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(rows)), [l for _, l in rows])
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], ls="", marker=mk, color=col, markeredgecolor="white", markersize=4.5, label=l) for col, mk, l in BATCH.values()]
    handles.append(plt.Line2D([], [], color="#222222", lw=1.2, label="Ratio of means"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.8)
    fig.savefig(sys.argv[2])
    json.dump(data, open(sys.argv[2].rsplit(".", 1)[0] + "_data.json", "w"), indent=1)


if __name__ == "__main__":
    main()
