#!/usr/bin/env python3
"""Concept figure (Figure 1) of the SharedScan operator: query ranges over a block-partitioned document table,
the circular scan releasing one block to every attached query, a late query attaching mid-scan, and the
block's query-independent document prefixes resident in the GPU KV cache. Not a data figure."""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

plt.rcParams.update({"font.size": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42})
QC = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    nb, x0, w, yb = 8, 0.6, 1.05, 3.55
    # Document table partitioned into scan blocks (clustering-key order).
    for k in range(nb):
        hot = k == 3
        ax.add_patch(Rectangle((x0 + k * w, yb), w * 0.94, 0.55, fc="#FCE3D3" if hot else "#F2F2F2",
                               ec="#D55E00" if hot else "#999999", lw=1.0 if hot else 0.6))
        ax.text(x0 + k * w + w * 0.47, yb + 0.27, f"$B_{k + 1}$", ha="center", va="center")
    ax.text(x0, yb - 0.3, "documents in clustering-key order, $b$ per scan block", fontsize=5.5, color="0.35")
    # Query ranges above the table (q4 arrives while the scan is at B4).
    ranges = [(0, 6), (2, 8), (1, 5), (3, 8)]
    for i, (lo, hi) in enumerate(ranges):
        y = 4.45 + i * 0.42
        ax.plot([x0 + lo * w + 0.05, x0 + hi * w - 0.1], [y, y], color=QC[i], lw=2.4, solid_capstyle="butt")
        ax.text(x0 - 0.08, y, f"$q_{i + 1}$", va="center", ha="right", color=QC[i], fontsize=6)
    ax.text(x0 + 8 * w, 4.45 + 3 * 0.42 + 0.3, "$q_4$ arrives mid-scan and attaches at $B_4$", ha="right",
            va="center", color=QC[3], fontsize=5.3)
    # Scan pointer and circular wrap-around.
    ax.add_patch(FancyArrowPatch((x0 + 3.47 * w, 3.0), (x0 + 3.47 * w, yb - 0.02), arrowstyle="-|>", mutation_scale=7,
                                 color="#D55E00", lw=1.0))
    ax.text(x0 + 3.47 * w, 2.85, "scan position $\\theta$: release $B_4$\nto all queries that need it", ha="center",
            va="top", fontsize=5.5, color="#D55E00")
    ax.add_patch(FancyArrowPatch((x0 + nb * w - 0.1, 6.55), (x0 + 0.05, 6.55), connectionstyle="arc3,rad=0.08",
                                 arrowstyle="-|>", mutation_scale=6, color="0.55", lw=0.6, ls="--"))
    ax.text(x0 + nb * w * 0.5, 7.05, "circular scan wraps around", ha="center", fontsize=5.5, color="0.4")
    # GPU KV cache with the released block's resident prefixes and the attached questions.
    ax.add_patch(FancyBboxPatch((0.5, 0.15), 9.0, 1.55, boxstyle="round,pad=0.05", fc="#FFF8F2", ec="#D55E00", lw=0.8))
    ax.text(0.65, 1.52, "GPU KV cache ($C$ blocks)", fontsize=6, color="#D55E00", va="center")
    for j in range(4):
        xx = 0.75 + j * 1.25
        ax.add_patch(Rectangle((xx, 0.35), 1.05, 0.55, fc="#FCE3D3", ec="#D55E00", lw=0.6))
        ax.text(xx + 0.52, 0.62, f"$\\langle\\pi, d_{{{j + 1}}}\\rangle$", ha="center", va="center", fontsize=6)
    ax.text(5.85, 0.62, "...", ha="center", va="center")
    ax.text(6.15, 0.95, "one prefix per document of $B_4$,\nreused by every attached $p^1_q$\nand stage-2 $p^2_q$;"
            " resident\nwhile $W_k \\leq C-1$", fontsize=5.2, va="center")
    for i in range(4):
        ax.add_patch(FancyArrowPatch((x0 + (3.2 + 0.18 * i) * w, yb - 0.05), (1.3 + i * 1.0, 0.95),
                                     arrowstyle="-|>", mutation_scale=5, color=QC[i], lw=0.6, alpha=0.8))
    fig.tight_layout(pad=0.1)
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, "fig_overview.pdf"))
    fig.savefig(os.path.join(a.out, "fig_overview.png"), dpi=200)


if __name__ == "__main__":
    main()
