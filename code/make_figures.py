#!/usr/bin/env python3
"""Data figures for the SharedScan paper, generated from real result files and the frozen predictions.

Inputs (cluster paths under the project scratch root, passed with --root):
  results/confirm/*.json   confirmatory round B1 (frozen, CONFIRM_PREREG.md)
  results/sweep/*.json     follow-up blocks B2 (batching baselines) and B3 (boundary cells)
  results/confirm/confirm_freeze.json, results/sweep/sweep_freeze.json   frozen model predictions
Outputs (--out): fig_main.pdf, fig_regime.pdf, fig_predict.pdf, fig_quality.pdf, figure_data.json (small derived
data behind every panel). Cells whose result files are missing or incomplete are skipped and listed in
figure_data.json["skipped"].
"""
import argparse
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

# Okabe-Ito palette; one color per execution policy, used consistently across figures.
POLICY_STYLE = {
    "P1_eager": ("vLLM eager + APC", "#999999", "o"),
    "P1_eager_rerun": ("vLLM eager (rerun)", "#BBBBBB", "o"),
    "P2_window10": ("Window 10 s", "#56B4E9", "v"),
    "P2_window30": ("Window 30 s", "#0072B2", "s"),
    "P2_window60": ("Window 60 s", "#003F5C", "^"),
    "P2_adaptive": ("Adaptive batch", "#009E73", "D"),
    "SGL_lpm": ("SGLang LPM", "#E69F00", "P"),
    "SGL_P6": ("SharedScan on SGLang", "#CC79A7", "X"),
    "P6_lazyscan_b16": ("SharedScan b=16", "#F0A3C8", "<"),
    "P6_lazyscan": ("SharedScan b=32", "#D55E00", "*"),
    "P6_lazyscan_b64": ("SharedScan b=64", "#8C2D04", ">"),
}
CORPUS_LABEL = {"arxiv": "arXiv abstracts", "pubmed": "PubMed excerpts"}
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def load_cell(root, sub, cell, policies, seeds, m=None):
    """Return {seed: {policy: run}} if every seed has every policy, completed, else None."""
    out = {}
    for s in seeds:
        p = os.path.join(root, "results", sub, f"{cell}_s{s}.json")
        try:
            R = json.load(open(p))
        except (OSError, ValueError):
            return None
        # Same completeness rule as the scorer: the run list must be exactly the frozen policy list, in order.
        if [r["policy"] for r in R["runs"]] != list(policies):
            return None
        if any(r.get("seed", s) != s or (m is not None and r.get("completed") != m) for r in R["runs"]):
            return None
        runs = {r["policy"]: r for r in R["runs"]}
        out[s] = runs
    return out


def cached(r):
    return r["cached_tokens"] / r["prompt_tokens"]


def footprint_ratio(corp, b, C):
    sp = [math.ceil(x / 16.0) for x in corp["shared_prefix_tokens"]]
    return max(sum(sp[i:i + b]) for i in range(0, len(sp), b)) / (C - 1)


def fig_main(root, F, S, out, data, skipped):
    """Per-seed makespan and p95 for every policy at 1,600 KV blocks and load 3.0, one row per corpus."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6), gridspec_kw=dict(width_ratios=[1, 1]))
    for row, corp in enumerate(("pubmed", "arxiv")):
        series = []  # (key, [(seed, makespan, p95, p50)])
        b1 = load_cell(root, "confirm", f"{corp}_k1600_l3", F["cells"][f"{corp}_k1600_l3"]["policies"], [10, 11, 12])
        if b1:
            for pol in ("P1_eager", "P2_window30", "P6_lazyscan_b16", "P6_lazyscan", "P6_lazyscan_b64"):
                series.append((pol, [(s, b1[s][pol]["makespan"], b1[s][pol]["p95"], b1[s][pol]["p50"]) for s in b1]))
        else:
            skipped.append(f"{corp}_k1600_l3")
        sg = load_cell(root, "confirm", f"{corp}_sgl_k1600_l3", F["cells"][f"{corp}_sgl_k1600_l3"]["policies"], [10, 11, 12])
        if sg:
            series.append(("SGL_lpm", [(s, sg[s]["P1_eager"]["makespan"], sg[s]["P1_eager"]["p95"], sg[s]["P1_eager"]["p50"]) for s in sg]))
            series.append(("SGL_P6", [(s, sg[s]["P6_lazyscan"]["makespan"], sg[s]["P6_lazyscan"]["p95"], sg[s]["P6_lazyscan"]["p50"]) for s in sg]))
        else:
            skipped.append(f"{corp}_sgl_k1600_l3")
        if S is not None:
            b2 = load_cell(root, "sweep", f"{corp}_batch_k1600_l3", S["cells"][f"{corp}_batch_k1600_l3"]["policies"], [10, 11, 12])
            if b2:
                for pol in ("P2_window10", "P2_window60", "P2_adaptive"):
                    series.append((pol, [(s, b2[s][pol]["makespan"], b2[s][pol]["p95"], b2[s][pol]["p50"]) for s in b2]))
            else:
                skipped.append(f"{corp}_batch_k1600_l3")
        data.setdefault("main", {})[corp] = {k: v for k, v in series}
        order = sorted(series, key=lambda kv: np.mean([x[1] for x in kv[1]]))  # fastest policy on top
        for col, (idx, label, logx) in enumerate(((1, "Makespan (s)", False), (2, "Query latency, p50 (open) to p95 (filled), s", True))):
            ax = axes[row, col]
            for y, (key, pts) in enumerate(order):
                name, color, marker = POLICY_STYLE[key]
                for j, pt in enumerate(pts):
                    dy = (j - 1) * 0.18
                    if col == 0:
                        ax.plot(pt[1], y + dy, marker=marker, color=color, ms=4, ls="none")
                    else:
                        ax.plot([pt[3], pt[2]], [y + dy, y + dy], color=color, lw=0.8)
                        ax.plot(pt[3], y + dy, marker=marker, mfc="white", mec=color, ms=3.5, ls="none")
                        ax.plot(pt[2], y + dy, marker=marker, color=color, ms=4, ls="none")
                if col == 0:
                    ax.plot([np.mean([p[1] for p in pts])] * 2, [y - 0.35, y + 0.35], color=color, lw=1.2)
            ax.set_ylim(len(order) - 0.5, -0.5)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([POLICY_STYLE[k][0] for k, _ in order] if col == 0 else [])
            if logx:
                ax.set_xscale("log")
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel(label)
            ax.grid(axis="x", lw=0.3, alpha=0.5)
            ax.set_title(f"{CORPUS_LABEL[corp]}: {'makespan' if col == 0 else 'latency'} (3 seeds; bar = mean)"
                         if col == 0 else f"{CORPUS_LABEL[corp]}: per-query latency (3 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_main.pdf"))
    plt.close(fig)


def gather_p6_cells(root, F, S, skipped):
    """Every (corpus, C, load, b) cell with SharedScan and both baselines: observed per-seed gain, cached fraction,
    and frozen predictions of P and M1."""
    cells = []
    specs = []
    for name, c in F["cells"].items():
        if c["backend"] != "vllm":
            continue
        for v in [p for p in c["policies"] if p.startswith("P6")]:
            specs.append(("confirm", F, name, c, v))
    if S is not None:
        for name, c in S["cells"].items():
            if name.endswith("_batch_k1600_l3"):
                continue
            v = [p for p in c["policies"] if p.startswith("P6")][0]
            specs.append(("sweep", S, name, c, v))
    for sub, FF, name, c, v in specs:
        runs = load_cell(root, sub, name, c["policies"], c["seeds"], c.get("m"))
        if runs is None:
            skipped.append(name)
            continue
        base = ("P1_eager", "P2_window30")
        gains = [min(runs[s][b]["makespan"] for b in base) / runs[s][v]["makespan"] for s in runs]
        # Cell-level gain as in the preregistered scoring: best baseline by mean makespan over mean SharedScan makespan.
        gain_cell = min(np.mean([runs[s][b]["makespan"] for s in runs]) for b in base) / np.mean([runs[s][v]["makespan"] for s in runs])
        cf = [cached(runs[s][v]) for s in runs]
        bsize = 32 if v == "P6_lazyscan" else int(v.split("_b")[-1])
        pred = {}
        for m in ("P", "M1"):
            A = FF["predictions"][m]["per_cell"][name]
            pred[m] = dict(gain=min(A[b]["makespan"] for b in base) / A[v]["makespan"], cached=A[v]["cached_frac"])
        cells.append(dict(cell=name, sub=sub, corpus=c["corpus"], C=c["blocks"], load=c["load"], b=bsize, variant=v,
                          ratio=footprint_ratio(FF["corpora"][c["corpus"]], bsize, c["blocks"]),
                          gains=gains, gain=float(gain_cell), cached=cf, cached_mean=float(np.mean(cf)), pred=pred,
                          prospective=True))
    return cells


def fig_regime(cells, out, data):
    """Observed SharedScan gain and cached fraction against the single-block footprint ratio, with the frozen
    primary model's predictions; load-3.0 cells only (load-1.0 cells are ties by design and shown in the table)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    hi = [c for c in cells if c["load"] == 3.0]
    data["regime"] = hi
    for ax, key, ylab in ((axes[0], "gain", "Makespan gain vs best baseline"), (axes[1], "cached", "SharedScan cached-token fraction")):
        ax.axvline(1.0, color="k", lw=0.6, ls="--")
        ax.text(1.02, 0.97, "footprint = KV capacity", transform=ax.get_xaxis_transform(), fontsize=5.5, va="top")
        if key == "gain":
            ax.axhspan(0.95, 1.05, color="0.9", lw=0)
        for c in hi:
            color = "#D55E00" if c["corpus"] == "pubmed" else "#0072B2"
            mk = {16: "<", 32: "*", 48: "h", 64: ">"}[c["b"]]
            obs = c["gains"] if key == "gain" else c["cached"]
            ax.plot([c["ratio"]] * len(obs), obs, ls="none", marker=mk, color=color, ms=3.5, alpha=0.75)
            pr = c["pred"]["P"][key if key == "gain" else "cached"]
            ax.plot(c["ratio"], pr, ls="none", marker=mk, mfc="none", mec="k", ms=6, mew=0.7)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Max single-block prefix footprint / (KV blocks - 1)")
        ax.set_ylabel(ylab)
        ax.grid(lw=0.3, alpha=0.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker="o", color="#D55E00", ms=4, label="PubMed (observed, per seed)"),
               Line2D([], [], ls="none", marker="o", color="#0072B2", ms=4, label="arXiv (observed, per seed)"),
               Line2D([], [], ls="none", marker="o", mfc="none", mec="k", ms=6, label="Frozen prediction (model P)")]
    present = {c["b"] for c in hi}
    handles += [Line2D([], [], ls="none", marker=m, color="0.3", ms=4, label=f"b={b}")
                for b, m in ((16, "<"), (32, "*"), (48, "h"), (64, ">")) if b in present]
    fig.legend(handles=handles, loc="lower center", frameon=False, ncol=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, "fig_regime.pdf"))
    plt.close(fig)


def fig_predict(cells, out, data):
    """Win/tie/loss classification of every prospective cell by the primary model P and the diagnostic M1
    (fitted kappa), and predicted against observed gains."""
    def cls(g):
        return "win" if g > 1.05 else ("loss" if g < 0.95 else "tie")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    rows = sorted(cells, key=lambda c: (c["corpus"], c["load"], c["C"], c["b"]))
    labels = [f"{c['corpus'][:2]} C={c['C']} L={c['load']:g} b={c['b']}" for c in rows]
    colmap = {"win": "#009E73", "tie": "#E69F00", "loss": "#CC79A7"}
    for i, c in enumerate(rows):
        o = cls(c["gain"])
        for j, m in enumerate(("observed", "P", "M1")):
            k = o if m == "observed" else cls(c["pred"][m]["gain"])
            ok = (k == o)
            ax.add_patch(plt.Rectangle((j, i), 0.95, 0.9, color=colmap[k], alpha=1.0 if (m == "observed" or ok) else 0.35))
            ax.text(j + 0.47, i + 0.45, k if m == "observed" else (k + ("" if ok else " x")), ha="center", va="center", fontsize=5.5)
    ax.set_xlim(0, 3)
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([0.47, 1.47, 2.47])
    ax.set_xticklabels(["Observed", "P (primary)", "M1 (fitted kappa)"])
    ax.set_yticks([i + 0.45 for i in range(len(rows))])
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_title("Makespan winner class (x = misclassified)")
    ax.spines[:].set_visible(False)
    ax = axes[1]
    lim = [0.4, 2.2]
    ax.plot(lim, lim, color="k", lw=0.6)
    ax.fill_between(lim, [x * 0.8 for x in lim], [x * 1.2 for x in lim], color="0.9", lw=0)
    for c in cells:
        for m, mk, col in (("P", "o", "#D55E00"), ("M1", "x", "#0072B2")):
            ax.plot(c["gain"], c["pred"][m]["gain"], marker=mk, color=col, ls="none", ms=4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observed gain (ratio of mean makespans)")
    ax.set_ylabel("Frozen predicted gain")
    ax.set_title("Predicted vs observed (band = +/-20%)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color="#D55E00", label="P (primary)"),
                       Line2D([], [], marker="x", ls="none", color="#0072B2", label="M1 (fitted kappa)")], frameon=False, loc="upper left")
    ax.grid(lw=0.3, alpha=0.5)
    data["predict"] = [dict(cell=c["cell"], variant=c["variant"], obs=c["gain"], P=c["pred"]["P"]["gain"], M1=c["pred"]["M1"]["gain"]) for c in rows]
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_predict.pdf"))
    plt.close(fig)


def fig_quality(root, F, out, data, skipped):
    """Stage-1 label agreement with the first eager run for every policy, against the eager rerun (noise floor)."""
    rows = []
    for name, c in F["cells"].items():
        runs = load_cell(root, "confirm", name, c["policies"], c["seeds"], c.get("m"))
        if runs is None:
            continue
        for pol in c["policies"][1:]:
            a = [runs[s][pol]["stage1_agreement"] for s in runs]
            w = [runs[s][pol]["stage1_pairs"] for s in runs]
            rows.append(dict(cell=name, policy=pol if c["backend"] == "vllm" else ("SGL_P6" if pol == "P6_lazyscan" else pol),
                             mean=float(np.average(a, weights=w)), lo=float(min(a)), hi=float(max(a)),
                             pairs=int(sum(runs[s][pol]["stage1_pairs"] for s in runs))))
    data["quality"] = rows
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.4, 0.16 * len(rows) + 0.8))
    for i, r in enumerate(rows):
        name, color, marker = POLICY_STYLE.get(r["policy"], (r["policy"], "k", "o"))
        ax.plot([1 - r["hi"], 1 - r["lo"]], [i, i], color=color, lw=1)
        ax.plot(1 - r["mean"], i, marker=marker, color=color, ms=3.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['cell']}: {POLICY_STYLE.get(r['policy'], (r['policy'],))[0]}" for r in rows], fontsize=5)
    ax.set_xlabel("Stage-1 disagreement with first eager run (1 - agreement)")
    ax.invert_yaxis()
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_quality.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="project scratch root holding results/")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    F = json.load(open(os.path.join(a.root, "results/confirm/confirm_freeze.json")))
    sp = os.path.join(a.root, "results/sweep/sweep_freeze.json")
    S = json.load(open(sp)) if os.path.exists(sp) else None
    data, skipped = {}, []
    fig_main(a.root, F, S, a.out, data, skipped)
    cells = gather_p6_cells(a.root, F, S, skipped)
    fig_regime(cells, a.out, data)
    fig_predict(cells, a.out, data)
    fig_quality(a.root, F, a.out, data, skipped)
    data["skipped"] = sorted(set(skipped))
    data["inputs"] = sorted(os.path.relpath(p, a.root) for p in glob.glob(os.path.join(a.root, "results/*/*_s1?.json")))
    json.dump(data, open(os.path.join(a.out, "figure_data.json"), "w"), indent=1, default=float)
    print("skipped:", data["skipped"])
    print("cells:", [(c["cell"], c["variant"], round(c["gain"], 3), round(c["pred"]["P"]["gain"], 3)) for c in cells])


if __name__ == "__main__":
    main()
