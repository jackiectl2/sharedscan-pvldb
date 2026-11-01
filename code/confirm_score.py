#!/usr/bin/env python3
"""Score the confirmatory round (CONFIRM_PREREG.md) from result files and the frozen inputs.

Strictness: the freeze file must match its recorded sha256; every pre-registered (cell, seed) result
must exist with the frozen metadata, policies, trace and completed == m, else the verdict is
INCOMPLETE. Model tests use the pre-registered rules unchanged and are applied to every model; the
predictive-model and block-choice claims depend only on the pre-designated primary model.

Model tests (each model):
  T1 classification: P6-variant vs best-baseline makespan ratio classified win (> 1.05), tie
     (0.95-1.05) or loss (< 0.95); >= 8 of 10 scored cells correct, including >= 4 of 5 long-corpus cells.
  T2 long-corpus high-load accuracy: in every long-corpus load-3.0 scored cell, |predicted - observed|
     P6 cached fraction <= 0.12 and |predicted - observed| / observed makespan gain <= 0.20.
  T3 block choice: per corpus, the model's choice among P6 block sizes 16/32/64 at 1,600 blocks and
     load 3.0 has observed p95 <= 1.10 x the best observed p95 and observed makespan <= 1.05 x the
     best observed makespan among the three.
Everything else is descriptive: per-policy errors, empirical gains, SGLang, agreement, realized load.
"""
import argparse
import hashlib
import json
import os

LONG = "pubmed"
BASELINES = ("P1_eager", "P2_window30")
P6_VARIANTS = ("P6_lazyscan_b16", "P6_lazyscan", "P6_lazyscan_b64")


def classify(r, tie):
    return "win" if r > tie[1] else ("loss" if r < tie[0] else "tie")


def load_cell(F, name, results, problems):
    """Load a cell's seed files; returns (runs, valid). A cell with any problem is not scored."""
    cell = F["cells"][name]
    runs, n0 = {}, len(problems)
    for s in cell["seeds"]:
        p = os.path.join(results, f"{name}_s{s}.json")
        if not os.path.exists(p):
            problems.append(f"missing {p}")
            continue
        try:
            R = json.load(open(p))
        except (OSError, ValueError) as e:
            problems.append(f"unreadable {p}: {e}")
            continue
        meta = dict(cell=name, freeze_sha256=F["_sha"], corpus=cell["corpus"], blocks=cell["blocks"],
                    load=cell["load"], backend=cell["backend"], m=cell["m"], n_docs=cell["n"], apc=True,
                    overlap=cell["overlap"], family="filter", model_revision=F["model_revision"], model=F["model"],
                    p6_block=cell["p6_block"], inflight=2, sgl_policy=cell["sgl_policy"])
        for k, v in meta.items():
            if R.get(k) != v:
                problems.append(f"{name}_s{s}: {k}={R.get(k)!r} expected {v!r}")
        got = [r["policy"] for r in R["runs"]]
        if got != cell["policies"]:
            problems.append(f"{name}_s{s}: policies {got} expected {cell['policies']}")
        frozen = F["traces"][cell["trace"]][str(s)]
        for r in R["runs"]:
            if r.get("seed") != s or r.get("completed") != cell["m"]:
                problems.append(f"{name}_s{s}/{r['policy']}: seed {r.get('seed')} completed {r.get('completed')}")
            keys = ("qid", "arrival", "lo", "n", "p1", "p2")
            if [{k: q[k] for k in keys} for q in r["trace"]] != [{k: q[k] for k in keys} for q in frozen]:
                problems.append(f"{name}_s{s}/{r['policy']}: trace differs from the frozen trace")
        runs[s] = R
    return runs, len(problems) == n0 and len(runs) == len(cell["seeds"])


def aggregate(runs):
    """Mean over seeds of each policy's metrics (as in the gate evaluator); agreement pooled by pairs."""
    acc = {}
    for R in runs.values():
        for r in R["runs"]:
            o = acc.setdefault(r["policy"], dict(makespan=[], p95=[], cached=[], s1=[], s1n=[], s2=[], s2n=[], idres=[]))
            o["makespan"].append(r["makespan"]); o["p95"].append(r["p95"])
            o["cached"].append(r["cached_tokens"] / r["prompt_tokens"])
            o["s1"].append(r["stage1_agreement"]); o["s1n"].append(r["stage1_pairs"])
            o["s2"].append(r["stage2_agreement"]); o["s2n"].append(r["stage2_pairs"])
            o["idres"].append(r["queries_with_identical_result"])
    out = {}
    for pol, o in acc.items():
        def pooled(x, n):
            num = sum(a * b for a, b in zip(o[x], o[n]) if a is not None)
            den = sum(b for a, b in zip(o[x], o[n]) if a is not None)
            return (num / den if den else None), den
        s1, s1n = pooled("s1", "s1n")
        s2, s2n = pooled("s2", "s2n")
        k = len(o["makespan"])
        out[pol] = dict(makespan=sum(o["makespan"]) / k, p95=sum(o["p95"]) / k, cached=sum(o["cached"]) / k,
                        stage1_agreement=s1, stage1_pairs=s1n, stage2_agreement=s2, stage2_pairs=s2n,
                        identical_result=sum(o["idres"]) / k, per_seed_makespan=o["makespan"], per_seed_p95=o["p95"])
    return out


def best(A, key):
    b = min(BASELINES, key=lambda x: A[x][key])
    return b, A[b][key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--freeze-sha", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sha = hashlib.sha256(open(a.freeze, "rb").read()).hexdigest()
    assert sha == a.freeze_sha, f"freeze file sha256 {sha} != recorded {a.freeze_sha}"
    F = json.load(open(a.freeze))
    F["_sha"] = sha
    tie = tuple(F["tie_band"])
    problems, obs, raw = [], {}, {}
    for name in F["cells"]:
        raw[name], valid = load_cell(F, name, a.results, problems)
        if valid:
            obs[name] = aggregate(raw[name])
    rep = dict(freeze_sha256=sha, primary_model=F["primary_model"], problems=problems)
    corpora = list(F["corpora"])
    scored_keys = [(r["cell"], r["variant"], r["corpus"], r["high_load"]) for r in F["predictions"][F["primary_model"]]["scored"]]
    # Observed scored quantities.
    observed = []
    for name, var, corp, hl in scored_keys:
        if name not in obs:
            continue
        A = obs[name]
        bm, mk = best(A, "makespan")
        bp, p95 = best(A, "p95")
        g = mk / A[var]["makespan"]
        observed.append(dict(cell=name, variant=var, corpus=corp, high_load=hl, makespan_gain=g, cls=classify(g, tie),
                             p95_gain=p95 / A[var]["p95"], p6_cached_frac=A[var]["cached"], best_baseline_makespan=bm,
                             best_baseline_p95=bp))
    obs_by = {(o["cell"], o["variant"]): o for o in observed}
    obs_choice_ok = {}
    for c in corpora:
        name = f"{c}_k1600_l3"
        if name in obs:
            A = obs[name]
            bp = min(A[v]["p95"] for v in P6_VARIANTS)
            bmk = min(A[v]["makespan"] for v in P6_VARIANTS)
            obs_choice_ok[c] = {v: dict(p95_ratio=A[v]["p95"] / bp, makespan_ratio=A[v]["makespan"] / bmk,
                                        ok=A[v]["p95"] <= 1.10 * bp and A[v]["makespan"] <= 1.05 * bmk) for v in P6_VARIANTS}
    models = {}
    for m, P in F["predictions"].items():
        rows = []
        for pr in P["scored"]:
            o = obs_by.get((pr["cell"], pr["variant"]))
            if o is None:
                continue
            rows.append(dict(cell=pr["cell"], variant=pr["variant"], corpus=pr["corpus"], high_load=pr["high_load"],
                             pred_gain=pr["makespan_gain"], obs_gain=o["makespan_gain"], pred_cls=pr["cls"], obs_cls=o["cls"],
                             correct=pr["cls"] == o["cls"], pred_cached=pr["p6_cached_frac"], obs_cached=o["p6_cached_frac"],
                             cached_err=abs(pr["p6_cached_frac"] - o["p6_cached_frac"]),
                             gain_rel_err=abs(pr["makespan_gain"] - o["makespan_gain"]) / o["makespan_gain"],
                             pred_p95_gain=pr["p95_gain"], obs_p95_gain=o["p95_gain"]))
        n_ok = sum(r["correct"] for r in rows)
        n_long = sum(r["correct"] for r in rows if r["corpus"] == LONG)
        t1 = len(rows) == 10 and n_ok >= 8 and n_long >= 4
        t2_rows = [r for r in rows if r["corpus"] == LONG and r["high_load"]]
        t2 = len(t2_rows) == 4 and all(r["cached_err"] <= 0.12 and r["gain_rel_err"] <= 0.20 for r in t2_rows)
        t3 = {c: dict(choice=P["block_choice"][c], **obs_choice_ok[c][P["block_choice"][c]])
              for c in corpora if c in obs_choice_ok}
        t3_pass = len(t3) == len(corpora) and all(v["ok"] for v in t3.values())
        errs = {}
        for name, by_pol in P["per_cell"].items():
            if name in obs:
                by_pol = dict(by_pol)
                if "P1_eager_rerun" in obs[name]:
                    by_pol["P1_eager_rerun"] = by_pol["P1_eager"]  # the frozen eager prediction applies to the rerun
                errs[name] = {pol: dict(pred_cached=v["cached_frac"], obs_cached=obs[name][pol]["cached"],
                                        cached_abs_err=abs(v["cached_frac"] - obs[name][pol]["cached"]),
                                        pred_makespan=v["makespan"], obs_makespan=obs[name][pol]["makespan"],
                                        makespan_rel_err=(v["makespan"] - obs[name][pol]["makespan"]) / obs[name][pol]["makespan"],
                                        pred_p95=v["p95"], obs_p95=obs[name][pol]["p95"],
                                        p95_rel_err=(v["p95"] - obs[name][pol]["p95"]) / obs[name][pol]["p95"])
                              for pol, v in by_pol.items() if pol in obs[name]}
        models[m] = dict(T1=dict(correct=f"{n_ok}/{len(rows)}", long_correct=f"{n_long}/{sum(r['corpus'] == LONG for r in rows)}",
                                 pass_=t1),
                         T2=dict(rows=[{k: r[k] for k in ("cell", "variant", "cached_err", "gain_rel_err")} for r in t2_rows],
                                 pass_=t2),
                         T3=dict(per_corpus=t3, pass_=t3_pass), all_pass=t1 and t2 and t3_pass, scored=rows, per_policy=errs)
    rep["models"] = models
    rep["observed_scored"] = observed
    rep["observed_block_choice"] = obs_choice_ok
    # Descriptive empirical results.
    emp = {}
    for name, A in obs.items():
        cell = F["cells"][name]
        e = dict(corpus=cell["corpus"], blocks=cell["blocks"], load=cell["load"], backend=cell["backend"],
                 per_policy={p: {k: v[k] for k in ("makespan", "p95", "cached", "stage1_agreement", "stage1_pairs",
                                                   "stage2_agreement", "stage2_pairs", "identical_result",
                                                   "per_seed_makespan", "per_seed_p95")} for p, v in A.items()})
        if cell["backend"] == "vllm":
            for v in P6_VARIANTS:
                if v in A:
                    bm, mk = best(A, "makespan")
                    bp, p95 = best(A, "p95")
                    e[v] = dict(makespan_gain_vs_best=mk / A[v]["makespan"], p95_gain_vs_best=p95 / A[v]["p95"],
                                makespan_gain_vs_eager=A["P1_eager"]["makespan"] / A[v]["makespan"],
                                p95_gain_vs_eager=A["P1_eager"]["p95"] / A[v]["p95"],
                                makespan_gain_vs_window=A["P2_window30"]["makespan"] / A[v]["makespan"],
                                p95_gain_vs_window=A["P2_window30"]["p95"] / A[v]["p95"])
        else:
            e["P6_vs_native_lpm"] = dict(makespan_gain=A["P1_eager"]["makespan"] / A["P6_lazyscan"]["makespan"],
                                         p95_gain=A["P1_eager"]["p95"] / A["P6_lazyscan"]["p95"])
        per_seed = {}
        for s_, R in sorted(raw[name].items()):
            M = {r["policy"]: r for r in R["runs"]}
            if cell["backend"] == "vllm":
                per_seed[s_] = {v: dict(makespan_gain_vs_best=min(M[b]["makespan"] for b in BASELINES) / M[v]["makespan"],
                                        p95_gain_vs_best=min(M[b]["p95"] for b in BASELINES) / M[v]["p95"],
                                        makespan_gain_vs_eager=M["P1_eager"]["makespan"] / M[v]["makespan"],
                                        p95_gain_vs_eager=M["P1_eager"]["p95"] / M[v]["p95"],
                                        makespan_gain_vs_window=M["P2_window30"]["makespan"] / M[v]["makespan"],
                                        p95_gain_vs_window=M["P2_window30"]["p95"] / M[v]["p95"])
                                for v in P6_VARIANTS if v in M}
            else:
                per_seed[s_] = dict(makespan_gain=M["P1_eager"]["makespan"] / M["P6_lazyscan"]["makespan"],
                                    p95_gain=M["P1_eager"]["p95"] / M["P6_lazyscan"]["p95"])
        e["per_seed_gains"] = per_seed
        t1_live = [R["calibration_single_query_s"] for R in raw[name].values()]
        e["realized_load"] = [cell["load"] * t / cell["t1_hat"] for t in t1_live]
        emp[name] = e
    rep["empirical"] = emp
    complete = not problems and len(obs) == len(F["cells"])
    pm = models.get(F["primary_model"], {})
    if not complete:
        rep["VERDICT"] = "INCOMPLETE"
    elif pm.get("all_pass"):
        rep["VERDICT"] = (f"PASS: primary model {F['primary_model']} passes T1, T2 and T3; the predictive-model and "
                          "block-choice claims stand (model developed post hoc, tested prospectively)")
    else:
        failed = [t for t in ("T1", "T2", "T3") if not pm.get(t, {}).get("pass_")]
        rep["VERDICT"] = (f"WITHDRAW: primary model {F['primary_model']} fails {', '.join(failed)}; withdraw the "
                          "predictive-model and block-choice claims (empirical shared-scan results only)")
    print(json.dumps({k: rep[k] for k in ("freeze_sha256", "primary_model", "problems", "VERDICT")}, indent=1))
    for m, v in models.items():
        print(m, "T1", v["T1"], "T2", v["T2"]["pass_"], "T3", {c: (x["choice"], x["ok"]) for c, x in v["T3"]["per_corpus"].items()})
    json.dump(rep, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
