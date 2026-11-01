#!/usr/bin/env python3
"""Build and freeze every input of the confirmatory round (CONFIRM_PREREG.md) before any job runs.

Writes one JSON file with
  * the documents of each corpus (ids and texts), their prompt token statistics under the frozen
    query templates, and the selection rule;
  * the cell specifications (corpus, KV blocks, load, policies, seeds, backend);
  * the arrival traces for seeds 10-12 (rate = load / (t1_hat * 0.75), t1_hat = v1's isolated
    single-query time from the frozen token statistics; the same trace is used by every policy of a
    cell, and the SGLang cell reuses the vLLM 1,600-block load-3 trace);
  * predictions for every vLLM cell, policy and seed, the scored quantities and each model's
    block-size choice, for the pre-designated primary model P (v1 with the configured usable KV
    capacity, num_gpu_blocks - 1, in place of the fitted kappa; alpha, beta and pass rate unchanged;
    developed post hoc, review round 7) and three reported diagnostics: M1 (v1 exactly as frozen
    for the gate, kappa 0.7), M2 (v2 block-step interleaving, kappa 0.7) and M3 (v2 with the
    configured usable capacity);
  * the Hugging Face revisions of the model and both corpora;
  * the sha256 of every script involved and of the v1 parameter file.
The scorer (confirm_score.py) asserts the sha256 of this file.
"""
import argparse
import datetime
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from zoneinfo import ZoneInfo  # fails closed if America/Detroit tz data is unavailable

import confirm_live as H
import sscan_model as v1
import sscan_model_v2 as v2

SEEDS = [10, 11, 12]
M, N, OVERLAP, MEAN_FRAC = 60, 1000, "high", 0.75
CORPORA = {
    "arxiv": dict(max_len=1024, source="gfissore/arxiv-abstracts-2021 (arxiv-abstracts.jsonl.gz)",
                  rule="first 400,000 lines; abstracts of 400-2,000 characters after whitespace normalization; "
                       "shuffled with random.Random(0); first 1,000; sorted by arXiv id (as in the gate)"),
    "pubmed": dict(max_len=2048, source="ccdv/pubmed-summarization, document/test parquet (Cohan et al. 2018, "
                                        "PubMed Open Access articles)",
                   rule="articles with >= 3,000 characters after whitespace normalization, cut to their first "
                        "3,000 characters; ids pm<row index>; shuffled with random.Random(0); first 1,000; sorted by id"),
}
BASELINES = ("P1_eager", "P2_window30")
P6_VARIANTS = {"P6_lazyscan_b16": 16, "P6_lazyscan": 32, "P6_lazyscan_b64": 64}
TIE = (0.95, 1.05)


def cells():
    out = {}
    for c, spec in CORPORA.items():
        base = dict(corpus=c, n=N, m=M, overlap=OVERLAP, family="filter", seeds=SEEDS, max_len=spec["max_len"],
                    backend="vllm", sgl_policy=None, p6_block=32)
        out[f"{c}_k1600_l1"] = dict(base, blocks=1600, load=1.0, policies=["P1_eager", "P2_window30", "P6_lazyscan"])
        out[f"{c}_k1600_l3"] = dict(base, blocks=1600, load=3.0,
                                    policies=["P1_eager", "P1_eager_rerun", "P2_window30", "P6_lazyscan_b16",
                                              "P6_lazyscan", "P6_lazyscan_b64"])
        out[f"{c}_k2400_l3"] = dict(base, blocks=2400, load=3.0, policies=["P1_eager", "P2_window30", "P6_lazyscan"])
        out[f"{c}_sgl_k1600_l3"] = dict(base, blocks=1600, load=3.0, backend="sglang", sgl_policy="lpm",
                                        policies=["P1_eager", "P6_lazyscan"], trace=f"{c}_k1600_l3")
    for k, v in out.items():
        v.setdefault("trace", k)
    return out


def scored_cells(corpus):
    """The five model-scored cells of a corpus: (cell, P6 variant)."""
    return [(f"{corpus}_k1600_l1", "P6_lazyscan"), (f"{corpus}_k1600_l3", "P6_lazyscan_b16"),
            (f"{corpus}_k1600_l3", "P6_lazyscan"), (f"{corpus}_k1600_l3", "P6_lazyscan_b64"),
            (f"{corpus}_k2400_l3", "P6_lazyscan")]


def classify(ratio):
    return "win" if ratio > TIE[1] else ("loss" if ratio < TIE[0] else "tie")


def block_choice(per_policy):
    """Prospective block-size rule: among P6 variants whose makespan is within 5% of the best variant,
    choose the one with the lowest p95."""
    mk = {v: per_policy[v]["makespan"] for v in P6_VARIANTS}
    ok = [v for v in P6_VARIANTS if mk[v] <= 1.05 * min(mk.values())]
    return min(ok, key=lambda v: per_policy[v]["p95"])


def model_cfg(corp, blocks):
    return dict(n=N, doc_tokens=corp["doc_tokens"], shared_prefix=corp["shared_prefix_tokens"],
                preamble=corp["preamble_tokens"], kv_tokens=blocks * 16, block=32, family="filter", inflight=2, apc=True)


def sim_primary(cfg, trace, policy, params, rng):
    """v1 with cache capacity num_gpu_blocks - 1 (vLLM 0.30 reserves one null block) instead of
    kappa * num_gpu_blocks; v1.simulate reads kv_tokens only to size its block cache."""
    return v1.simulate(dict(cfg, kv_tokens=cfg["kv_tokens"] - v1.BLK), trace, policy, dict(params, kappa=1.0), rng)


MODELS = {"P": (sim_primary, {}), "M1": (v1.simulate, {}), "M2": (v2.simulate, {}),
          "M3": (v2.simulate, {"capacity": "configured"})}
PRIMARY = "P"


def hf_revision(repo, filename):
    """Commit hash of a cached dataset file (from its snapshots/<commit>/ path)."""
    import re
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, filename, repo_type="dataset")
    return re.search(r"snapshots/([0-9a-f]{40})/", path).group(1)


def _predict(task):
    model, cell_name, seed, policy, cfg, trace, params = task
    sim, extra = MODELS[model]
    return model, cell_name, seed, policy, v2.predict_run(cfg, trace, policy, dict(params, **extra), reps=3, sim=sim)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="v1 parameter file frozen for the gate (model_params.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gate-ref", required=True, help="gate run C1_s0.json: arXiv prompts are unchanged, so their "
                                                      "token statistics must match it exactly")
    a = ap.parse_args()
    params = json.load(open(a.params))["params"]
    here = os.path.dirname(os.path.abspath(__file__))
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    model_path = snapshot_download("Qwen/Qwen3-8B")
    tok = AutoTokenizer.from_pretrained(model_path)
    corpora = {}
    for c, spec in CORPORA.items():
        H.setup_corpus(c)
        docs = {"pubmed": H.load_pubmed_docs}.get(c, H.load_docs)(N, 0)
        pr = H.Prompts(tok, docs)
        doc_tok, shared, pre = H.token_stats(pr, len(docs))
        assert len(docs) == N and max(doc_tok) < spec["max_len"], (c, len(docs), max(doc_tok))
        if c == "arxiv":
            G = json.load(open(a.gate_ref))
            assert (doc_tok, shared, pre) == (G["doc_tokens"], G["shared_prefix_tokens"], G["preamble_tokens"]), \
                "arXiv token statistics differ from the gate run: tokenizer or template mismatch"
        blob = json.dumps(docs, sort_keys=True).encode()
        if c == "arxiv":
            rev = hf_revision("gfissore/arxiv-abstracts-2021", "arxiv-abstracts.jsonl.gz")
        else:
            from huggingface_hub import list_repo_files
            f = next(x for x in list_repo_files("ccdv/pubmed-summarization", repo_type="dataset")
                     if x.startswith("document/test") and x.endswith(".parquet"))
            rev = hf_revision("ccdv/pubmed-summarization", f) + " " + f
        corpora[c] = dict(source=spec["source"], selection_rule=spec["rule"], dataset_revision=rev, docs=docs,
                          docs_sha256=hashlib.sha256(blob).hexdigest(), doc_tokens=doc_tok,
                          shared_prefix_tokens=shared, preamble_tokens=pre, corpus_prefix_tokens=sum(doc_tok),
                          example_prompt=tok.decode(pr.single(0, 0)), predicates=list(H.PREDICATES))
        print(c, "docs", len(docs), "prompt tokens mean %.1f max %d" % (sum(doc_tok) / N, max(doc_tok)),
              "shared prefix mean %.1f" % (sum(shared) / N), "preamble", pre, flush=True)
    C = cells()
    traces, tasks = {}, []
    for name, cell in C.items():
        cfg = model_cfg(corpora[cell["corpus"]], cell["blocks"])
        cell["t1_hat"] = v1.isolated_t1(cfg, params)
        cell["rate"] = cell["load"] / (cell["t1_hat"] * MEAN_FRAC)
        if cell["trace"] == name:
            H.setup_corpus(cell["corpus"])
            traces[name] = {str(s): H.make_trace(N, M, 1000 + s, cell["rate"], OVERLAP) for s in SEEDS}
    for name, cell in C.items():
        if cell["trace"] != name:
            assert C[cell["trace"]]["rate"] == cell["rate"]
        if cell["backend"] != "vllm":
            continue
        cfg = model_cfg(corpora[cell["corpus"]], cell["blocks"])
        for s in SEEDS:
            for pol in cell["policies"]:
                if pol == "P1_eager_rerun":
                    continue
                for model in MODELS:
                    tasks.append((model, name, s, pol, cfg, traces[cell["trace"]][str(s)], params))
    raw = {m: {} for m in MODELS}
    with ProcessPoolExecutor(a.workers) as ex:
        for model, name, s, pol, p in ex.map(_predict, tasks):
            raw[model].setdefault(name, {}).setdefault(str(s), {})[pol] = p
    preds = {}
    for model in MODELS:
        agg = {}
        for name, by_seed in raw[model].items():
            pols = list(next(iter(by_seed.values())))
            agg[name] = {pol: {k: sum(by_seed[str(s)][pol][k] for s in SEEDS) / len(SEEDS)
                               for k in ("cached_frac", "makespan", "p95", "mean_latency")} for pol in pols}
        scored = []
        for c in CORPORA:
            for name, var in scored_cells(c):
                A = agg[name]
                bm = min(BASELINES, key=lambda b: A[b]["makespan"])
                bp = min(BASELINES, key=lambda b: A[b]["p95"])
                g = A[bm]["makespan"] / A[var]["makespan"]
                scored.append(dict(cell=name, variant=var, corpus=c, high_load=C[name]["load"] == 3.0,
                                   makespan_gain=g, cls=classify(g), p95_gain=A[bp]["p95"] / A[var]["p95"],
                                   p6_cached_frac=A[var]["cached_frac"], best_baseline_makespan=bm))
        choice = {c: block_choice(agg[f"{c}_k1600_l3"]) for c in CORPORA}
        preds[model] = dict(per_seed=raw[model], per_cell=agg, scored=scored, block_choice=choice)
        for r in scored:
            print(model, r["cell"], r["variant"], "gain %.3f %s p95 gain %.2f p6 cached %.3f" % (
                r["makespan_gain"], r["cls"], r["p95_gain"], r["p6_cached_frac"]), flush=True)
        print(model, "block choice", choice, flush=True)
    est = {}
    for name, cell in C.items():  # job-time estimate: predicted makespans (max over models) per seed
        if cell["backend"] == "vllm":
            per = []
            for s in SEEDS:
                tot = 0.0
                for pol in cell["policies"]:
                    pp = "P1_eager" if pol == "P1_eager_rerun" else pol
                    tot += max(raw[m][name][str(s)][pp]["makespan"] for m in MODELS)
                per.append(tot)
            est[name] = per
    scripts = ["confirm_live.py", "confirm_freeze.py", "confirm_score.py", "sscan_model.py", "sscan_model_v2.py"]
    stamp = datetime.datetime.now(ZoneInfo("America/Detroit")).isoformat(timespec="seconds")
    out = dict(protocol="CONFIRM_PREREG.md", written_at=stamp,
               code_sha256={s: sha(os.path.join(here, s)) for s in scripts},
               params_v1=params, params_v1_file=os.path.abspath(a.params), params_v1_sha256=sha(a.params),
               v2_constants=dict(step_tokens=v2.STEP_TOKENS, max_seqs=v2.MAX_SEQS, block_tokens=v2.BLK),
               models={"P": "sscan_model.simulate with cache capacity num_gpu_blocks - 1 (confirm_freeze.sim_primary); "
                            "v1 alpha, beta, pass rate. PRIMARY",
                       "M1": "sscan_model.simulate, v1 parameters (kappa 0.7). Diagnostic",
                       "M2": "sscan_model_v2.simulate, v1 parameters (kappa 0.7). Diagnostic",
                       "M3": "sscan_model_v2.simulate, v1 alpha/beta/pass rate, pool = num_gpu_blocks - 1. Diagnostic"},
               primary_model=PRIMARY, model="Qwen/Qwen3-8B", model_revision=os.path.basename(model_path.rstrip("/")),
               tie_band=list(TIE), mean_frac=MEAN_FRAC, corpora=corpora, cells=C, traces=traces,
               predictions=preds, predicted_policy_seconds_per_seed=est)
    json.dump(out, open(a.out, "w"))
    print("written_at", stamp, "sha256", sha(a.out), flush=True)


if __name__ == "__main__":
    main()
