#!/usr/bin/env python3
"""Self-test of confirm_score.py on synthetic results built from a (dry-run) freeze file.

Case A: every cell's observations equal the primary model's own per-seed predictions -> PASS.
Case B: the PubMed block-32 cell at 1,600 blocks and load 3.0 is made a loss -> WITHDRAW (T1/T2).
Case C: one result file missing -> INCOMPLETE.
Case D: every seed of one cell holds only its first policy (a timed-out job) -> INCOMPLETE, no crash.
Case E: one result file truncated mid-write -> INCOMPLETE, no crash.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile


def write_results(F, sha, d, perturb=False, drop=None, partial=None, truncate=None):
    P = F["predictions"][F["primary_model"]]["per_seed"]
    for name, cell in F["cells"].items():
        src = name if cell["backend"] == "vllm" else cell["trace"]
        for s in cell["seeds"]:
            if drop == (name, s):
                continue
            runs = []
            for pol in (cell["policies"][:1] if partial == name else cell["policies"]):
                pp = "P1_eager" if pol == "P1_eager_rerun" else pol
                p = dict(P[src][str(s)][pp])
                if perturb and name == "pubmed_k1600_l3" and pol == "P6_lazyscan":
                    p["makespan"] *= 3.0
                    p["cached_frac"] = 0.05
                prompt = 1_000_000
                runs.append(dict(policy=pol, seed=s, completed=cell["m"], trace=F["traces"][cell["trace"]][str(s)],
                                 makespan=p["makespan"], p95=p["p95"], prompt_tokens=prompt,
                                 cached_tokens=int(round(p["cached_frac"] * prompt)), stage1_agreement=1.0,
                                 stage1_pairs=100, stage2_agreement=1.0, stage2_pairs=10, queries_with_identical_result=1.0))
            R = dict(cell=name, freeze_sha256=sha, corpus=cell["corpus"], blocks=cell["blocks"], load=cell["load"],
                     backend=cell["backend"], m=cell["m"], n_docs=cell["n"], apc=True, overlap=cell["overlap"],
                     family="filter", model_revision=F["model_revision"], model=F["model"], p6_block=cell["p6_block"],
                     inflight=2, sgl_policy=cell["sgl_policy"], calibration_single_query_s=cell["t1_hat"],
                     runs=runs)
            txt = json.dumps(R)
            open(os.path.join(d, f"{name}_s{s}.json"), "w").write(txt[:len(txt) // 2] if truncate == (name, s) else txt)


def run(freeze, sha, d):
    out = os.path.join(d, "score.json")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "confirm_score.py"),
                    "--freeze", freeze, "--freeze-sha", sha, "--results", d, "--out", out], check=True,
                   stdout=subprocess.DEVNULL)
    return json.load(open(out))


def main():
    freeze = sys.argv[1]
    sha = hashlib.sha256(open(freeze, "rb").read()).hexdigest()
    F = json.load(open(freeze))
    for case, kw, expect in (("A", {}, "PASS"), ("B", dict(perturb=True), "WITHDRAW"),
                             ("C", dict(drop=("arxiv_k2400_l3", 11)), "INCOMPLETE"),
                             ("D", dict(partial="pubmed_k1600_l3"), "INCOMPLETE"),
                             ("E", dict(truncate=("pubmed_k2400_l3", 12)), "INCOMPLETE")):
        with tempfile.TemporaryDirectory() as d:
            write_results(F, sha, d, **kw)
            rep = run(freeze, sha, d)
            ok = rep["VERDICT"].startswith(expect)
            print(case, "expected", expect, "got:", rep["VERDICT"][:100], "|",
                  {m: (v["T1"]["correct"], v["T2"]["pass_"], v["T3"]["pass_"]) for m, v in rep["models"].items()})
            assert ok, case
            if case == "A":
                e = rep["models"]["P"]["per_policy"]["pubmed_k1600_l3"]["P1_eager_rerun"]
                assert abs(e["makespan_rel_err"]) < 1e-9, e  # rerun scored against the frozen eager prediction
    print("SELFTEST_OK")


if __name__ == "__main__":
    main()
