# Experiment Plan: SharedScan (PVLDB Vol. 20)
## Experiment Blocks

### B1. Frozen confirmatory round

- 24 jobs: arXiv and PubMed × {1,600 blocks at load 1.0; 1,600 at 3.0 with b = 16/32/64; 2,400 at 3.0} × seeds 10–12, plus SGLang LPM vs P6 at 1,600 blocks and load 3.0.
- Scorer: `confirm_score.py`, giving PASS, WITHDRAW or INCOMPLETE.
- Jobs 64678388–64678411, about 41 GPU-hours, expected to finish around 2026-10-24T00:00–02:00.

### B2. Batching baselines (anti-claim a)

- **Cells:** PubMed and arXiv at 1,600 blocks and load 3.0, seeds 10–12, on the **same frozen traces** as B1's `k1600_l3`.
- **Policies, in one job per (corpus, seed):**
  - P1_eager, the quality reference;
  - P2_adaptive: when the previous batch's stage-1 work finishes, release all queries that arrived since, document-major;
  - P2_window10 and P2_window60;
  - P6_lazyscan with b = 32, re-run in the same job for a paired comparison.
- **Metrics:** makespan, p95, mean latency, cached fraction, agreement.
- **Expected outcome:** P6 shows a lower p95 than every batching policy at equal or better makespan. If adaptive batching matches P6 on p95, C1 narrows to request-level caching only, and this is reported honestly.
- **Cost:** about 3 × 2.1 h (PubMed) + 3 × 0.6 h (arXiv) ≈ 8 GPU-hours.

### B3. Boundary crossover (C2, out-of-sample)

- **Cells:** PubMed at load 3.0, with the frozen traces for seeds 10 and 11. The ratio is `max W_k / (C − 1)`:

  | Blocks C | b | Ratio |
  |---:|---:|---:|
  | 1,200 | 32 | 1.19 |
  | 1,200 | 16 | 0.63 |
  | 1,600 | 48 | 1.33 |
  | 3,200 | 64 | 0.88 |

- **Policies:** P1_eager, P2_window30, P6 at the cell's b.
- **Predictions** from P, M1, M2 and M3 are frozen in `sweep_freeze.json` before submission.
- **Cost:** about 8 jobs × 1.7 h ≈ 14 GPU-hours.

### B4. Quality validity (no new runs)

- From B1, and B2 where available:
  - stage-1 agreement over all pairs;
  - stage-2 agreement conditional on both runs passing stage 1;
  - identical final results;
  - all against P1_eager, with the P1_eager_rerun noise floor.

### B5. Controls and robustness (added 2026-10-25 after B1 PASS and the result-to-claim review)

Purpose: answer the reviewer questions that B1-B3 leave open, without changing any frozen claim. Qwen3-8B, 1,600 KV blocks, load 3.0, seeds 10-11, predictions of P (primary) and M1-M3 frozen in `results/b5/b5_freeze.json` before submission (`b5_freeze.py`). Descriptive; no pass/fail test depends on B5.

- **B5a, APC off** (`pubmed_k1600_l3_noapc`): P1_eager, P2_window30, P6_lazyscan with automatic prefix caching disabled, on the frozen PubMed traces. Tests that SharedScan's gain comes from prefix reuse and not from scheduling alone.
- **B5b, blocks in flight** (`pubmed_k1600_l3_inflight`): P1_eager (quality reference), P6_lazyscan_F1, P6_lazyscan (F=2), P6_lazyscan_F4. Tests the sensitivity to F.
- **B5c, low range overlap** (`<corpus>_k1600_l3_low`, both corpora): new traces whose ranges cover 5-20% of the collection, rate = 3.0 / (t1_hat × 0.125), with the same t1_hat as the confirmatory cell; P1_eager, P2_window30, P6_lazyscan. Tests whether the gain persists when queries share fewer documents.
- **Code:** `sweep_live_b5.py` is `sweep_live.py` plus (i) a cell key `apc` (default true) and (ii) policy names `P6_lazyscan[_b<N>][_F<k>]`; the B2/B3 harness file is unchanged. Wrapper `slurm_examples/b5_vllm_lane.sbatch`, same lanes, same R+P <= 8 GPU rows.
- **Cost:** 8 jobs; about 1.5 h each for PubMed and 0.7 h for arXiv, about 11 GPU-hours.
- **Deferred:** Qwen3-4B (needs a fresh α calibration run before its predictions can be frozen) and multi-token outputs.

### B6. Contiguous ranges (added 2026-10-25 after auto-review round 1)

Purpose: the confirmatory traces draw each range start uniformly and wrap around the end of the key order, so most ranges are two key intervals (127 of 180 B2 queries per corpus wrap); a SQL `BETWEEN` predicate selects one contiguous interval. B6 repeats the main comparison with contiguous ranges.

- **Cells:** `<corpus>_k1600_l3_nowrap`, both corpora, seeds 10-12, Qwen3-8B, 1,600 KV blocks, load 3.0, each B6 trace copies the frozen `<corpus>_k1600_l3` trace query by query (arrival time, range length, predicates) and redraws only the range start, uniformly among the positions at which the whole range fits, from an independent RNG (`b6_freeze.nowrap_trace`, seed 5000+s); paired comparison. Middle documents are covered more often than the ends; the paper states the sampling rule.
- **Policies:** P1_eager, P1_eager_rerun (arXiv only: noise floor for result-set agreement), P2_window30, P2_adaptive (no prediction), P6_lazyscan_b16, P6_lazyscan.
- **Predictions:** P and M1-M3 frozen in `results/b6/b6_freeze.json` before submission. B6 replaces the wrapped-range B1/B2 numbers as the paper's main comparison if its traces complete; B1 remains the preregistered test of the model.
- **Code:** harness `sweep_live_b5.py` unchanged; `b6_freeze.py`, `slurm_examples/b6_vllm_lane.sbatch`, `slurm_examples/b6_freeze.sbatch`.
- **Cost:** PubMed about 2.2 h per trace, arXiv about 1.0 h per trace; about 10 GPU-hours.

### B7. Second model (added 2026-10-25; addresses the single-model scope)

Purpose: test whether the gains and the capacity model carry to a second model without refitting anything but measured quantities. Qwen3-4B-Instruct-2507 has the same tokenizer vocabulary as Qwen3-8B (identical tokenizer.json) and the same KV size per token (36 layers, 8 KV heads, head dimension 128), so the KV budgets carry over; its chat template differs, so prompt token statistics are recomputed.

- **Step 1, calibration** (`slurm_examples/b7_calib_lane.sbatch`): one session mirroring the Qwen3-8B calibration (C1 seed 0: arXiv, 800 KV blocks, load 2.0, 30 queries, P1_eager, P1_eager_rerun, P2_window30, P6_lazyscan), arrival rate from the 4B's own measured t1.
- **Step 2, frozen predictions** (`b7_freeze.py`, `slurm_examples/b7_freeze.sbatch`): α from the two eager runs exactly as for Qwen3-8B (least squares through the origin when collinear, β = 0) but without sscan_model.fit's scale search (which selected 1.0 for Qwen3-8B); pass rate as in sscan_model's fit command; κ = 0.7 carried over for the diagnostic M1/M2. Both corpora re-tokenized with the 4B template (arXiv statistics asserted equal to the calibration session's). Cells `<corpus>_k1600_l3_q4b`, seeds 10-11, 1,600 KV blocks, load 3.0 relative to Qwen3-4B (rate = 3 / (t1_hat × 0.75), t1_hat from the model with the new parameters), harness traces with every range start redrawn to be contiguous (independent RNG, seed 5000+s); policies P1_eager, P2_window30, P2_adaptive (no prediction), P6_lazyscan_b16, P6_lazyscan. Predictions of P and M1-M3 frozen before submission; descriptive. Before freezing, the calibration's pass rate is sanity-checked against Qwen3-8B's (0.178), because the harness records any first token other than "Yes" as "No".
- **Code:** the harness is unchanged (`sweep_live_b5.py`): the B7 freeze file's top-level model, revision and corpora are the 4B's, which the harness reads and asserts. Wrapper `slurm_examples/b7_vllm_lane.sbatch`.
- **Cost:** calibration about 0.7 h; step 2 four jobs, about 6 GPU-hours.

### B8. Scope robustness (added 2026-11-01)

Purpose: test reviewer objections to scope. Code `b8_freeze.py`, `b8_live.py`, `b8_predicates.py`; harness `sweep_live_b5.py` unchanged. Predictions of P and M1 frozen before any job; descriptive; unfavorable results are reported. Cells are described in the docstring of `b8_freeze.py`: distinct predicates, bursty arrivals, arXiv extraction, F=1 against adaptive on B6 traces, and larger collections.

### B8b. More seeds of B8 (added 2026-11-01, after seeing seeds 10 and 11)

Purpose: the cells of B8 with two traces varied by trace, and the hyperexponential generator's realized CV was 2.71 and 1.01 (target 3). B8b adds seeds 12 and 13 of the distinct-predicate, hyperexponential and extraction cells. This is an expansion made after seeing seeds 10 and 11, not a preregistered four-seed experiment. Code `b8b_freeze.py`, `cluster/b8b_vllm_lane.sbatch`; predictions frozen before any job. Seeds were fixed as the next two integers and none is dropped or selected by realized CV or load. Stopping rule: run in the queued order until the official deadline; report every completed seed and list unfinished ones. Report per seed, seeds 10-11, seeds 12-13 and pooled; pooled values are descriptive. Freeze sha256 is recorded in PROJECT-STATUS.md when written.

### B8c. Further seeds with the GPUs left free until the deadline (added 2026-11-01, after seeing the results of B8 and most of B8b)

Purpose: use the GPU slots that free up before the official deadline, adding traces where B8 and B8b have few. B8c adds seeds 12 and 13 of the two arXiv scale cells (5,000 documents, 1,600 and 10,000 blocks; 3.5 to 3.8 h each) and seeds 14 and 15 of the distinct-predicate, hyperexponential and extraction cells. PubMed scale cells are not extended because each seed takes about 6 h and would not finish; this follows from runtime, not results. This is an expansion made after seeing results, not a preregistered experiment. Code `b8c_freeze.py`, `cluster/b8c_vllm_lane.sbatch`; predictions frozen before any job, and the freeze asserts that regenerating the earlier seeds reproduces their frozen traces. Seeds are the next integers; none is dropped or selected by realized CV, load or outcome. Queue order, fixed before any B8c job: arXiv scale s12, s13 at both block counts ahead of the remaining B8b jobs (they must start early to finish), then the arXiv extraction, distinct and bursty cells at s14, the same at s15, then PubMed bursty and distinct at s14, then s15. Stopping rule: run in that order until the official deadline; report every completed seed and list unfinished ones; a seed's incomplete run is not used. Report per seed and by expansion batch (10-11, 12-13, 14-15 or scale 10-11, 12-13) and pooled; pooled values are descriptive. Same reporting obligations as B8 and B8b. Freeze sha256 is recorded in PROJECT-STATUS.md when written.

### B8d. Seeds 16 and 17 of the arXiv distinct-predicate, bursty, and extraction cells

Purpose: use the GPUs left free near the deadline for jobs short enough to finish, in place of PubMed seeds 14 and 15, which take longer. The choice is based on runtime. Cell definitions, arrival rates, trace construction, and parameters are those of `b8_freeze.json`. Traces and predictions are frozen in `b8d_freeze.json` before the runs; the seeds are the next integers, and none is dropped. Reported descriptively, by batch.
