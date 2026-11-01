# Confirmatory round for the shared-scan paper: pre-registration

Drafted 2026-10-23T13:07-04:00. No confirmatory job has been submitted, and none will be until the freeze record at the end of this file is filled in.

This round follows two cross-model reviews:
- **Round 6.** "Continue D provisionally for its live shared-scan result; the old gate is NO-GO, and only a clean new holdout can earn the predictive systems-paper claim."
- **Round 7.** "Conditional GO for a genuinely prospective confirmatory test with v1 at configured usable capacity as the sole primary model; NO-GO for the proposed v2-primary freeze."

The earlier gate (`SSCAN_PREREG.md`) remains procedurally NO-GO. This round is a new, independent test and does not repair that gate.

## Claims under test

- **Empirical (descriptive).** A lazy document-level shared scan (P6) improves latency and throughput over request-level prefix caching for concurrent semantic filters under constrained KV capacity, compared with:
  - vLLM eager submission with automatic prefix caching (APC);
  - a fixed 30-second batching window;
  - SGLang's native longest-prefix-match scheduler (LPM).

  The results are reported as measured. No threshold is attached, and the eager and window comparisons are reported separately.
- **Predictive model and block-size rule (tested; withdrawn on failure).** A capacity-and-reuse-distance model predicts, before execution:
  - in which cells P6 wins, ties or loses against the best baseline;
  - P6's cached fraction and gain on a new long-document corpus;
  - which scan block size to use.

## Models (all frozen before execution)

- **P, primary.** `sscan_model.simulate` (v1) with cache capacity equal to the configured usable KV capacity, `num_gpu_blocks − 1`, in place of the fitted κ. vLLM 0.30 reserves one null block (`vllm/v1/core/block_pool.py`, line 193).
  - α = 7.67e-5 s per computed token, β = 0, and stage-1 pass rate 0.178, from the v1 fit on gate cell C1 seed 0 (`results/sscan/model_params.json`).
  - Implemented as `confirm_freeze.sim_primary`.
  - **Developed post hoc**, after a development check on the gate data. That check showed that κ is not identified by the calibration cell: C1's predictions are nearly identical for every κ from 0.7 to 1.0. It also showed that κ = 0.7 underestimates the effective capacity in SciFact cell C7.
  - P is tested here only on new cells.
- **Reported diagnostics.** No claim depends on these.
  - **M1:** v1 exactly as frozen for the gate (κ = 0.7).
  - **M2:** v2 (`sscan_model_v2.py`), with block-step interleaving under the 2,048-token engine step budget, κ = 0.7.
  - **M3:** v2 with pool = `num_gpu_blocks − 1`.
  - v2 approximates vLLM's scheduler. Where vLLM would allocate a duplicate block after a prefix miss, v2 shares the existing one.
- **Development record** (`results/sscan/v2dev/`):
  - At κ = 0.7, both v1 and v2 miss C7. For P6, the predicted cached fraction is about 0.09–0.10, against 0.58 observed.
  - At κ = 1.0 (pool = B), the maximum errors over the gate cells are:
    - v1: cached fraction 0.033, makespan 5.1%;
    - v2: cached fraction 0.047, makespan 5.1%.
  - The exact primary model P (pool = B − 1) was checked the same way (job 64677115, `v2_gate_check_primary.json`): maximum cached-fraction error 0.033, maximum makespan error 5.1%. On C7 it predicts gains of 1.07 and 1.23, against 1.09 and 1.29 observed.

## Harness and frozen inputs

- **Harness:** `confirm_live.py`, derived from the gate harness `shared_scan_live.py`.
  - **Engine and model:** Qwen3-8B, with the revision recorded in the freeze, on 1× L40S.
  - **vLLM 0.30.0 `AsyncLLM`:** APC on; `num_gpu_blocks_override` = the cell's KV blocks; `max_num_batched_tokens = 2048` and `max_num_seqs = 256`. These are pinned to the values every gate run logged.
  - **SGLang 0.5.20:** `max_total_tokens` = 16 × the KV blocks; `schedule_policy = lpm`.
  - **Prompt layout:** query-last.
  - **Policies:**
    - **P1_eager:** the quality reference.
    - **P1_eager_rerun:** output variability.
    - **P2_window30:** the fixed 30-second window.
    - **P6_lazyscan:** circular block scan with 32 documents per block and 2 blocks of stage-1 work in flight.
    - **P6_lazyscan_b16 and P6_lazyscan_b64:** the same scan with 16 or 64 documents per block.
- **Frozen mode.** The harness reads the documents, trace and policy list of one cell and seed from the freeze file. It also forces the frozen P6 block size (32), the in-flight count (2), APC on, the SGLang policy (LPM) and the model. Results are written atomically after each policy. It stops unless all of the following hold:
  - the freeze file's sha256 matches;
  - the model revision matches;
  - its prompts reproduce the frozen token counts exactly;
  - every prompt fits within `max_model_len`.
- **Corpora.** N = 1,000 documents each.
  - **arXiv:** exactly the gate's corpus and templates.
    - Source: `gfissore/arxiv-abstracts-2021`, first 400,000 lines.
    - Rule: abstracts of 400–2,000 characters; shuffled with `random.Random(0)`; first 1,000 taken; sorted by id.
    - `max_model_len` 1,024. The freeze builder asserts that its token statistics equal gate run C1 seed 0's.
  - **PubMed, the new long-document corpus:** `ccdv/pubmed-summarization`, test split (Cohan et al. 2018, built from PubMed Open Access articles).
    - Rule: articles with at least 3,000 characters after whitespace normalization, cut to their first 3,000 characters; ids are row indices; shuffled with `random.Random(0)`; first 1,000 taken; sorted by id.
    - Template: "Read the following excerpt from a biomedical research article and answer the question that follows.", followed by the excerpt and then the question.
    - Predicates: SciFact's 16, with "the abstract" replaced by "the text".
    - `max_model_len` 2,048.
  - The dataset revisions, document ids and texts, and the token statistics are all in the freeze file.
- **Traces.**
  - Seeds 10, 11 and 12; m = 60 queries per trace; high overlap (query ranges of 50–100% of the corpus).
  - Half the queries have a second stage.
  - Poisson arrivals at rate = load / (t1_hat × 0.75), where t1_hat is v1's isolated single-query time computed from the frozen token statistics.
  - Every policy in a cell uses the same trace, and the SGLang cell uses the vLLM 1,600-block, load-3.0 trace.
  - The live warm-up time is recorded so the realized load can be reported.

## Cells

| Cell | Corpus | KV blocks | Load | Backend | Policies (in run order) |
|---|---|---:|---:|---|---|
| `arxiv_k1600_l1` | arXiv | 1,600 | 1.0 | vLLM | P1_eager, P2_window30, P6_lazyscan |
| `arxiv_k1600_l3` | arXiv | 1,600 | 3.0 | vLLM | P1_eager, P1_eager_rerun, P2_window30, P6_lazyscan_b16, P6_lazyscan, P6_lazyscan_b64 |
| `arxiv_k2400_l3` | arXiv | 2,400 | 3.0 | vLLM | P1_eager, P2_window30, P6_lazyscan |
| `arxiv_sgl_k1600_l3` | arXiv | 1,600 | 3.0 | SGLang LPM | P1_eager, P6_lazyscan |
| `pubmed_k1600_l1` | PubMed | 1,600 | 1.0 | vLLM | as `arxiv_k1600_l1` |
| `pubmed_k1600_l3` | PubMed | 1,600 | 3.0 | vLLM | as `arxiv_k1600_l3` |
| `pubmed_k2400_l3` | PubMed | 2,400 | 3.0 | vLLM | as `arxiv_k2400_l3` |
| `pubmed_sgl_k1600_l3` | PubMed | 1,600 | 3.0 | SGLang LPM | P1_eager, P6_lazyscan |

Each cell runs seeds 10–12, one job per cell and seed (24 jobs).

**Scored cells.** Ten (cell, P6 variant) pairs, five per corpus:
- `k1600_l1`, P6 with block 32;
- `k1600_l3` with block 16;
- `k1600_l3` with block 32;
- `k1600_l3` with block 64;
- `k2400_l3` with block 32.

## Scoring (`confirm_score.py`, frozen)

- **Aggregation.** Per cell, each policy's makespan, p95 (nearest rank, from arrival) and cached fraction are averaged over the three seeds.
- **Gain.** Gain = best-baseline makespan / P6-variant makespan, where the best baseline is the better of P1_eager and P2_window30 by makespan.
- **Class.**
  - **win:** gain > 1.05;
  - **tie:** 0.95 ≤ gain ≤ 1.05;
  - **loss:** gain < 0.95.

  Predictions are classified the same way.
- **Prospective block-choice rule.** Among the P6 block sizes {16, 32, 64} whose predicted makespan in `<corpus>_k1600_l3` is within 5% of the best predicted makespan, choose the one with the lowest predicted p95.

**Tests**, applied to every model; only P's result decides the claims:

- **T1.** At least 8 of the 10 scored cells are classified correctly, including at least 4 of the 5 PubMed cells.
- **T2.** In each of the four PubMed load-3.0 scored cells:
  - |predicted − observed| P6 cached fraction ≤ 0.12, **and**
  - |predicted − observed| / observed makespan gain ≤ 0.20.
- **T3.** For each corpus, the chosen block size's observed p95 is ≤ 1.10 × the best observed p95 among the three sizes, **and** its observed makespan is ≤ 1.05 × the best observed makespan among the three.

**Verdict.**
- **INCOMPLETE** if any pre-registered (cell, seed) result is missing or fails a metadata, policy, trace, revision or completion check.
- **PASS** if P passes T1, T2 and T3. The predictive-model and block-choice claims then stand, stated as a model developed post hoc and tested prospectively.
- **WITHDRAW** otherwise: the predictive-model and block-choice claims are withdrawn, and only the empirical shared-scan results are reported.
  - Refitting on these cells would require another fresh holdout.
  - Choosing whichever diagnostic model did best after the fact is not allowed.

**Always reported:**
- all prediction errors for P, M1, M2 and M3 (cached fraction, makespan and p95 for every policy and cell);
- the empirical gains of every P6 variant against eager, against the window and against the best baseline, with per-seed values;
- the SGLang LPM vs P6 gains;
- stage-1 agreement over all (query, document) pairs; stage-2 agreement conditional on both runs passing stage 1, with its denominator; and the share of queries with an identical final result. Each is given against P1_eager, with the P1_eager_rerun noise floor in the `k1600_l3` cells;
- the realized load (load × live t1 / t1_hat).

## Reviews of this protocol and its code

- **Protocol:** research-review rounds 6 and 7 (cross-model review).
- **Code:** three Codex code-review rounds. 
  - Round 1 found four defects: the frozen block and SGLang policy were overridable; the family flag was set before the frozen settings; a partial result could crash the scorer; prediction errors and per-seed gains were missing.
  - Round 2 found two more: a truncated result file could crash the scorer, and the eager rerun had no prediction errors.
  - All six were fixed. Round 3: "Yes—ready to freeze."
- **Scorer self-test:** cluster job 64677983, on synthetic results from a dry-run freeze. Expected and obtained: PASS when the observations equal P's predictions; WITHDRAW under a perturbation; INCOMPLETE for a missing file, a partial cell and a truncated file.

## Compute

- **Model jobs.** One 1× L40S job per (cell, seed) on one GPU partition, with 8 CPUs and 60 GB.
  - Walltime is 1.3 × the estimated runtime, computed by `confirm_submit.py`:
    - **vLLM:** the predicted policy makespans for that cell and seed (maximum over the four models) + the warm-up query (t1_hat) + 1,000 s of start-up. The start-up figure is the largest overhead measured in the gate jobs (457–992 s).
    - **SGLang:** the predicted vLLM eager makespan + 1.7 × the predicted vLLM P6 makespan + 1.5 × t1_hat + 1,000 s. In the gate, SGLang eager was close to vLLM eager, and SGLang P6 was up to 1.70× slower than vLLM P6.
  - The jobs are packed into four `afterany` dependency lanes, longest job first, so at most 4 of this project's GPUs are in use at once. The total estimate is about 41 GPU-hours.
- **Freeze and scoring jobs.** CPU jobs on one CPU partition.

## Freeze record

Written 2026-10-23T13:29-04:00, **before any confirmatory GPU job was submitted**. No confirmatory result exists at this time.

- **Frozen file.** `results/confirm/confirm_freeze.json` (on the cluster: `results/confirm/confirm_freeze.json`, made read-only; local copy also read-only).
  - `written_at`: 2026-10-23T13:28:32-04:00.
  - sha256: **`56eda5388de68d211a242f9baca09fa727c4eb337804067b817b3fef5c49f4b2`**.
  - Produced by freeze job 64678141 (log `results/confirm/vldb1-cfreeze-64678141.out`).
- **Code sha256, recorded inside the freeze.** These match the cluster copies, checked before the freeze.

  | File | sha256 |
  |---|---|
  | `confirm_live.py` | `f1bc7602…a89e9` |
  | `confirm_freeze.py` | `38537964…587ab` |
  | `confirm_score.py` | `fbe9a48a…a75a2` |
  | `sscan_model.py` | `beca44a8…4060c` (unchanged since the gate) |
  | `sscan_model_v2.py` | `072bce40…ca7e` |

  v1 parameter file `model_params.json`: sha256 `67dc356e…7553`.
- **Revisions.**
  - Model: Qwen/Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218`.
  - arXiv dataset revision `e4c5fbd4dec8e46a5dc869216fe1c94cc585757a`. Document-list sha256 `ac11eea9…185d`.
  - PubMed dataset revision `6b30a2cae59b11ed77cb19959bffccbbd18e1106` (`document/test-00000-of-00001.parquet`). Document-list sha256 `14d07b05…35d9`.
- **Token statistics.**
  - arXiv: mean prompt 262.0 tokens, maximum 636, mean shared prefix 238.0. They equal gate run C1_s0's, as asserted by the builder.
  - PubMed: mean 708.5, maximum 1,033, shared prefix 680.5.
- **Isolated single-query time t1_hat.** 18.9 s for arXiv and 53.1 s for PubMed. Arrival rates:

  | Corpus | Load 1.0 (queries/s) | Load 3.0 (queries/s) |
  |---|---:|---:|
  | arXiv | 0.0707 | 0.2120 |
  | PubMed | 0.0251 | 0.0753 |

- **Primary model P's frozen predictions.**

  | Cell | Variant | Gain | Class | P6 cached fraction |
  |---|---|---:|---|---:|
  | arXiv, 1,600 blocks, load 1.0 | block 32 | 1.016 | tie | 0.343 |
  | arXiv, 1,600 blocks, load 3.0 | block 16 | 1.147 | win | 0.664 |
  | arXiv, 1,600 blocks, load 3.0 | block 32 | 1.144 | win | 0.662 |
  | arXiv, 1,600 blocks, load 3.0 | block 64 | 1.123 | win | 0.654 |
  | arXiv, 2,400 blocks, load 3.0 | block 32 | 1.137 | win | 0.662 |
  | PubMed, 1,600 blocks, load 1.0 | block 32 | 1.011 | tie | 0.291 |
  | PubMed, 1,600 blocks, load 3.0 | block 16 | 1.668 | win | 0.658 |
  | PubMed, 1,600 blocks, load 3.0 | block 32 | 1.644 | win | 0.647 |
  | PubMed, 1,600 blocks, load 3.0 | block 64 | 0.656 | loss | 0.109 |
  | PubMed, 2,400 blocks, load 3.0 | block 32 | 1.649 | win | 0.649 |

  Block choice: 16 for both corpora.

  The diagnostics M1 and M2 differ from P on PubMed at 1,600 blocks, load 3.0, block 32: they predict a loss there. All predictions are in the frozen file.

