# SharedScan: supplemental material

Code, frozen protocols, and measured results for the paper "SharedScan" (PVLDB Vol. 20 submission). SharedScan is a
physical operator that places the question after the document and admits arriving queries to a lazy circular block
scan, on unmodified vLLM and SGLang.

## Layout

- `code/`: benchmark harness, model, freeze, scoring, and figure scripts. Each freeze file records the sha256 of the code it used; `b8_freeze.py` here is the later revision recorded by `b8_scale_freeze.json`.
  - `confirm_live.py`: harness of the confirmatory round (B1). `sweep_live.py`, `sweep_live_b5.py`: later blocks.
  - `sscan_model.py`, `sscan_model_v2.py`: capacity model and diagnostics.
  - `*_freeze.py`: build the frozen inputs and predictions before any run. `confirm_score.py`: scorer.
  - `confirm_rerun.py`, `confirm_rerun_summary.py`, `label_flips.py`: repeated eager runs and label analysis.
  - `b8_freeze.py`, `b8b_freeze.py`, `b8c_freeze.py`, `b8d_freeze.py`, `b8_live.py`, `b8_predicates.py`, `b8_smoke.py`: scope block B8 and its seed extensions. `b8_summary.py`, `b8_rerun.py`, `b8_rerun_summary.py`: summary, repeated eager runs on extraction.
  - `make_scope.py`, `make_scope_numbers.py`, `make_scope_fig.py`: scope table, prose numbers, and scope figure.
  - `make_figures.py`, `make_numbers.py`, `make_robust.py`, `table_style.py`: data figures, generated numbers, and table cell formatting. `derived/fig_pipeline.svg` is the source of Figure 1.
- `slurm_examples/`: example job scripts with placeholders. Set `PROJECT_ROOT` and `SIF` and replace the `<...>` values before use; the GPU type and module line are those of our cluster. Set `PROJECT_ROOT` and `SIF` and replace the `<...>` values before use; the GPU type and module line are those of our cluster.
- `protocol/`: the confirmatory preregistration and the design of each experiment block.
- `results/`: measured results per block: `sscan` (the Qwen3-8B calibration session), `confirm` (B1), `sweep` (B2, B3), `b5`, `b6`, `b7`, `b8`, `confirm_rerun`.
- `derived/`: small data behind the figures and tables.

`results/b8/*_freeze.json` list every frozen scope trace. `derived/b8_unfinished.json` lists, per cell, the frozen seeds that did not complete: a file with fewer policies than its cell lists is not counted by `b8_summary.py`. `derived/b8_job_resources.json` records each job's CPU and memory allocation.

## What was removed from the released results

Per-prompt labels are kept only for the 2,400-block traces and their reruns, which `label_flips.py` reads; every other
run keeps its agreement metrics, latencies, and trace. Freeze files omit document texts, apart from short public example prompts, and machine-specific paths, so
the `freeze_sha256` values in result files refer to the unmodified freeze files. Document ids, token statistics,
traces, cell specifications, and predictions are kept. Rebuilding a freeze file with its `*_freeze.py` script
recreates the texts from the public datasets named in that file.

## Paper artifacts

| Paper artifact | Input | Script |
|---|---|---|
| Figures 2-4 and reported numbers | `results/confirm`, `results/sweep` | `code/make_figures.py`, `code/make_numbers.py` |
| Figure 1 | none | `derived/fig_pipeline.svg` |
| Robustness table | `results/b5`, `results/b6`, `results/b7` | `code/make_robust.py` |
| Scope table, scope figure, and scope numbers | `results/b8` via `derived/b8_summary.json` | `code/b8_summary.py`, `code/make_scope.py`, `code/make_scope_fig.py`, `code/make_scope_numbers.py` |
| Answer agreement, repeated eager runs | `results/confirm`, `results/confirm_rerun`, `results/b5`-`b7` | `code/confirm_rerun_summary.py`, `code/label_flips.py` |

Example: `python3 code/make_robust.py --root . --tex <OUTPUT_DIR>/robust_table.tex --json <OUTPUT_DIR>/robust_data.json`.
`code/b8_rerun_summary.py` needs per-prompt labels that are not released for the extraction reruns, so `derived/b8_rerun_summary.json` is the output of the original run. Other scripts that read per-prompt labels need the files as released here; those that read freeze files use only the kept fields.

## Rerunning an experiment

Environment: Python 3 with the packages in `requirements.txt`, one NVIDIA L40S GPU, and the Hugging Face models and datasets named in the freeze files, at the revisions recorded there. Build a Singularity image from the official vLLM 0.30.0 container, or from one with SGLang 0.5.20 for the SGLang cells, and point `SIF` at it.

Each `code/*_freeze.py` prints its arguments with `--help` and writes a new freeze file. A rebuilt freeze file has a new sha256; pass that value. Calibration parameters are in `results/sscan/model_params_released.json`.

```bash
python3 code/confirm_live.py --freeze <FREEZE_JSON> --freeze-sha <SHA256> --cell <CELL> --seed <SEED> --save-labels --out <OUTPUT_JSON>
```

`slurm_examples/` shows how the same runs were submitted on a Slurm cluster. Set `PROJECT_ROOT` to the repository root and `SIF` to the image.

Tables and generated numbers: `python3 code/make_scope.py derived/b8_summary.json <OUTPUT_TEX>`, `python3 code/make_scope_numbers.py derived/b8_summary.json <OUTPUT_TEX>`, and `python3 code/make_scope_fig.py derived/b8_summary.json <OUTPUT_PDF>`.
