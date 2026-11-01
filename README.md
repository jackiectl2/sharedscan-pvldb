# SharedScan: supplemental material

Code, frozen protocols, and measured results for the paper "SharedScan" (PVLDB Vol. 20 submission). SharedScan is a
physical operator that places the question after the document and admits arriving queries to a lazy circular block
scan, on unmodified vLLM and SGLang.

## Layout

- `code/`: benchmark harness, model, freeze, scoring, and figure scripts, unmodified from the runs.
  - `confirm_live.py`: harness of the confirmatory round (B1). `sweep_live.py`, `sweep_live_b5.py`: later blocks.
  - `sscan_model.py`, `sscan_model_v2.py`: capacity model and diagnostics.
  - `*_freeze.py`: build the frozen inputs and predictions before any run. `confirm_score.py`: scorer.
  - `confirm_rerun.py`, `confirm_rerun_summary.py`, `label_flips.py`: repeated eager runs and label analysis.
  - `b8_freeze.py`, `b8_live.py`, `b8_predicates.py`, `b8_smoke.py`: scope-robustness block B8 (results added as they finish).
  - `make_figures.py`, `make_numbers.py`, `make_robust.py`, `make_overview.py`: figures and tables.
- `slurm_examples/`: example job scripts with placeholders.
- `protocol/`: the confirmatory preregistration and the design of each experiment block.
- `results/`: measured results per block: `sscan` (the Qwen3-8B calibration session), `confirm` (B1), `sweep` (B2, B3), `b5`, `b6`, `b7`, `confirm_rerun`.
- `derived/`: small data behind the figures and tables.

## What was removed from the released results

Per-prompt labels are kept only for the 2,400-block traces and their reruns, which `label_flips.py` reads; every other
run keeps its agreement metrics, latencies, and trace. Freeze files omit document texts and machine-specific paths, so
the `freeze_sha256` values in result files refer to the unmodified freeze files. Document ids, token statistics,
traces, cell specifications, and predictions are kept. Rebuilding a freeze file with its `*_freeze.py` script
recreates the texts from the public datasets named in that file.

## Paper artifacts

| Paper artifact | Input | Script |
|---|---|---|
| Figures 2-4 and reported numbers | `results/confirm`, `results/sweep` | `code/make_figures.py`, `code/make_numbers.py` |
| Figure 1 | none | `code/make_overview.py` |
| Robustness table | `results/b5`, `results/b6`, `results/b7` | `code/make_robust.py` |
| Answer agreement, repeated eager runs | `results/confirm`, `results/confirm_rerun`, `results/b5`-`b7` | `code/confirm_rerun_summary.py`, `code/label_flips.py` |

Example: `python3 code/make_robust.py --root . --tex <OUTPUT_DIR>/robust_table.tex --json <OUTPUT_DIR>/robust_data.json`.
Scripts that read per-prompt labels need the files as released here; those that read freeze files use only the kept fields.

## Rerunning an experiment

Environment: Python 3 with vLLM 0.30.0 (or SGLang 0.5.20 for the SGLang cells), one NVIDIA L40S GPU, and the Hugging
Face models and datasets named in the freeze files, at the revisions recorded there. Rebuild a freeze file, then:

```bash
export PROJECT_ROOT=<PROJECT_ROOT>   # directory of your choice holding jobs/, results/, envs/, logs/
python3 code/confirm_live.py --freeze <FREEZE_JSON> --freeze-sha <SHA256> --cell <CELL> --seed <SEED> --save-labels --out <OUTPUT_JSON>
```

`slurm_examples/` shows how the runs were submitted on a Slurm cluster, with the environment in a Singularity image.
A rebuilt freeze file has a new sha256; pass that value.
