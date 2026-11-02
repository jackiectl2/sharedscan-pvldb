#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=<LOG_DIR>/%x-%j.out
# Block B7 step 2 (protocol/EXPERIMENT_BLOCKS.md): one frozen Qwen3-4B cell and seed (model from the freeze file). Usage: sbatch --time=... --job-name=...
#   b7_vllm_lane.sbatch <cell> <seed> <freeze sha256>. Walltime per submission = 1.3 x the estimated runtime
# (model-predicted policy makespans for that cell and seed + startup and warm-up).
set -euo pipefail
R=${PROJECT_ROOT:?set to the root of this repository}
# Preflight.
[ -f $R/code/sweep_live_b5.py ] || { echo "missing code/sweep_live_b5.py" >&2; exit 1; }
[ "$(sha256sum $R/results/b7/b7_freeze.json | cut -d' ' -f1)" = "$3" ] || { echo "b7 freeze sha mismatch" >&2; exit 1; }
# PubMed cells run only after the PubMed first-token smoke check has shown no reply other than Yes/No.
case "$1" in pubmed*) grep -q '"other": 0,' $R/results/b7/smoke_q4b_first_tokens_pubmed.json 2>/dev/null \
  || { echo "PubMed smoke check missing or failed" >&2; exit 1; } ;; esac
mkdir -p $R/results/b7
module load singularity/4.4.1
C=$(mktemp -d); trap 'rm -rf $C' EXIT
export HF_HOME=$R/hf HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
singularity exec --nv --home $C --env HF_HOME=$HF_HOME,HF_HUB_OFFLINE=1,XDG_CACHE_HOME=$C,VLLM_CACHE_ROOT=$C/vllm,TRITON_CACHE_DIR=$C/triton,TORCHINDUCTOR_CACHE_DIR=$C/inductor ${SIF:?path to the Singularity image} \
  python3 $R/code/sweep_live_b5.py --freeze $R/results/b7/b7_freeze.json --freeze-sha "$3" \
  --cell "$1" --seed "$2" --save-labels --out $R/results/b7/${1}_s${2}.json
echo B7_OK
