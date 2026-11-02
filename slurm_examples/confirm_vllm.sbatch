#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=<LOG_DIR>/%x-%j.out
# Confirmatory round (CONFIRM_PREREG.md): one frozen cell and seed. Usage: sbatch --time=... --job-name=...
#   confirm_vllm.sbatch <cell> <seed> <freeze sha256>. Walltime per submission = 1.3 x the estimated runtime
# (model-predicted policy makespans for that cell and seed + startup and warm-up).
set -euo pipefail
R=${PROJECT_ROOT:?set to the root of this repository}
module load singularity/4.4.1
C=$(mktemp -d); trap 'rm -rf $C' EXIT
export HF_HOME=$R/hf HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
singularity exec --nv --home $C --env HF_HOME=$HF_HOME,HF_HUB_OFFLINE=1,XDG_CACHE_HOME=$C,VLLM_CACHE_ROOT=$C/vllm,TRITON_CACHE_DIR=$C/triton,TORCHINDUCTOR_CACHE_DIR=$C/inductor ${SIF:?path to the Singularity image} \
  python3 $R/code/confirm_live.py --freeze $R/results/confirm/confirm_freeze.json --freeze-sha "$3" \
  --cell "$1" --seed "$2" --save-labels --out $R/results/confirm/${1}_s${2}.json
echo CONFIRM_OK
