#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --output=<LOG_DIR>/%x-%j.out
# Block B8c (protocol/EXPERIMENT_BLOCKS.md): one frozen cell and seed. Usage: sbatch --time=... --job-name=...
#   b8c_vllm_lane.sbatch <cell> <seed> <freeze sha256>. Walltime per submission = 1.3 x the measured runtime of the same cell.
set -euo pipefail
R=${PROJECT_ROOT:?set to the root of this repository}
case "$1" in *_n[0-9]*) FZ=$R/results/b8/b8c_scale_freeze.json ;; *) FZ=$R/results/b8/b8c_freeze.json ;; esac
O=$R/results/b8/${1}_s${2}.json
[ "$(sha256sum "$FZ" | cut -d' ' -f1)" = "$3" ] || { echo "b8 freeze sha mismatch" >&2; exit 1; }
[ ! -e "$O" ] || { echo "$O exists; refusing to overwrite" >&2; exit 1; }
case "$1" in *_dist) grep -q '"other": 0,' $R/results/b8/smoke.json 2>/dev/null || { echo "smoke check missing or failed" >&2; exit 1; } ;; esac
module load singularity/4.4.1
C=$(mktemp -d); trap 'rm -rf $C' EXIT
export HF_HOME=$R/hf HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
singularity exec --nv --home $C --env HF_HOME=$HF_HOME,HF_HUB_OFFLINE=1,XDG_CACHE_HOME=$C,VLLM_CACHE_ROOT=$C/vllm,TRITON_CACHE_DIR=$C/triton,TORCHINDUCTOR_CACHE_DIR=$C/inductor ${SIF:?path to the Singularity image} \
  python3 $R/code/b8_live.py --freeze "$FZ" --freeze-sha "$3" --cell "$1" --seed "$2" --save-labels --out "$O"
echo B8_OK
