#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=<LOG_DIR>/%x-%j.out
# Block B8 (refine-logs/EXPERIMENT_PLAN.md): one frozen cell and seed. Usage: sbatch --time=... --job-name=...
#   b8_vllm_lane.sbatch <cell> <seed> <freeze sha256>. Walltime per submission = 1.3 x the estimated runtime
# (model-predicted policy makespans for that cell and seed + startup and warm-up).
set -euo pipefail
R=<PROJECT_ROOT>
case "$1" in *_n[0-9]*) FZ=$R/results/b8/b8_scale_freeze.json ;; *) FZ=$R/results/b8/b8_freeze.json ;; esac
O=$R/results/b8/${1}_s${2}.json
# Preflight before feeding the lane: a deployment error must not propagate down the lane.
[ "$(sha256sum "$FZ" | cut -d' ' -f1)" = "$3" ] || { echo "b8 freeze sha mismatch" >&2; exit 1; }
[ ! -e "$O" ] || { echo "$O exists; refusing to overwrite" >&2; exit 1; }
case "$1" in *_dist) grep -q '"other": 0,' $R/results/b8/smoke.json 2>/dev/null || { echo "smoke check missing or failed" >&2; exit 1; } ;; esac
source $R/jobs/lane_feed.sh
lane_feed || echo "lane_feed failed for lane ${LANE:-?}" >&2
module load singularity/4.4.1
C=/tmp/${USER}-vldb1-${SLURM_JOB_ID}; mkdir -p $C; trap 'rm -rf $C' EXIT
export HF_HOME=$R/hf HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
singularity exec --nv --home $C --env HF_HOME=$HF_HOME,HF_HUB_OFFLINE=1,XDG_CACHE_HOME=$C,VLLM_CACHE_ROOT=$C/vllm,TRITON_CACHE_DIR=$C/triton,TORCHINDUCTOR_CACHE_DIR=$C/inductor $R/envs/vldb1-vllm0.30.0-v1.sif \
  python3 $R/jobs/b8_live.py --freeze "$FZ" --freeze-sha "$3" --cell "$1" --seed "$2" --save-labels --out "$O"
echo B8_OK
