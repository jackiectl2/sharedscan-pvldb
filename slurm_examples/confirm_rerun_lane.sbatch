#!/bin/bash
#SBATCH --job-name=sscan-crerun
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/%x-%j.out
# Same-configuration eager rerun (P1_eager, P1_eager_rerun) of one frozen confirmatory cell and seed. Usage:
#   confirm_rerun_lane.sbatch <cell> <seed> <freeze sha256>. Walltime per submission = 1.3 x the estimated runtime
# (twice the B1 P1_eager makespan of that cell and seed + startup measured in B1).
set -euo pipefail
# Lane wrapper: identical to confirm_vllm_lane.sbatch except for the runner and the output directory.
source ${PROJECT_ROOT:?}/jobs/lane_feed.sh
lane_feed || echo "lane_feed failed for lane ${LANE:-?}" >&2
R=${PROJECT_ROOT:?}
O=$R/results/confirm_rerun/${1}_s${2}.json
mkdir -p "$(dirname "$O")" && test -w "$(dirname "$O")"
[ ! -e "$O" ] || { echo "$O exists; refusing to overwrite" >&2; exit 1; }
module load singularity/4.4.1
C=/tmp/${USER}-sscan-${SLURM_JOB_ID}; mkdir -p $C; trap 'rm -rf $C' EXIT
export HF_HOME=$R/hf HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
singularity exec --nv --home $C --env HF_HOME=$HF_HOME,HF_HUB_OFFLINE=1,XDG_CACHE_HOME=$C,VLLM_CACHE_ROOT=$C/vllm,TRITON_CACHE_DIR=$C/triton,TORCHINDUCTOR_CACHE_DIR=$C/inductor $R/envs/sscan-vllm0.30.0-v1.sif \
  python3 $R/jobs/confirm_rerun.py --freeze $R/results/confirm/confirm_freeze.json --freeze-sha "$3" \
  --cell "$1" --seed "$2" --save-labels --out "$O"
echo CONFIRM_RERUN_OK
