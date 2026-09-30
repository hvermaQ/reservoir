#!/bin/bash
#SBATCH --job-name=qrc-sweep
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/qrc_%A_%a.out
#
# Sharded sweep over a SLURM array. Each array task takes one shard of the unit
# list and writes its own per-unit files, so no two tasks ever write the same
# path and no coordination is needed between them.
#
#   sbatch --array=0-15 scripts/slurm_array.sh configs/timeseries_seeds.json
#
# IMPORTANT: run `prepare` ONCE, before submitting. If every array task builds
# the dataset cache itself, they all parse the same 1.1 GB of CSV simultaneously
# and race on the same cache files.
#
#   python3 scripts/qrc.py prepare -c configs/timeseries_seeds.json
#
set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/slurm_array.sh <config.json>}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
JOBS="${SLURM_CPUS_PER_TASK:-4}"

# One BLAS thread per worker; the driver sets these too, but exporting here
# covers anything that imports numpy before the driver does.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p logs
echo "shard ${SHARD}/${NSHARDS}  jobs=${JOBS}  config=${CONFIG}"

python3 scripts/qrc.py run -c "${CONFIG}" \
    --shard "${SHARD}/${NSHARDS}" \
    --jobs "${JOBS}" \
    --start-method fork

# Aggregate once, from the last shard to finish, or run it by hand afterwards:
#   python3 scripts/qrc.py aggregate -c configs/timeseries_seeds.json
