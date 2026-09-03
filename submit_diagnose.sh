#!/bin/bash
#SBATCH --job-name=gpucb_diag
#SBATCH --account=ih70
#SBATCH --partition=comp
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%a_%j.out
#SBATCH --error=logs/%x_%a_%j.err

# ==========================================================================
# Posterior variance diagnostic — 2 jobs (one per benchmark).
#
# USAGE:
#   sbatch submit_diagnose.slurm
#
#   Array task 0 = sine,  task 1 = branin
# ==========================================================================

PROJECT_DIR=/scratch2/ih70/dilmia/PROJECT_2/gpucb

module purge
cd $PROJECT_DIR
mkdir -p logs
source $PROJECT_DIR/venv/bin/activate

BENCHMARKS=(sine branin)
BENCH=${BENCHMARKS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Job       : $SLURM_JOB_ID  array task $SLURM_ARRAY_TASK_ID"
echo "Benchmark : $BENCH"
echo "Start     : $(date)"
echo "Node      : $(hostname)"
echo "=========================================="

export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python diagnose_variance.py --benchmark $BENCH

echo "=========================================="
echo "Done : $(date)"
echo "=========================================="
