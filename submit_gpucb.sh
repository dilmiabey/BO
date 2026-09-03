#!/bin/bash
#SBATCH --job-name=gpucb
#SBATCH --account=ih70
#SBATCH --partition=comp
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==========================================================================
# Runs one experiment config (all seeds x benchmarks x noise x algos).
#
# USAGE:
#   sbatch --export=CONFIG=configs/warm_candidate_current.yaml submit_experiment.slurm
# ==========================================================================

PROJECT_DIR=/scratch2/ih70/dilmia/PROJECT_2/gpucb

module purge
cd $PROJECT_DIR
mkdir -p logs results plots

source $PROJECT_DIR/venv/bin/activate

CONFIG=${CONFIG:-configs/warm_candidate_current.yaml}

echo "=========================================="
echo "Job       : $SLURM_JOB_ID"
echo "Config    : $CONFIG"
echo "Start     : $(date)"
echo "Node      : $(hostname)"
echo "=========================================="

export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python run.py --config $PROJECT_DIR/$CONFIG

echo "=========================================="
echo "Done : $(date)"
echo "=========================================="
