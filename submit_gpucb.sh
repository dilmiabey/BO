#!/bin/bash
#SBATCH --job-name=gpucb
#SBATCH --account=ih70
#SBATCH --partition=comp          # CPU partition — adjust if needed (sinfo)
#SBATCH --time=04:00:00           # L-BFGS is slower than grid search; 4h is safe
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4               # 5 jobs, one per seed (seeds 0..4)
#SBATCH --output=logs/%x_%a_%j.out
#SBATCH --error=logs/%x_%a_%j.err

# ==========================================================================
# GP-UCB experiment  |  L-BFGS acquisition optimisation
# 5 seeds x 2 benchmarks x 2 noise types x 2 algorithms = 40 trials
#
# Each array task = one seed, runs all 8 configs sequentially.
#
# Acquisition optimisation:
#   --n_restarts 10   : 10 random starting points per BO iteration
#   --lbfgs_steps 100 : up to 100 L-BFGS steps per restart
#
# Results: results/seed_{seed}/{benchmark}_{noise}_{algorithm}.npy
# ==========================================================================

PROJECT_DIR=/scratch2/ih70/dilmia/PROJECT_2/gpucb

module purge
# No CUDA module needed — CPU-only experiment

cd $PROJECT_DIR
mkdir -p logs results plots

source $PROJECT_DIR/venv/bin/activate

SEED=$SLURM_ARRAY_TASK_ID

echo "=========================================="
echo "Job       : $SLURM_JOB_ID  array task $SLURM_ARRAY_TASK_ID"
echo "Seed      : $SEED"
echo "Start     : $(date)"
echo "Node      : $(hostname)"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "=========================================="

export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python run_seed.py \
    --seed          $SEED                  \
    --results_dir   $PROJECT_DIR/results   \
    --n_init        5                      \
    --n_iter        50                     \
    --n_restarts    10                     \
    --lbfgs_steps   100                    \
    --gp_train_iter 50

echo "=========================================="
echo "Seed $SEED done : $(date)"
echo "=========================================="
