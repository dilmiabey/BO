#!/bin/bash
#SBATCH --job-name=gpucb
#SBATCH --account=ih70
#SBATCH --partition=comp          # CPU partition — adjust if your cluster uses a different name
#SBATCH --time=02:00:00           # 2h is generous; each seed typically takes ~20-40 min
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4         # GPyTorch uses MKL threads; 4 is a good balance
#SBATCH --array=0-4               # 5 jobs, one per seed (seeds 0..4)
#SBATCH --output=logs/%x_%a_%j.out
#SBATCH --error=logs/%x_%a_%j.err

# ==========================================================================
# GP-UCB experiment  |  5 seeds × 2 benchmarks × 2 noise types × 2 algos
#
#  Array task ID = seed index
#
#  Each job runs all 8 configurations for its seed:
#    sine     × homo   × gpucb
#    sine     × homo   × gpucb_pv
#    sine     × hetero × gpucb
#    sine     × hetero × gpucb_pv
#    branin   × homo   × gpucb
#    branin   × homo   × gpucb_pv
#    branin   × hetero × gpucb
#    branin   × hetero × gpucb_pv
#
#  Results written to:
#    results/seed_{seed}/{benchmark}_{noise}_{algorithm}.npy
#
#  After all 5 jobs complete, run plot_results.py separately (see README).
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
# Limit MKL/OpenBLAS threads to match allocated CPUs
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python run_seed.py \
    --seed          $SEED          \
    --results_dir   $PROJECT_DIR/results \
    --n_init        5              \
    --n_iter        50             \
    --n_cand        2000           \
    --gp_train_iter 50

echo "=========================================="
echo "Seed $SEED done : $(date)"
echo "=========================================="
