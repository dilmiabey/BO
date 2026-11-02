#!/bin/bash
#SBATCH --job-name=gpucb
#SBATCH --account=ih70
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=logs/%x_seed%a_%A.out
#SBATCH --error=logs/%x_seed%a_%A.err

# ==========================================================================
# Array job: one task per seed.
# Array task ID = seed (0..4).  Each task runs all benchmark/noise/algo
# combos for its single seed.
#
# USAGE:
#   sbatch --export=CONFIG=configs/cold_candidate_reference_tabpfn.yaml submit_experiment.slurm
#
# Each seed gets its own log file: logs/gpucb_seed{0..4}_{jobid}.out
# ==========================================================================

PROJECT_DIR=/scratch2/ih70/dilmia/PROJECT_2/gpucb

module purge
module load cuda

cd $PROJECT_DIR
mkdir -p logs results plots

source $PROJECT_DIR/venv/bin/activate

CONFIG=${CONFIG:-configs/warm_candidate_current.yaml}
SEED=$SLURM_ARRAY_TASK_ID

echo "=========================================="
echo "Job       : $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Config    : $CONFIG"
echo "Seed      : $SEED"
echo "Start     : $(date)"
echo "Node      : $(hostname)"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TABPFN_MODEL_CACHE_DIR=$PROJECT_DIR/tabpfn_cache
export HF_HOME=$PROJECT_DIR/hf_cache
export HF_HUB_OFFLINE=1

# Fail fast if PyTorch can't see the GPU (e.g. a CPU-only torch wheel)
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available to torch'; print('torch', torch.__version__, '->', torch.cuda.get_device_name(0))" || exit 1

python run.py --config $PROJECT_DIR/$CONFIG --seed $SEED

echo "=========================================="
echo "Seed $SEED done : $(date)"
echo "=========================================="
