#!/bin/bash
#SBATCH --job-name=gpucb_plot
#SBATCH --account=ih70
#SBATCH --partition=comp
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==========================================================================
# Run AFTER all seed jobs complete.
# Either submit with --dependency=afterok:<array_job_id>
# or run interactively:  sbatch submit_plot.slurm
# ==========================================================================

PROJECT_DIR=/scratch2/ih70/dilmia/PROJECT_2/gpucb

module purge
cd $PROJECT_DIR
source $PROJECT_DIR/venv/bin/activate

export PYTHONUNBUFFERED=1

python plot_results.py \
    --results_dir $PROJECT_DIR/results \
    --plots_dir   $PROJECT_DIR/plots   \
    --n_seeds     5

echo "Plots written to $PROJECT_DIR/plots/"
