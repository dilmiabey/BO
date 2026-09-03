#!/bin/bash
#SBATCH --job-name=gpucb_plot
#SBATCH --account=ih70
#SBATCH --partition=comp
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# USAGE:
#   JID=$(sbatch --parsable --export=CONFIG=configs/xxx.yaml submit_experiment.slurm)
#   sbatch --dependency=afterok:$JID --export=CONFIG=configs/xxx.yaml submit_plot.slurm

PROJECT_DIR=/scratch2/ih70/dilmia/PROJECT_2/gpucb
module purge
cd $PROJECT_DIR
source $PROJECT_DIR/venv/bin/activate

CONFIG=${CONFIG:-configs/warm_candidate_current.yaml}
export PYTHONUNBUFFERED=1

echo "Plotting for config: $CONFIG"
python plot_results.py --config $PROJECT_DIR/$CONFIG
echo "Done: $(date)"
