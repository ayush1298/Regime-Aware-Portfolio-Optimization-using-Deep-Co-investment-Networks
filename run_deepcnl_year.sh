#!/bin/bash
#SBATCH --job-name=deepcnl_year
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpupart_24hour
#SBATCH --time=24:00:00
#SBATCH --output=logs/deepcnl_%j.out
#SBATCH --error=logs/deepcnl_%j.err

# Usage: sbatch run_deepcnl_year.sh <YEAR>
# Example: sbatch run_deepcnl_year.sh 2010

YEAR=${1:-2010}

echo "========================================"
echo "DeepCNL Graph Generation — Year: $YEAR"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "========================================"

cd $SLURM_SUBMIT_DIR

# Activate conda environment (adjust name if different)
source activate deepcnl_env 2>/dev/null || conda activate deepcnl_env 2>/dev/null || true

python -m mtp2.phase0_save_graphs --ticker_num 470 --rare_ratio 0.002 --year $YEAR

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
