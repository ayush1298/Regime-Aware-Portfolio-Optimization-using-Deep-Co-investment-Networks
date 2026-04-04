#!/bin/bash
#SBATCH --job-name=deepcnl_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpupart_24hour
#SBATCH --time=24:00:00
#SBATCH --output=logs/deepcnl_all_%j.out
#SBATCH --error=logs/deepcnl_all_%j.err

# Usage: mkdir -p logs && sbatch run_deepcnl_all.sh

echo "========================================"
echo "DeepCNL Graph Generation — All Years"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "========================================"

cd $SLURM_SUBMIT_DIR

# Activate conda environment (adjust name if different)
source activate deepcnl_env 2>/dev/null || conda activate deepcnl_env 2>/dev/null || true

python -m mtp2.phase0_save_graphs --ticker_num 470 --rare_ratio 0.002

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
