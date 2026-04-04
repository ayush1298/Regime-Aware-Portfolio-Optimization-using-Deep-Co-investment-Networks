#!/bin/bash
#SBATCH --job-name=deepcnl_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpupart_24hour
#SBATCH --time=23:59:00
#SBATCH --output=logs/deepcnl_all_%j.out
#SBATCH --error=logs/deepcnl_all_%j.err

# Usage: mkdir -p logs && sbatch run_deepcnl_all.sh

# ---- Environment Setup ----
# Initialize conda (required in non-interactive SLURM shells)
eval "$(conda shell.bash hook)"

# Create env if it doesn't exist, then activate
if ! conda env list | grep -q "deepcnl_env"; then
    echo "Creating deepcnl_env..."
    conda create -y -n deepcnl_env python=3.10
fi
conda activate deepcnl_env

# Install dependencies if missing
pip install torch torchvision pandas numpy scipy scikit-learn networkx matplotlib seaborn 2>/dev/null

echo "========================================"
echo "DeepCNL Graph Generation — All Years"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Python: $(which python)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "========================================"

cd $SLURM_SUBMIT_DIR

python -m mtp2.phase0_save_graphs --ticker_num 470 --rare_ratio 0.002

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
