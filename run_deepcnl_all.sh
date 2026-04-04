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
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "deepcnl_env"; then
    echo "Creating deepcnl_env..."
    conda create -y -n deepcnl_env python=3.10
fi
conda activate deepcnl_env

# Install PyTorch with CUDA 11.8 (supports P100/V100/A40)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null
pip install pandas numpy scipy scikit-learn networkx matplotlib seaborn 2>/dev/null

echo "========================================"
echo "DeepCNL Graph Generation — All Years"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "Started: $(date)"
echo "========================================"

cd $SLURM_SUBMIT_DIR

python -m mtp2.phase0_save_graphs --ticker_num 470 --rare_ratio 0.002

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
