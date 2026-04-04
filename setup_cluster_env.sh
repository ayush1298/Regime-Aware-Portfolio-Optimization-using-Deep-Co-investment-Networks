#!/bin/bash
# Run this ONCE on the cluster login node (NOT as a SLURM job)
# Usage: bash setup_cluster_env.sh

eval "$(conda shell.bash hook)"

echo "Creating deepcnl_env..."
conda create -y -n deepcnl_env python=3.10

conda activate deepcnl_env

echo "Installing PyTorch (CUDA 11.8 for P100/V100/A40)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "Installing other dependencies..."
pip install pandas numpy scipy scikit-learn networkx matplotlib seaborn

echo ""
echo "Verifying..."
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)"
python -c "import pandas, networkx, sklearn; print('All packages OK')"

echo ""
echo "Done! Now you can submit: sbatch run_deepcnl_all.sh"
