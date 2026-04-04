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

# Pre-requisite: run "bash setup_cluster_env.sh" once on login node
# Usage: mkdir -p logs && sbatch run_deepcnl_all.sh

eval "$(conda shell.bash hook)"
conda activate deepcnl_env

echo "========================================"
echo "DeepCNL Graph Generation — All Years"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Python: $(which python)"
echo "Started: $(date)"
echo "========================================"

cd $SLURM_SUBMIT_DIR

python -m mtp2.phase0_save_graphs --ticker_num 470 --rare_ratio 0.002

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
