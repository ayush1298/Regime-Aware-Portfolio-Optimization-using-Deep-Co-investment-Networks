#!/bin/bash
# Submit all 7 years as parallel SLURM jobs
# Usage: bash run_all_years.sh

mkdir -p logs

echo "Submitting DeepCNL jobs for years 2010-2016..."
echo ""

for year in 2010 2011 2012 2013 2014 2015 2016; do
    JOB_ID=$(sbatch --job-name=dcnl_${year} run_deepcnl_year.sh $year | awk '{print $4}')
    echo "Year $year -> Job $JOB_ID"
done

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "After all jobs complete, verify outputs:"
echo "  ls -la outputs/graphs/"
echo "  # Expect 14 files: deepcnl_graph_20{10..16}.pkl + pcc_graph_20{10..16}.pkl"
