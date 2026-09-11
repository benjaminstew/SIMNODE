#!/bin/bash
#SBATCH --job-name=cnode_scaling_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=cnode_scaling_sweep.log
# CPU-only job (no --gres=gpu requested).

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate cnode

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Unbuffered output so print() lands in the job log in real time.
export PYTHONUNBUFFERED=1

cd $HOME/CNODE

# run_scaling_sweep.py now REQUIRES --data-path and --results-path (paths relative to $HOME/CNODE).
# Set DATA to the CSV you want to sweep on.
DATA=data/bioreactor_simulation/raw_data/noise_level_01_eps0.01_n200.csv
RESULTS=data/bioreactor_simulation/scaling_sweep_results.json
python -m data.bioreactor_simulation.run_scaling_sweep --data-path="$DATA" --results-path="$RESULTS"
