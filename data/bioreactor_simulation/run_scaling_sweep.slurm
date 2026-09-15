#!/bin/bash

# ----- SLURM defaults (overridden by the sbatch flags built below) -----------
#SBATCH --job-name=cnode_scaling_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=%x.log
# CPU-only job (no --gres=gpu requested).

# =============================================================================
# STAGE 1 — login node: prompt for inputs, then submit this file to the queue.
# Runs only when NOT already inside a SLURM allocation ($SLURM_JOB_ID unset).
# =============================================================================
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Configure scaling-sweep job (press Enter to accept the [default]):"

    read -rp "  Job name              [cnode_scaling_sweep]: " JOB_NAME
    JOB_NAME="${JOB_NAME:-cnode_scaling_sweep}"

    read -rp "  CPU cores             [110]: " CPUS
    CPUS="${CPUS:-110}"

    read -rp "  RAM (GB)              [64G]: " MEM
    MEM="${MEM:-64G}"

    read -rp "  Wall-clock time       [15:00:00]: " TIME
    TIME="${TIME:-15:00:00}"

    # SLURM resolves --output relative to the submission directory, not the `cd`
    # in stage 2, so anchor the log to an absolute path to keep it beside the .json.
    read -rp "  Output log name       [${JOB_NAME}.log]: " LOG
    LOG="$HOME/CNODE/data/bioreactor_simulation/${LOG:-${JOB_NAME}.log}"

    read -rp "  Results .json file    [data/bioreactor_simulation/${JOB_NAME}.json]: " RESULTS
    RESULTS="${RESULTS:-data/bioreactor_simulation/${JOB_NAME}.json}"

    read -rp "  B start               [2]: " B_START
    B_START="${B_START:-1}"

    read -rp "  B end                 [100]: " B_END
    B_END="${B_END:-100}"

    read -rp "  Consensus aggregator  (mean|under-relaxed) [under-relaxed]: " AGGREGATOR
    AGGREGATOR="${AGGREGATOR:-under-relaxed}"

    read -rp "  Subproblem warmstart  (consensus|relaxed|previous) [consensus]: " WARMSTART
    WARMSTART="${WARMSTART:-consensus}"

    echo
    echo "About to submit:"
    echo "  job-name=$JOB_NAME  cpus-per-task=$CPUS  mem=$MEM  time=$TIME  output=$LOG"
    echo "  results=$RESULTS  B=$B_START..$B_END  aggregator=$AGGREGATOR  warmstart=$WARMSTART"
    read -rp "Proceed? [y/N]: " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]([Ee][Ss])?$ ]]; then
        echo "Aborted; nothing submitted."
        exit 1
    fi

    # Re-run this same file through sbatch. The flags override the #SBATCH
    # defaults above; the positionals are read back in stage 2 as $1..$5.
    exec sbatch \
        --job-name="$JOB_NAME" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --output="$LOG" \
        "$0" "$RESULTS" "$B_START" "$B_END" "$AGGREGATOR" "$WARMSTART"
fi

# =============================================================================
# STAGE 2 — compute node: the actual job (runs under SLURM, $SLURM_JOB_ID set).
# =============================================================================
set -euo pipefail

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate cnode

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Unbuffered output so print() lands in the job log in real time.
export PYTHONUNBUFFERED=1

cd $HOME/CNODE

# Fixed for now (not requested as a user input); edit here to change the dataset.
DATA=data/bioreactor_simulation/raw_data/noise_level_01_eps0.0_B200.csv

# Values passed by stage 1 (with defaults, in case the file is ever submitted
# directly with `sbatch <file> ...` positionals instead of interactively).
RESULTS="${1:-data/bioreactor_simulation/scaling_sweep_results.json}"
B_START="${2:-1}"
B_END="${3:-100}"
AGGREGATOR="${4:-mean}"
WARMSTART="${5:-previous}"

echo "=================================================================="
echo "SLURM job name : ${SLURM_JOB_NAME:-?} (id ${SLURM_JOB_ID:-?})"
echo "CPU cores      : ${SLURM_CPUS_PER_TASK:-?}"
echo "RAM (MB/node)  : ${SLURM_MEM_PER_NODE:-?}"
echo "Data           : $DATA"
echo "Results        : $RESULTS"
echo "B start        : $B_START"
echo "B end          : $B_END"
echo "Aggregator     : $AGGREGATOR"
echo "Warmstart      : $WARMSTART"
echo "=================================================================="

python -m data.bioreactor_simulation.run_scaling_sweep \
    --data-path="$DATA" \
    --results-path="$RESULTS" \
    --B_start "$B_START" \
    --B_end "$B_END" \
    --consensus-aggregator="$AGGREGATOR" \
    --subproblem-warmstart="$WARMSTART"
