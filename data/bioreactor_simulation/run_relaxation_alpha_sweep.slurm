#!/bin/bash

# ----- SLURM defaults (overridden by the sbatch flags built below) -----------
#SBATCH --job-name=cnode_relaxation_alpha_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=%x.log
# CPU-only job (no --gres=gpu requested).

# =============================================================================
# Like run_warm_start_setting_sweep.slurm, but sweeps the under-relaxation
# factor alpha instead of the warmstart setting: stage 1 prompts for a space-
# separated list of alpha values and submits ONE independent job per value
# (they run in parallel on the cluster). Each alpha gets its own job name, log
# and results .json (suffixed _a<alpha>).
#
# NOTE: alpha only has any effect when the consensus aggregator is 'relaxed'
# (θ̂ = α·θ + (1-α)·z). Under 'mean' the alpha value is ignored, so this sweep
# defaults the aggregator to relaxed.
# =============================================================================

# =============================================================================
# STAGE 1 — login node: prompt for inputs, then submit one job per alpha value.
# Runs only when NOT already inside a SLURM allocation ($SLURM_JOB_ID unset).
# =============================================================================
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Configure relaxation-alpha scaling-sweep jobs (press Enter to accept the [default]):"

    read -rp "  Job name              [cnode_relaxation_alpha_sweep]: " JOB_NAME
    JOB_NAME="${JOB_NAME:-cnode_relaxation_alpha_sweep}"

    read -rp "  CPU cores             [110]: " CPUS
    CPUS="${CPUS:-110}"

    read -rp "  RAM (GB)              [64G]: " MEM
    MEM="${MEM:-64G}"

    read -rp "  Wall-clock time       [15:00:00]: " TIME
    TIME="${TIME:-15:00:00}"

    # Base results path; each alpha's job writes <base>_a<alpha>.json beside it.
    read -rp "  Results .json base    [data/bioreactor_simulation/${JOB_NAME}.json]: " RESULTS_BASE
    RESULTS_BASE="${RESULTS_BASE:-data/bioreactor_simulation/${JOB_NAME}.json}"

    read -rp "  B start               [2]: " B_START
    B_START="${B_START:-2}"

    read -rp "  B end                 [100]: " B_END
    B_END="${B_END:-100}"

    read -rp "  Training repeats/B    [1]: " REPEATS
    REPEATS="${REPEATS:-1}"

    read -rp "  Hidden layer width    [16]: " HIDDEN_SIZE
    HIDDEN_SIZE="${HIDDEN_SIZE:-16}"

    # Which models to train (y/n). Defaults: ADMM + monolithic on, sequential off.
    read -rp "  Run ADMM baseline        (y/n) [y]: " RUN_ADMM
    RUN_ADMM="${RUN_ADMM:-y}"
    read -rp "  Run monolithic baseline  (y/n) [y]: " RUN_MONO
    RUN_MONO="${RUN_MONO:-y}"
    read -rp "  Run sequential baseline  (y/n) [n]: " RUN_SEQ
    RUN_SEQ="${RUN_SEQ:-n}"

    # ADMM-only settings: prompt only when ADMM is actually being trained; otherwise these
    # are inert, so silently use the defaults. (Alpha itself is the swept dimension below.)
    if [[ "$RUN_ADMM" =~ ^[Yy] ]]; then
        read -rp "  Consensus aggregator  (mean|relaxed) [relaxed]: " AGGREGATOR
        AGGREGATOR="${AGGREGATOR:-relaxed}"

        read -rp "  Subproblem warmstart  (consensus|relaxed|previous) [consensus]: " WARMSTART
        WARMSTART="${WARMSTART:-consensus}"

        read -rp "  ADMM wall-clock budget/run (min, 0=no cap) [60]: " BUDGET
        BUDGET="${BUDGET:-60}"

        read -rp "  Max ADMM iterations/run  [2000]: " MAX_ADMM_ITERS
        MAX_ADMM_ITERS="${MAX_ADMM_ITERS:-2000}"
        read -rp "  IPOPT mu_init, late ADMM solves [0.1]: " LATE_MU_INIT
        LATE_MU_INIT="${LATE_MU_INIT:-0.1}"
    else
        AGGREGATOR="relaxed"
        WARMSTART="consensus"
        BUDGET="60"
        MAX_ADMM_ITERS="2000"
        LATE_MU_INIT="0.1"
    fi

    # The swept dimension: a space-separated list of under-relaxation alpha values.
    read -rp "  Relaxation alpha values  [0.7 0.8 1.0 1.5]: " ALPHAS
    ALPHAS="${ALPHAS:-0.7 0.8 1.0 1.5}"

    echo
    echo "About to submit ONE job per alpha value:"
    echo "  job-name=$JOB_NAME  cpus-per-task=$CPUS  mem=$MEM  time=$TIME"
    echo "  results base=$RESULTS_BASE  B=$B_START..$B_END  repeats=$REPEATS"
    echo "  aggregator=$AGGREGATOR  warmstart=$WARMSTART  hidden width=$HIDDEN_SIZE"
    echo "  admm budget=$BUDGET min  max admm iters=$MAX_ADMM_ITERS  late mu_init=$LATE_MU_INIT"
    echo "  run: admm=$RUN_ADMM monolithic=$RUN_MONO sequential=$RUN_SEQ"
    echo "  alpha values=[$ALPHAS]"
    read -rp "Proceed? [y/N]: " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]([Ee][Ss])?$ ]]; then
        echo "Aborted; nothing submitted."
        exit 1
    fi

    for ALPHA in $ALPHAS; do
        # Per-alpha job name, log and results file (results: strip .json, append _a<alpha>.json).
        A_JOB="${JOB_NAME}_a${ALPHA}"
        # SLURM resolves --output relative to the submission dir, not the stage-2 cd, so anchor it.
        A_LOG="$HOME/CNODE/data/bioreactor_simulation/${A_JOB}.log"
        A_RESULTS="${RESULTS_BASE%.json}_a${ALPHA}.json"
        echo "  submitting alpha=$ALPHA -> $A_RESULTS (log $A_LOG)"
        sbatch \
            --job-name="$A_JOB" \
            --cpus-per-task="$CPUS" \
            --mem="$MEM" \
            --time="$TIME" \
            --output="$A_LOG" \
            "$0" "$A_RESULTS" "$B_START" "$B_END" "$AGGREGATOR" "$WARMSTART" \
                 "$HIDDEN_SIZE" "$REPEATS" "$ALPHA" "$RUN_ADMM" "$RUN_MONO" "$RUN_SEQ" "$BUDGET" "$MAX_ADMM_ITERS" "$LATE_MU_INIT"
    done
    echo "All jobs submitted."
    exit 0
fi

# =============================================================================
# STAGE 2 — compute node: the actual job (runs under SLURM, $SLURM_JOB_ID set).
# One alpha value per job.
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

# Fixed for now (not requested as user inputs); edit here to change the datasets.
# TRAIN_DATA: CSV the training sets are drawn from (e.g. noisy observations).
# TEST_DATA : CSV the fixed test set is drawn from (e.g. noise-free trajectories). Set it
#             equal to TRAIN_DATA for the original single-file train/test split.
TRAIN_DATA=data/bioreactor_simulation/raw_data/150trajectories_noise_0.0.csv
TEST_DATA=data/bioreactor_simulation/raw_data/150trajectories_noise_0.0.csv

# Values passed by stage 1 (with defaults, in case the file is ever submitted
# directly with `sbatch <file> ...` positionals instead of interactively).
RESULTS="${1:-data/bioreactor_simulation/relaxation_alpha_sweep_results.json}"
B_START="${2:-2}"
B_END="${3:-100}"
AGGREGATOR="${4:-relaxed}"
WARMSTART="${5:-consensus}"
HIDDEN_SIZE="${6:-16}"
REPEATS="${7:-1}"
ALPHA="${8:-0.8}"
RUN_ADMM="${9:-y}"
RUN_MONO="${10:-y}"
RUN_SEQ="${11:-n}"
BUDGET="${12:-60}"
MAX_ADMM_ITERS="${13:-2000}"
LATE_MU_INIT="${14:-0.1}"

# Translate the yes/no answers into CLI flags.
ADMM_FLAG="--run_admm_baseline"
[[ "$RUN_ADMM" =~ ^[Yy] ]] || ADMM_FLAG="--no-run_admm_baseline"
MONO_FLAG="--run_monolithic_baseline"
[[ "$RUN_MONO" =~ ^[Yy] ]] || MONO_FLAG="--no-run_monolithic_baseline"
SEQ_FLAG="--no-run_sequential_baseline"
[[ "$RUN_SEQ" =~ ^[Yy] ]] && SEQ_FLAG="--run_sequential_baseline"

echo "=================================================================="
echo "SLURM job name : ${SLURM_JOB_NAME:-?} (id ${SLURM_JOB_ID:-?})"
echo "CPU cores      : ${SLURM_CPUS_PER_TASK:-?}"
echo "RAM (MB/node)  : ${SLURM_MEM_PER_NODE:-?}"
echo "Train data     : $TRAIN_DATA"
echo "Test data      : $TEST_DATA"
echo "Results        : $RESULTS"
echo "B start        : $B_START"
echo "B end          : $B_END"
echo "Repeats/B      : $REPEATS"
echo "Hidden width   : $HIDDEN_SIZE"
echo "Aggregator     : $AGGREGATOR"
echo "Warmstart      : $WARMSTART"
echo "Relaxation α   : $ALPHA"
echo "ADMM budget/run: $BUDGET min (0=no cap)"
echo "Max ADMM iters : $MAX_ADMM_ITERS"
echo "Late mu_init   : $LATE_MU_INIT"
echo "Run admm/mono/seq : $RUN_ADMM/$RUN_MONO/$RUN_SEQ"
echo "=================================================================="

python -m data.bioreactor_simulation.run_scaling_sweep \
    --train_data_path="$TRAIN_DATA" \
    --test_data_path="$TEST_DATA" \
    --results_path="$RESULTS" \
    --B_start "$B_START" \
    --B_end "$B_END" \
    --hidden_layer_size "$HIDDEN_SIZE" \
    --num_training_repeats "$REPEATS" \
    --consensus_aggregator="$AGGREGATOR" \
    --subproblem_warmstart="$WARMSTART" \
    --relaxation_alpha "$ALPHA" \
    --training_clock_budget "$BUDGET" \
    --max_admm_iterations "$MAX_ADMM_ITERS" \
    --late_mu_init "$LATE_MU_INIT" \
    $ADMM_FLAG $MONO_FLAG $SEQ_FLAG
