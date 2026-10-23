#!/bin/bash

# ----- SLURM defaults (overridden by the sbatch flags built below) -----------
#SBATCH --job-name=cnode_rho_max_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=%x.log
# CPU-only job (no --gres=gpu requested).

# =============================================================================
# Like run_scaling_sweep.slurm, but sweeps the upper clamp on the adaptive ADMM
# penalty (--rho_max): stage 1 prompts for a space-separated list of rho_max
# values and submits ONE independent job per value (they run in parallel on the
# cluster). Each value gets its own job name, log and results .json (suffixed
# _rhomax<value>).
#
# NOTE: rho_max only matters with the adaptive-ρ scheme (on by default) and only
# affects the ADMM model, so the monolithic baseline defaults to OFF here.
# α, ν, hidden width and K are prompted explicitly so this sweep does not fall
# back to run_scaling_sweep.py's argparse defaults (α=0.8, ν=1e-4).
# =============================================================================

# =============================================================================
# STAGE 1 — login node: prompt for inputs, then submit one job per rho_max value.
# Runs only when NOT already inside a SLURM allocation ($SLURM_JOB_ID unset).
# =============================================================================
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Configure rho_max scaling-sweep jobs (press Enter to accept the [default]):"

    read -rp "  Job name              [cnode_rho_max_sweep]: " JOB_NAME
    JOB_NAME="${JOB_NAME:-cnode_rho_max_sweep}"

    read -rp "  CPU cores             [110]: " CPUS
    CPUS="${CPUS:-110}"

    read -rp "  RAM (GB)              [64G]: " MEM
    MEM="${MEM:-64G}"

    read -rp "  Wall-clock time       [15:00:00]: " TIME
    TIME="${TIME:-15:00:00}"

    # Base results path; each value's job writes <base>_rhomax<value>.json beside it.
    read -rp "  Results .json base    [data/bioreactor_simulation/${JOB_NAME}.json]: " RESULTS_BASE
    RESULTS_BASE="${RESULTS_BASE:-data/bioreactor_simulation/${JOB_NAME}.json}"

    read -rp "  B start               [80]: " B_START
    B_START="${B_START:-80}"

    read -rp "  B end                 [80]: " B_END
    B_END="${B_END:-80}"

    read -rp "  Training repeats/B    [2]: " REPEATS
    REPEATS="${REPEATS:-2}"

    read -rp "  Hidden layer width    [16]: " HIDDEN_SIZE
    HIDDEN_SIZE="${HIDDEN_SIZE:-16}"

    read -rp "  Batches per subproblem (K)  [1]: " K
    K="${K:-1}"

    # ν applies to every model, so it is prompted regardless of which baselines run.
    read -rp "  L2 reg (nu)           [1e-3]: " NU
    NU="${NU:-1e-3}"

    # Which models to train (y/n). rho_max only affects ADMM, so monolithic defaults to off.
    read -rp "  Run ADMM baseline        (y/n) [y]: " RUN_ADMM
    RUN_ADMM="${RUN_ADMM:-y}"
    read -rp "  Run monolithic baseline  (y/n) [n]: " RUN_MONO
    RUN_MONO="${RUN_MONO:-n}"
    read -rp "  Run sequential baseline  (y/n) [n]: " RUN_SEQ
    RUN_SEQ="${RUN_SEQ:-n}"

    # ADMM-only settings: prompt only when ADMM is actually being trained; otherwise these
    # are inert, so silently use the defaults. (rho_max itself is the swept dimension below.)
    if [[ "$RUN_ADMM" =~ ^[Yy] ]]; then
        read -rp "  Consensus aggregator  (mean|relaxed) [relaxed]: " AGGREGATOR
        AGGREGATOR="${AGGREGATOR:-relaxed}"

        read -rp "  Relaxation alpha      [1.5]: " ALPHA
        ALPHA="${ALPHA:-1.5}"

        read -rp "  Subproblem warmstart  (consensus|relaxed|previous) [consensus]: " WARMSTART
        WARMSTART="${WARMSTART:-consensus}"

        read -rp "  ADMM wall-clock budget/run (min, 0=no cap) [60]: " BUDGET
        BUDGET="${BUDGET:-60}"

        read -rp "  Max ADMM iterations/run  [3000]: " MAX_ADMM_ITERS
        MAX_ADMM_ITERS="${MAX_ADMM_ITERS:-3000}"
        read -rp "  IPOPT mu_init, late ADMM solves [0.1]: " LATE_MU_INIT
        LATE_MU_INIT="${LATE_MU_INIT:-0.1}"
    else
        AGGREGATOR="relaxed"
        ALPHA="1.5"
        WARMSTART="consensus"
        BUDGET="60"
        MAX_ADMM_ITERS="3000"
        LATE_MU_INIT="0.1"
    fi

    # The swept dimension: a space-separated list of upper clamps on the adaptive ρ.
    read -rp "  rho_max values        [50 200 1000]: " RHO_MAXS
    RHO_MAXS="${RHO_MAXS:-50 200 1000}"

    echo
    echo "About to submit ONE job per rho_max value:"
    echo "  job-name=$JOB_NAME  cpus-per-task=$CPUS  mem=$MEM  time=$TIME"
    echo "  results base=$RESULTS_BASE  B=$B_START..$B_END  repeats=$REPEATS  hidden width=$HIDDEN_SIZE  K=$K  nu=$NU"
    echo "  aggregator=$AGGREGATOR  alpha=$ALPHA  warmstart=$WARMSTART"
    echo "  admm budget=$BUDGET min  max admm iters=$MAX_ADMM_ITERS  late mu_init=$LATE_MU_INIT"
    echo "  run: admm=$RUN_ADMM monolithic=$RUN_MONO sequential=$RUN_SEQ"
    echo "  rho_max values=[$RHO_MAXS]"
    read -rp "Proceed? [y/N]: " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]([Ee][Ss])?$ ]]; then
        echo "Aborted; nothing submitted."
        exit 1
    fi

    for RHO_MAX in $RHO_MAXS; do
        # Per-value job name, log and results file (results: strip .json, append _rhomax<value>.json).
        R_JOB="${JOB_NAME}_rhomax${RHO_MAX}"
        # SLURM resolves --output relative to the submission dir, not the stage-2 cd, so anchor it.
        R_LOG="$HOME/CNODE/data/bioreactor_simulation/${R_JOB}.log"
        R_RESULTS="${RESULTS_BASE%.json}_rhomax${RHO_MAX}.json"
        echo "  submitting rho_max=$RHO_MAX -> $R_RESULTS (log $R_LOG)"
        sbatch \
            --job-name="$R_JOB" \
            --cpus-per-task="$CPUS" \
            --mem="$MEM" \
            --time="$TIME" \
            --output="$R_LOG" \
            "$0" "$R_RESULTS" "$B_START" "$B_END" "$REPEATS" "$AGGREGATOR" "$WARMSTART" \
                 "$RUN_ADMM" "$RUN_MONO" "$RUN_SEQ" "$BUDGET" "$MAX_ADMM_ITERS" \
                 "$HIDDEN_SIZE" "$K" "$ALPHA" "$NU" "$RHO_MAX" "$LATE_MU_INIT"
    done
    echo "All jobs submitted."
    exit 0
fi

# =============================================================================
# STAGE 2 — compute node: the actual job (runs under SLURM, $SLURM_JOB_ID set).
# One rho_max value per job.
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
TRAIN_DATA=data/bioreactor_simulation/raw_data/250trajectories_30points_0.005noise.csv
TEST_DATA=data/bioreactor_simulation/raw_data/testset_75trajectories_30points_0.0noise.csv

# Values passed by stage 1 (with defaults, in case the file is ever submitted
# directly with `sbatch <file> ...` positionals instead of interactively).
RESULTS="${1:-data/bioreactor_simulation/rho_max_sweep_results.json}"
B_START="${2:-80}"
B_END="${3:-80}"
REPEATS="${4:-2}"
AGGREGATOR="${5:-relaxed}"
WARMSTART="${6:-consensus}"
RUN_ADMM="${7:-y}"
RUN_MONO="${8:-n}"
RUN_SEQ="${9:-n}"
BUDGET="${10:-60}"
MAX_ADMM_ITERS="${11:-3000}"
HIDDEN_SIZE="${12:-16}"
K="${13:-1}"
ALPHA="${14:-1.5}"
NU="${15:-1e-3}"
RHO_MAX="${16:-50}"
LATE_MU_INIT="${17:-0.1}"

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
echo "Batches/subp K : $K"
echo "L2 reg (nu)    : $NU"
echo "Aggregator     : $AGGREGATOR"
echo "Relaxation α   : $ALPHA"
echo "Warmstart      : $WARMSTART"
echo "rho_max        : $RHO_MAX"
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
    --num_batches_per_subp "$K" \
    --num_training_repeats "$REPEATS" \
    --consensus_aggregator="$AGGREGATOR" \
    --relaxation_alpha "$ALPHA" \
    --subproblem_warmstart="$WARMSTART" \
    --l2_reg_param "$NU" \
    --rho_max "$RHO_MAX" \
    --training_clock_budget "$BUDGET" \
    --max_admm_iterations "$MAX_ADMM_ITERS" \
    --late_mu_init "$LATE_MU_INIT" \
    $ADMM_FLAG $MONO_FLAG $SEQ_FLAG
