#!/bin/bash

# ----- SLURM defaults (overridden by the sbatch flags built below) -----------
#SBATCH --job-name=cnode_l2_reg_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=%x.log
# CPU-only job (no --gres=gpu requested).

# =============================================================================
# Like run_relaxation_alpha_sweep.slurm, but sweeps the L2 regularisation weight
# ν (--l2_reg_param) instead of the relaxation factor: stage 1 prompts for a
# space-separated list of ν values and submits ONE independent job per value
# (they run in parallel on the cluster). Each ν gets its own job name, log and
# results .json (suffixed _nu<value>).
#
# NOTE: ν is a shared objective parameter applied to BOTH the ADMM and the
# monolithic models (the same ν weights ‖θ‖² in each), so it is prompted
# regardless of which baselines are trained -- unlike the ADMM-only knobs
# (aggregator / warmstart / budget), which are only asked for when
# the ADMM baseline is actually being trained.
# =============================================================================

# =============================================================================
# STAGE 1 — login node: prompt for inputs, then submit one job per ν value.
# Runs only when NOT already inside a SLURM allocation ($SLURM_JOB_ID unset).
# =============================================================================
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Configure L2-regularisation (ν) scaling-sweep jobs (press Enter to accept the [default]):"

    read -rp "  Job name              [cnode_l2_reg_sweep]: " JOB_NAME
    JOB_NAME="${JOB_NAME:-cnode_l2_reg_sweep}"

    read -rp "  CPU cores             [110]: " CPUS
    CPUS="${CPUS:-110}"

    read -rp "  RAM (GB)              [64G]: " MEM
    MEM="${MEM:-64G}"

    read -rp "  Wall-clock time       [15:00:00]: " TIME
    TIME="${TIME:-15:00:00}"

    # Base results path; each ν's job writes <base>_nu<value>.json beside it.
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
    # are inert, so silently use the defaults. (ν itself is the swept dimension below and
    # applies to every model, so it is prompted outside this block.)
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

    # The swept dimension: a space-separated list of L2 regularisation weights ν.
    read -rp "  L2 reg values (nu)    [1e-6 1e-5 1e-4 1e-3 1e-2]: " NUS
    NUS="${NUS:-1e-6 1e-5 1e-4 1e-3 1e-2}"

    echo
    echo "About to submit ONE job per nu value:"
    echo "  job-name=$JOB_NAME  cpus-per-task=$CPUS  mem=$MEM  time=$TIME"
    echo "  results base=$RESULTS_BASE  B=$B_START..$B_END  repeats=$REPEATS"
    echo "  aggregator=$AGGREGATOR  warmstart=$WARMSTART  hidden width=$HIDDEN_SIZE"
    echo "  admm budget=$BUDGET min  max admm iters=$MAX_ADMM_ITERS  late mu_init=$LATE_MU_INIT"
    echo "  run: admm=$RUN_ADMM monolithic=$RUN_MONO sequential=$RUN_SEQ"
    echo "  nu values=[$NUS]"
    read -rp "Proceed? [y/N]: " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]([Ee][Ss])?$ ]]; then
        echo "Aborted; nothing submitted."
        exit 1
    fi

    for NU in $NUS; do
        # Per-nu job name, log and results file (results: strip .json, append _nu<value>.json).
        NU_JOB="${JOB_NAME}_nu${NU}"
        # SLURM resolves --output relative to the submission dir, not the stage-2 cd, so anchor it.
        NU_LOG="$HOME/CNODE/data/bioreactor_simulation/${NU_JOB}.log"
        NU_RESULTS="${RESULTS_BASE%.json}_nu${NU}.json"
        echo "  submitting nu=$NU -> $NU_RESULTS (log $NU_LOG)"
        sbatch \
            --job-name="$NU_JOB" \
            --cpus-per-task="$CPUS" \
            --mem="$MEM" \
            --time="$TIME" \
            --output="$NU_LOG" \
            "$0" "$NU_RESULTS" "$B_START" "$B_END" "$AGGREGATOR" "$WARMSTART" \
                 "$HIDDEN_SIZE" "$REPEATS" "$NU" "$RUN_ADMM" "$RUN_MONO" "$RUN_SEQ" "$BUDGET" "$MAX_ADMM_ITERS" "$LATE_MU_INIT"
    done
    echo "All jobs submitted."
    exit 0
fi

# =============================================================================
# STAGE 2 — compute node: the actual job (runs under SLURM, $SLURM_JOB_ID set).
# One nu value per job.
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
RESULTS="${1:-data/bioreactor_simulation/l2_reg_sweep_results.json}"
B_START="${2:-2}"
B_END="${3:-100}"
AGGREGATOR="${4:-relaxed}"
WARMSTART="${5:-consensus}"
HIDDEN_SIZE="${6:-16}"
REPEATS="${7:-1}"
NU="${8:-1e-4}"
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
echo "L2 reg (nu)    : $NU"
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
    --l2_reg_param "$NU" \
    --training_clock_budget "$BUDGET" \
    --max_admm_iterations "$MAX_ADMM_ITERS" \
    --late_mu_init "$LATE_MU_INIT" \
    $ADMM_FLAG $MONO_FLAG $SEQ_FLAG
