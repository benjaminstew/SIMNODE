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
    B_START="${B_START:-2}"

    read -rp "  B end                 [100]: " B_END
    B_END="${B_END:-100}"

    read -rp "  Training repeats/B    [1]: " REPEATS
    REPEATS="${REPEATS:-1}"

    # Which models to train (y/n). Defaults: ADMM + monolithic on, sequential off.
    read -rp "  Run ADMM baseline        (y/n) [y]: " RUN_ADMM
    RUN_ADMM="${RUN_ADMM:-y}"
    read -rp "  Run monolithic baseline  (y/n) [y]: " RUN_MONO
    RUN_MONO="${RUN_MONO:-y}"
    read -rp "  Run sequential baseline  (y/n) [n]: " RUN_SEQ
    RUN_SEQ="${RUN_SEQ:-n}"

    # ADMM-only settings: prompt only when ADMM is actually being trained; otherwise these
    # are inert, so silently use the defaults.
    if [[ "$RUN_ADMM" =~ ^[Yy] ]]; then
        read -rp "  Consensus aggregator  (mean|relaxed) [relaxed]: " AGGREGATOR
        AGGREGATOR="${AGGREGATOR:-relaxed}"

        read -rp "  Subproblem warmstart  (consensus|relaxed|previous) [consensus]: " WARMSTART
        WARMSTART="${WARMSTART:-consensus}"

        read -rp "  ADMM wall-clock budget/run (min, 0=no cap) [60]: " BUDGET
        BUDGET="${BUDGET:-60}"

        # ADMM iteration budget: rule max(FLOOR, ceil(CONST / S)), S = largest subproblem size, or a fixed cap.
        read -rp "  ADMM iteration-budget rule max(floor, ceil(const/S))  (y/n) [y]: " ITER_RULE
        ITER_RULE="${ITER_RULE:-y}"
        if [[ "$ITER_RULE" =~ ^[Yy] ]]; then
            read -rp "  Iteration-budget constant [1500]: " ITER_CONST
            ITER_CONST="${ITER_CONST:-1500}"
            read -rp "  Iteration-budget floor    [80]: " ITER_FLOOR
            ITER_FLOOR="${ITER_FLOOR:-80}"
            MAX_ADMM_ITERS="2000"
        else
            read -rp "  Max ADMM iterations/run  [2000]: " MAX_ADMM_ITERS
            MAX_ADMM_ITERS="${MAX_ADMM_ITERS:-2000}"
            ITER_CONST="1500"
            ITER_FLOOR="80"
        fi
        read -rp "  IPOPT mu_init, late ADMM solves [0.1]: " LATE_MU_INIT
        LATE_MU_INIT="${LATE_MU_INIT:-0.1}"

        # Subproblem-size rule: num_subproblems = max(MIN_NUM_SUBP, ceil(B / MAX_TRAJS_PER_SUBP)),
        # split evenly. If off, every B uses a fixed K trajectories per subproblem instead.
        read -rp "  Subproblem-size rule  (y/n) [y]: " SUBP_RULE
        SUBP_RULE="${SUBP_RULE:-y}"
        if [[ "$SUBP_RULE" =~ ^[Yy] ]]; then
            read -rp "  Max trajectories per subproblem [20]: " MAX_TRAJS_PER_SUBP
            MAX_TRAJS_PER_SUBP="${MAX_TRAJS_PER_SUBP:-20}"
            read -rp "  Min number of subproblems      [4]: " MIN_NUM_SUBP
            MIN_NUM_SUBP="${MIN_NUM_SUBP:-4}"
            K="1"
        else
            read -rp "  Batches per subproblem (K)  [1]: " K
            K="${K:-1}"
            MAX_TRAJS_PER_SUBP="20"
            MIN_NUM_SUBP="4"
        fi
    else
        AGGREGATOR="relaxed"
        WARMSTART="consensus"
        BUDGET="60"
        MAX_ADMM_ITERS="2000"
        ITER_RULE="n"
        ITER_CONST="1500"
        ITER_FLOOR="80"
        LATE_MU_INIT="0.1"
        SUBP_RULE="n"
        MAX_TRAJS_PER_SUBP="20"
        MIN_NUM_SUBP="4"
        K="1"
    fi

    echo
    echo "About to submit:"
    echo "  job-name=$JOB_NAME  cpus-per-task=$CPUS  mem=$MEM  time=$TIME  output=$LOG"
    echo "  results=$RESULTS  B=$B_START..$B_END  repeats=$REPEATS  aggregator=$AGGREGATOR  warmstart=$WARMSTART"
    echo "  admm budget=$BUDGET min  max admm iters=$MAX_ADMM_ITERS  late mu_init=$LATE_MU_INIT"
    echo "  iteration-budget rule=$ITER_RULE (max($ITER_FLOOR, ceil($ITER_CONST/S)); fixed cap $MAX_ADMM_ITERS used if n)"
    echo "  subproblem-size rule=$SUBP_RULE (max trajs/subp=$MAX_TRAJS_PER_SUBP, min subps=$MIN_NUM_SUBP)  K=$K"
    echo "  run: admm=$RUN_ADMM monolithic=$RUN_MONO sequential=$RUN_SEQ"
    read -rp "Proceed? [y/N]: " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]([Ee][Ss])?$ ]]; then
        echo "Aborted; nothing submitted."
        exit 1
    fi

    # Re-run this same file through sbatch. The flags override the #SBATCH
    # defaults above; the positionals are read back in stage 2 as $1..$19.
    exec sbatch \
        --job-name="$JOB_NAME" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --output="$LOG" \
        "$0" "$RESULTS" "$B_START" "$B_END" "$REPEATS" "$AGGREGATOR" "$WARMSTART" \
             "$RUN_ADMM" "$RUN_MONO" "$RUN_SEQ" "$BUDGET" "$MAX_ADMM_ITERS" "$LATE_MU_INIT" \
             "$SUBP_RULE" "$MAX_TRAJS_PER_SUBP" "$MIN_NUM_SUBP" "$K" \
             "$ITER_RULE" "$ITER_CONST" "$ITER_FLOOR"
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

# Fixed for now (not requested as user inputs); edit here to change the datasets.
# TRAIN_DATA: CSV the training sets are drawn from (e.g. noisy observations).
# TEST_DATA : CSV the fixed test set is drawn from (e.g. noise-free trajectories). Set it
#             equal to TRAIN_DATA for the original single-file train/test split.
TRAIN_DATA=data/bioreactor_simulation/raw_data/250trajectories_30points_0.005noise.csv
TEST_DATA=data/bioreactor_simulation/raw_data/testset_75trajectories_30points_0.0noise.csv

# Values passed by stage 1 (with defaults, in case the file is ever submitted
# directly with `sbatch <file> ...` positionals instead of interactively).
RESULTS="${1:-data/bioreactor_simulation/scaling_sweep_results.json}"
B_START="${2:-2}"
B_END="${3:-100}"
REPEATS="${4:-1}"
AGGREGATOR="${5:-relaxed}"
WARMSTART="${6:-consensus}"
RUN_ADMM="${7:-y}"
RUN_MONO="${8:-y}"
RUN_SEQ="${9:-n}"
BUDGET="${10:-60}"
MAX_ADMM_ITERS="${11:-2000}"
LATE_MU_INIT="${12:-0.1}"
SUBP_RULE="${13:-n}"
MAX_TRAJS_PER_SUBP="${14:-20}"
MIN_NUM_SUBP="${15:-4}"
K="${16:-1}"
ITER_RULE="${17:-n}"
ITER_CONST="${18:-1500}"
ITER_FLOOR="${19:-80}"

# Translate the yes/no answers into CLI flags.
ADMM_FLAG="--run_admm_baseline"
[[ "$RUN_ADMM" =~ ^[Yy] ]] || ADMM_FLAG="--no-run_admm_baseline"
MONO_FLAG="--run_monolithic_baseline"
[[ "$RUN_MONO" =~ ^[Yy] ]] || MONO_FLAG="--no-run_monolithic_baseline"
ITER_RULE_FLAG="--no-admm_iter_rule"
[[ "$ITER_RULE" =~ ^[Yy] ]] && ITER_RULE_FLAG="--admm_iter_rule"
RULE_FLAG="--no-subp_size_rule"
[[ "$SUBP_RULE" =~ ^[Yy] ]] && RULE_FLAG="--subp_size_rule"
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
echo "Aggregator     : $AGGREGATOR"
echo "Warmstart      : $WARMSTART"
echo "ADMM budget/run: $BUDGET min (0=no cap)"
echo "Max ADMM iters : $MAX_ADMM_ITERS (ignored if iteration-budget rule on)"
echo "Iter-budget rule: $ITER_RULE (max($ITER_FLOOR, ceil($ITER_CONST/S)))"
echo "Late mu_init   : $LATE_MU_INIT"
echo "Subp-size rule : $SUBP_RULE (max trajs/subp $MAX_TRAJS_PER_SUBP, min subps $MIN_NUM_SUBP)"
echo "Batches/subp K : $K (ignored if subp-size rule on)"
echo "Run admm/mono/seq : $RUN_ADMM/$RUN_MONO/$RUN_SEQ"
echo "=================================================================="

python -m data.bioreactor_simulation.run_scaling_sweep \
    --train_data_path="$TRAIN_DATA" \
    --test_data_path="$TEST_DATA" \
    --results_path="$RESULTS" \
    --B_start "$B_START" \
    --B_end "$B_END" \
    --num_training_repeats "$REPEATS" \
    --consensus_aggregator="$AGGREGATOR" \
    --subproblem_warmstart="$WARMSTART" \
    --training_clock_budget "$BUDGET" \
    --max_admm_iterations "$MAX_ADMM_ITERS" \
    --late_mu_init "$LATE_MU_INIT" \
    --num_batches_per_subp "$K" \
    --max_trajs_per_subp "$MAX_TRAJS_PER_SUBP" \
    --min_num_subp "$MIN_NUM_SUBP" \
    --admm_iter_const "$ITER_CONST" \
    --admm_iter_floor "$ITER_FLOOR" \
    $ITER_RULE_FLAG $RULE_FLAG $ADMM_FLAG $MONO_FLAG $SEQ_FLAG
