#!/bin/bash

# =============================================================================
# Submit a multi-block ADMM scaling sweep as many small jobs: one call of
# run_scaling_sweep.slurm per block of CHUNK consecutive B (= T) values.
#
# Run on the login node:  bash data/bioreactor_simulation/submit_mb_chunks.slurm
# It prompts for the first and last B, then submits jobs named mb_<first>to<last>
# (e.g. mb_10to12, mb_13to15, ...); the last block may be shorter if the range
# does not divide evenly. Each job writes data/bioreactor_simulation/<name>.json
# and <name>.log.
#
# Every job uses the settings of the interactive mb_10to12 example:
#   10 cores, 32G, 15:00:00, 5 repeats, ADMM only (no monolithic / sequential),
#   relaxed aggregator, consensus warm start, standard ADMM,
#   60-min ADMM budget/run, iteration-budget rule max(80, ceil(1500/S)),
#   late mu_init 0.1, subproblem-size rule (max 20 trajs/subp, min 4 subps).
# Settings not prompted by run_scaling_sweep.slurm (alpha, nu, rho_max, hidden
# width, ...) fall back to run_scaling_sweep.py's argparse defaults, exactly as
# they do for an interactive submission.
# =============================================================================
set -euo pipefail

# run_scaling_sweep.slurm lives next to this file
SWEEP_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_scaling_sweep.slurm"
[[ -f "$SWEEP_SCRIPT" ]] || { echo "Cannot find $SWEEP_SCRIPT"; exit 1; }

read -rp "  First B (T) value         : " B_FIRST
read -rp "  Last B (T) value          : " B_LAST
read -rp "  B values per job          [3]: " CHUNK
CHUNK="${CHUNK:-3}"
for v in "$B_FIRST" "$B_LAST" "$CHUNK"; do
    [[ "$v" =~ ^[0-9]+$ ]] || { echo "Not a positive integer: '$v'"; exit 1; }
done
(( B_LAST >= B_FIRST && CHUNK >= 1 )) || { echo "Need last B >= first B and B values per job >= 1."; exit 1; }

# ----- fixed job settings (mb_10to12 example) -----
CPUS=10
MEM=32G
TIME=15:00:00
REPEATS=5
RUN_ADMM=y; RUN_MONO=n; RUN_SEQ=n
AGGREGATOR=relaxed
WARMSTART=consensus
BUDGET=60                 # ADMM wall-clock budget per run (min)
MAX_ADMM_ITERS=2000       # unused while the iteration-budget rule is on
LATE_MU_INIT=0.1
SUBP_RULE=y; MAX_TRAJS_PER_SUBP=20; MIN_NUM_SUBP=4; K=1
ITER_RULE=y; ITER_CONST=1500; ITER_FLOOR=80

# ----- build the job list -----
STARTS=(); ENDS=()
for (( s = B_FIRST; s <= B_LAST; s += CHUNK )); do
    e=$(( s + CHUNK - 1 )); (( e > B_LAST )) && e=$B_LAST
    STARTS+=("$s"); ENDS+=("$e")
done
NJOBS=${#STARTS[@]}

echo
echo "About to submit $NJOBS jobs (B $B_FIRST..$B_LAST, $CHUNK per job), each:"
echo "  cpus=$CPUS  mem=$MEM  time=$TIME  repeats=$REPEATS  run admm/mono/seq=$RUN_ADMM/$RUN_MONO/$RUN_SEQ"
echo "  aggregator=$AGGREGATOR  warmstart=$WARMSTART  admm budget=${BUDGET} min  late mu_init=$LATE_MU_INIT"
echo "  iteration-budget rule=$ITER_RULE (max($ITER_FLOOR, ceil($ITER_CONST/S)))"
echo "  subproblem-size rule=$SUBP_RULE (max trajs/subp=$MAX_TRAJS_PER_SUBP, min subps=$MIN_NUM_SUBP)"
echo "  jobs: $(for i in "${!STARTS[@]}"; do printf 'mb_%sto%s ' "${STARTS[$i]}" "${ENDS[$i]}"; done)"
read -rp "Proceed? [y/N]: " CONFIRM
[[ "$CONFIRM" =~ ^[Yy]([Ee][Ss])?$ ]] || { echo "Aborted; nothing submitted."; exit 1; }

for i in "${!STARTS[@]}"; do
    s=${STARTS[$i]}; e=${ENDS[$i]}
    NAME="mb_${s}to${e}"
    RESULTS="data/bioreactor_simulation/${NAME}.json"
    # SLURM resolves --output relative to the submission dir, so anchor the log beside the .json
    LOG="$HOME/CNODE/data/bioreactor_simulation/${NAME}.log"
    # Positionals in the exact order run_scaling_sweep.slurm's stage 2 reads them ($1..$19)
    sbatch \
        --job-name="$NAME" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --output="$LOG" \
        "$SWEEP_SCRIPT" "$RESULTS" "$s" "$e" "$REPEATS" "$AGGREGATOR" "$WARMSTART" \
             "$RUN_ADMM" "$RUN_MONO" "$RUN_SEQ" "$BUDGET" "$MAX_ADMM_ITERS" "$LATE_MU_INIT" \
             "$SUBP_RULE" "$MAX_TRAJS_PER_SUBP" "$MIN_NUM_SUBP" "$K" \
             "$ITER_RULE" "$ITER_CONST" "$ITER_FLOOR"
done
echo "Submitted $NJOBS jobs."
