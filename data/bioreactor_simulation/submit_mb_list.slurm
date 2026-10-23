#!/bin/bash

# =============================================================================
# Submit scaling-sweep jobs for a user-supplied LIST of B (= T) values: for each value,
# one multi-block ADMM job and/or one monolithic job, each a call of run_scaling_sweep.slurm
# (e.g. to fill in missing or incomplete B values). Companion to submit_mb_chunks.slurm,
# with identical ADMM run settings.
#
# Run on the login node:  bash data/bioreactor_simulation/submit_mb_list.slurm
# Enter the B values separated by spaces or commas (e.g. "12 23 33" or "12,23,33"), choose
# which models to run, then give CPU-core and RAM lists for each model: the i-th entry goes
# with the i-th B value. A single value is applied to every job; leaving a list blank uses
# its default. RAM may be given as "32" or "32G".
#
# Jobs and files (data/bioreactor_simulation/<name>.json / .log):
#   mb_<B>to<B>    multi-block ADMM only  (merges with submit_mb_chunks.slurm's mb_* files)
#   mono_<B>to<B>  monolithic only        (merges with the mono_* sweep files)
# Both use the same seeds per repeat (split seed + repeat index), so ADMM and monolithic
# results for the same (B, repeat) share the training set and the weight initialisation.
#
# Shared settings: 5 repeats, 5:30:00 wall clock per job (edit ADMM_TIME / MONO_TIME below).
# ADMM settings (same as submit_mb_chunks.slurm): relaxed aggregator, consensus warm start,
#   standard ADMM, 60-min ADMM budget/run, iteration-budget rule
#   max(80, ceil(1500/S)), late mu_init 0.1, subproblem-size rule (max 20 trajs/subp,
#   min 4 subps). The monolithic job ignores them.
# Settings not prompted by run_scaling_sweep.slurm (alpha, nu, rho_max, hidden width, ...)
# fall back to run_scaling_sweep.py's argparse defaults.
# =============================================================================
set -euo pipefail

# run_scaling_sweep.slurm lives next to this file
SWEEP_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_scaling_sweep.slurm"
[[ -f "$SWEEP_SCRIPT" ]] || { echo "Cannot find $SWEEP_SCRIPT"; exit 1; }

# expand_list NAME DEFAULT INPUT: blank -> DEFAULT for every B; one value -> applied to every B;
# otherwise must have one entry per B value (matched by position). Result goes in array NAME.
expand_list() {
    local name=$1 default=$2 input=${3//,/ }
    local -a vals=($input)
    (( ${#vals[@]} == 0 )) && vals=("$default")
    if (( ${#vals[@]} == 1 )); then
        local v=${vals[0]}; vals=(); for _ in "${B_VALUES[@]}"; do vals+=("$v"); done
    fi
    (( ${#vals[@]} == NB )) || { echo "Got ${#vals[@]} values for $NB B values ($name); give one per B value (or a single value)."; exit 1; }
    eval "$name=(\"\${vals[@]}\")"
}
check_cpus() { for c in "$@"; do [[ "$c" =~ ^[0-9]+$ ]] && (( c >= 1 )) || { echo "Not a positive integer core count: '$c'"; exit 1; }; done; }
check_mem()  { for m in "$@"; do [[ "$m" =~ ^[0-9]+[Gg]?$ ]] || { echo "RAM must look like 32 or 32G: '$m'"; exit 1; }; done; }

read -rp "  B (T) values, space/comma separated         : " B_LIST
B_VALUES=(${B_LIST//,/ })
NB=${#B_VALUES[@]}
(( NB > 0 )) || { echo "No B values given."; exit 1; }
for b in "${B_VALUES[@]}"; do
    [[ "$b" =~ ^[0-9]+$ ]] && (( b >= 1 )) || { echo "Not a positive integer B value: '$b'"; exit 1; }
done
# B values must be unique (the core / RAM lists are matched to them by position)
DUPES=$(printf '%s\n' "${B_VALUES[@]}" | sort -n | uniq -d | tr '\n' ' ')
[[ -z "$DUPES" ]] || { echo "Duplicate B values: $DUPES"; exit 1; }

read -rp "  Run multi-block ADMM jobs   (y/n) [y]       : " DO_ADMM
DO_ADMM="${DO_ADMM:-y}"
read -rp "  Run monolithic jobs         (y/n) [y]       : " DO_MONO
DO_MONO="${DO_MONO:-y}"
[[ "$DO_ADMM" =~ ^[Yy] || "$DO_MONO" =~ ^[Yy] ]] || { echo "Nothing to run."; exit 1; }

if [[ "$DO_ADMM" =~ ^[Yy] ]]; then
    read -rp "  ADMM CPU cores per B value (same order) [10] : " IN
    expand_list ADMM_CPUS 10 "$IN"; check_cpus "${ADMM_CPUS[@]}"
    read -rp "  ADMM RAM per B value (same order)       [32G]: " IN
    expand_list ADMM_MEM 32G "$IN"; check_mem "${ADMM_MEM[@]}"
fi
if [[ "$DO_MONO" =~ ^[Yy] ]]; then
    # IPOPT runs single-threaded here (OMP_NUM_THREADS=1 in run_scaling_sweep.slurm), so 2 cores is plenty
    read -rp "  Mono CPU cores per B value (same order) [2]  : " IN
    expand_list MONO_CPUS 2 "$IN"; check_cpus "${MONO_CPUS[@]}"
    read -rp "  Mono RAM per B value (same order)       [32G]: " IN
    expand_list MONO_MEM 32G "$IN"; check_mem "${MONO_MEM[@]}"
fi

# ----- fixed job settings -----
ADMM_TIME=05:30:00        # 5 repeats x 60-min ADMM budget + 30 min margin
MONO_TIME=05:30:00        # 5 monolithic solves (median ~10-18 min at B~100-160, rare ~1 h)
REPEATS=5
RUN_SEQ=n
AGGREGATOR=relaxed
WARMSTART=consensus
BUDGET=60                 # ADMM wall-clock budget per run (min)
MAX_ADMM_ITERS=2000       # unused while the iteration-budget rule is on
LATE_MU_INIT=0.1
SUBP_RULE=y; MAX_TRAJS_PER_SUBP=20; MIN_NUM_SUBP=4; K=1
ITER_RULE=y; ITER_CONST=1500; ITER_FLOOR=80

NJOBS=0
[[ "$DO_ADMM" =~ ^[Yy] ]] && NJOBS=$(( NJOBS + NB ))
[[ "$DO_MONO" =~ ^[Yy] ]] && NJOBS=$(( NJOBS + NB ))

echo
echo "About to submit $NJOBS jobs ($NB B values; ADMM=$DO_ADMM, monolithic=$DO_MONO), each with:"
echo "  repeats=$REPEATS  time: ADMM $ADMM_TIME / mono $MONO_TIME"
if [[ "$DO_ADMM" =~ ^[Yy] ]]; then
    echo "  ADMM: aggregator=$AGGREGATOR  warmstart=$WARMSTART  budget=${BUDGET} min  late mu_init=$LATE_MU_INIT"
    echo "        iteration-budget rule=$ITER_RULE (max($ITER_FLOOR, ceil($ITER_CONST/S)))"
    echo "        subproblem-size rule=$SUBP_RULE (max trajs/subp=$MAX_TRAJS_PER_SUBP, min subps=$MIN_NUM_SUBP)"
fi
echo "  jobs:"
for i in "${!B_VALUES[@]}"; do
    b=${B_VALUES[$i]}
    [[ "$DO_ADMM" =~ ^[Yy] ]] && printf '    mb_%sto%s     cpus=%s  mem=%sG\n' "$b" "$b" "${ADMM_CPUS[$i]}" "${ADMM_MEM[$i]%[Gg]}"
    [[ "$DO_MONO" =~ ^[Yy] ]] && printf '    mono_%sto%s   cpus=%s  mem=%sG\n' "$b" "$b" "${MONO_CPUS[$i]}" "${MONO_MEM[$i]%[Gg]}"
done
read -rp "Proceed? [y/N]: " CONFIRM
[[ "$CONFIRM" =~ ^[Yy]([Ee][Ss])?$ ]] || { echo "Aborted; nothing submitted."; exit 1; }

# submit_job NAME CPUS MEM TIME RUN_ADMM RUN_MONO B
submit_job() {
    local name=$1 cpus=$2 mem="${3%[Gg]}G" time=$4 run_admm=$5 run_mono=$6 b=$7
    local results="data/bioreactor_simulation/${name}.json"
    # SLURM resolves --output relative to the submission dir, so anchor the log beside the .json
    local log="$HOME/CNODE/data/bioreactor_simulation/${name}.log"
    # Positionals in the exact order run_scaling_sweep.slurm's stage 2 reads them ($1..$19)
    sbatch \
        --job-name="$name" \
        --cpus-per-task="$cpus" \
        --mem="$mem" \
        --time="$time" \
        --output="$log" \
        "$SWEEP_SCRIPT" "$results" "$b" "$b" "$REPEATS" "$AGGREGATOR" "$WARMSTART" \
             "$run_admm" "$run_mono" "$RUN_SEQ" "$BUDGET" "$MAX_ADMM_ITERS" "$LATE_MU_INIT" \
             "$SUBP_RULE" "$MAX_TRAJS_PER_SUBP" "$MIN_NUM_SUBP" "$K" \
             "$ITER_RULE" "$ITER_CONST" "$ITER_FLOOR"
}

for i in "${!B_VALUES[@]}"; do
    b=${B_VALUES[$i]}
    [[ "$DO_ADMM" =~ ^[Yy] ]] && submit_job "mb_${b}to${b}"   "${ADMM_CPUS[$i]}" "${ADMM_MEM[$i]}" "$ADMM_TIME" y n "$b"
    [[ "$DO_MONO" =~ ^[Yy] ]] && submit_job "mono_${b}to${b}" "${MONO_CPUS[$i]}" "${MONO_MEM[$i]}" "$MONO_TIME" n y "$b"
done
echo "Submitted $NJOBS jobs."
