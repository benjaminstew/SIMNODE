#!/bin/bash

# =============================================================================
# Submit the S=1 (single-trajectory subproblem) ADMM scaling sweep as one job per
# B (= T) value, each a call of run_scaling_sweep.slurm. The S=1 counterpart of
# the multi-block submissions (submit_mb_chunks.slurm / submit_mb_list.slurm).
#
# Run on the login node:  bash data/bioreactor_simulation/submit_s1_list.slurm
# It prompts for the first and last B, then submits one job per B named
# s1_<B>to<B> (e.g. s1_10to10, s1_11to11, ...). Each job writes
# data/bioreactor_simulation/<name>.json and <name>.log.
#
# Same settings as the multi-block runs except:
#   - one trajectory per subproblem (subproblem-size rule off, K=1 -> B subproblems)
#   - the iteration-budget rule therefore takes its S=1 branch,
#     max(600, ceil(20 * B)) (run_scaling_sweep.py's --admm_s1_iter_* defaults)
#   - cores per job = B + 3: one pool worker per subproblem plus headroom for
#     the main process
#   - 06:00:00 SLURM wall clock per job
# Everything else matches the multi-block runs: 32G, 5 repeats, ADMM only,
# relaxed aggregator, consensus warm start, standard ADMM, 60-min ADMM budget/run,
# late mu_init 0.1.
# Settings not prompted by run_scaling_sweep.slurm (alpha, nu, rho_max, hidden
# width, ...) fall back to run_scaling_sweep.py's argparse defaults.
# =============================================================================
set -euo pipefail

# run_scaling_sweep.slurm lives next to this file
SWEEP_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_scaling_sweep.slurm"
[[ -f "$SWEEP_SCRIPT" ]] || { echo "Cannot find $SWEEP_SCRIPT"; exit 1; }

read -rp "  First B (T) value         : " B_FIRST
read -rp "  Last B (T) value          : " B_LAST
for v in "$B_FIRST" "$B_LAST"; do
    [[ "$v" =~ ^[0-9]+$ ]] || { echo "Not a positive integer: '$v'"; exit 1; }
done
(( B_LAST >= B_FIRST )) || { echo "Need last B >= first B."; exit 1; }

# ----- fixed job settings (as the multi-block runs, but S=1) -----
CPU_HEADROOM=3            # cores per job = B + CPU_HEADROOM
MEM=32G
TIME=06:00:00
REPEATS=5
RUN_ADMM=y; RUN_MONO=n; RUN_SEQ=n
AGGREGATOR=relaxed
WARMSTART=consensus
BUDGET=60                 # ADMM wall-clock budget per run (min)
MAX_ADMM_ITERS=2000       # unused while the iteration-budget rule is on
LATE_MU_INIT=0.1
SUBP_RULE=n; MAX_TRAJS_PER_SUBP=20; MIN_NUM_SUBP=4; K=1   # rule off + K=1 -> one trajectory per subproblem
ITER_RULE=y; ITER_CONST=1500; ITER_FLOOR=80               # S=1 branch: max(600, ceil(20*B)); const/floor unused

NJOBS=$(( B_LAST - B_FIRST + 1 ))

echo
echo "About to submit $NJOBS jobs (one per B in $B_FIRST..$B_LAST), each:"
echo "  cpus=B+$CPU_HEADROOM  mem=$MEM  time=$TIME  repeats=$REPEATS  run admm/mono/seq=$RUN_ADMM/$RUN_MONO/$RUN_SEQ"
echo "  aggregator=$AGGREGATOR  warmstart=$WARMSTART  admm budget=${BUDGET} min  late mu_init=$LATE_MU_INIT"
echo "  S=1: subproblem-size rule=$SUBP_RULE, K=$K (B subproblems of 1 trajectory)"
echo "  iteration-budget rule=$ITER_RULE (S=1 branch: max(600, ceil(20*B)))"
echo "  jobs: $(for (( b = B_FIRST; b <= B_LAST; b++ )); do printf 's1_%sto%s(%s cpus) ' "$b" "$b" "$(( b + CPU_HEADROOM ))"; done)"
read -rp "Proceed? [y/N]: " CONFIRM
[[ "$CONFIRM" =~ ^[Yy]([Ee][Ss])?$ ]] || { echo "Aborted; nothing submitted."; exit 1; }

for (( b = B_FIRST; b <= B_LAST; b++ )); do
    NAME="s1_${b}to${b}"
    CPUS=$(( b + CPU_HEADROOM ))
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
        "$SWEEP_SCRIPT" "$RESULTS" "$b" "$b" "$REPEATS" "$AGGREGATOR" "$WARMSTART" \
             "$RUN_ADMM" "$RUN_MONO" "$RUN_SEQ" "$BUDGET" "$MAX_ADMM_ITERS" "$LATE_MU_INIT" \
             "$SUBP_RULE" "$MAX_TRAJS_PER_SUBP" "$MIN_NUM_SUBP" "$K" \
             "$ITER_RULE" "$ITER_CONST" "$ITER_FLOOR"
done
echo "Submitted $NJOBS jobs."
