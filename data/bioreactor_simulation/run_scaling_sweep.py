"""
Script to run a training-set-size (B) scaling sweep and extract per-B training metrics.

Over mutliple runs (diff. seeds), sweeps the number of training trajectories B (with a fixed test set) 
and, for each B, trains three Neural-ODE models on the same split:
-- monolithic collocation model (Pyomo/IPOPT, block-diagonal DC)   -> 'mono'
-- ADMM consensus collocation model (Pyomo/IPOPT, DC or IRR-DC)    -> 'admm'
-- sequential Neural ODE baseline (PyTorch / torchdiffeq)          -> 'seq'

The ADMM transcription is chosen with --transcription_method (dc | irrdc). IRR-DC adds the
integrated-residual penalty (1/(2 irrdc_param)) * sum_m w_m ||r(t_m)||^2 over an equidistant
residual grid of --num_res_eval_nodes points, and needs one trajectory per ADMM subproblem.

Produces .json file containing, for each B in sweep:
-- wall-clock training time (s) 
-- train and test MSEs
-- final train and test MSEs
-- primal and dual residuals at each iteration (for ADMM)
-- trained NN params per model (for forward prediction)
"""
import json
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

from data.bioreactor_simulation.run_training import run_training

STATE_VARS = ['G', 'O', 'X', 'Xd', 'P', 'L', 'CO2']
NUM_TEST = 75
SPLIT_SEED = 0 # fixes the test set (same for every B and repeat)

# Following funcs own the train/test split and the per-repeat seed derivation. run_training is a pure trainer
# and receives the resulting arrays + model_init_seed; it knows nothing about sweeps or repeats.

def compute_test_batch_ids(all_batch_ids, split_seed, num_test):
    """Held-out test ids, drawn ONCE from split_seed => identical for every B and every repeat."""
    test_perm = np.random.default_rng(split_seed).permutation(all_batch_ids)
    return sorted(int(x) for x in test_perm[:num_test])

def compute_train_batch_ids(all_batch_ids, test_batch_ids, split_seed, repeat_index, num_train):
    """Training ids: permute the non-test pool with a per-repeat seed (independent of B) and take the
    first num_train. => nested across B within a repeat; a different subset ordering across repeats.
    Deliberately NOT sorted: the random permutation order is kept so that the ADMM's contiguous
    subproblem grouping is random per repeat and (nearly) nested across B within a repeat."""
    pool = [b for b in all_batch_ids if b not in set(test_batch_ids)]
    train_perm = np.random.default_rng((split_seed, repeat_index)).permutation(pool)
    return [int(x) for x in train_perm[:num_train]]

def compute_model_init_seed(split_seed, repeat_index):
    """One well-mixed integer per (split_seed, repeat_index), independent of B. Shared by mono/admm/seq
    so all three start from the same initialisation within a repeat."""
    return int(np.random.SeedSequence((split_seed, repeat_index)).generate_state(1)[0])

def compute_num_subproblems(num_train, max_trajs_per_subp, min_num_subp):
    """Subproblem-size rule: at least `min_num_subp` subproblems, and none holding more than
    `max_trajs_per_subp` trajectories -> max(min_num_subp, ceil(B / max_trajs_per_subp)), capped at T.
    The T trajectories are then split evenly (group sizes differ by at most one)."""
    return min(num_train, max(min_num_subp, int(np.ceil(num_train / max_trajs_per_subp))))

def compute_admm_iter_budget(num_train, num_subproblems, iter_const, iter_floor,
                             s1_iter_per_traj=20, s1_iter_floor=600):
    """ADMM iteration budget, with S = ceil(B / num_subproblems) the largest subproblem's trajectory count.

    S >= 2 (multi-block): N = max(iter_floor, ceil(iter_const / S)). Larger subproblems reach the
        train-MSE plateau in fewer (but costlier, ~S seconds each) iterations, so N shrinks as S grows.
        Calibrated on the mb_1 sweep (const 1500, floor 80): covers ~92% of train-MSE plateaus overall,
        ~96% for B >= 71 (const 2000: ~96% / ~98%).
    S == 1 (one trajectory per subproblem, B subproblems): N = max(s1_iter_floor, ceil(s1_iter_per_traj * B)).
        Here the plateau iteration GROWS with B (~4 * B^1.26: medians ~165 / 415 / 1320 at B = 20 / 40 / 80),
        as B single-trajectory fits must be averaged into consensus. Calibrated on the S=1 runs in
        ISAMBARD/ (isambard_9/10 at B=20, run3/4 at B=40-60, run5/6 at B=80; mostly older code):
        (20, 600) covers ~97% of train-MSE plateaus."""
    S = int(np.ceil(num_train / num_subproblems))
    if S == 1:
        return max(s1_iter_floor, int(np.ceil(s1_iter_per_traj * num_train)))
    return max(iter_floor, int(np.ceil(iter_const / S)))

def build_trajectory_array(df, batch_ids, state_vars):
    """Slice the dataframe to the given batch ids and stack into (len(batch_ids), N, D) + time grid."""
    batches = {bid: df[df['Batch'] == f"Batch_{bid}"].reset_index(drop=True) for bid in batch_ids}
    grid = batches[batch_ids[0]]['Time'].values
    Y = np.array([batches[bid][state_vars].values for bid in batch_ids])  # (len(batch_ids), N, D)
    return grid, Y

def extract_trained_params(method: str, model) -> dict:
    """Pull a model's trained NN weights/biases as JSON-serialisable nested lists.

    The live model objects (Pyomo/IPOPT state) cannot be serialised, but the fitted
    NN parameters are small ndarrays. We want those so the plot scripts can
    forward-integrate trajectories without re-running training.
      - 'admm' -> consensus_Ws / consensus_bs (the converged consensus model)
      - 'mono' -> convert_weights() / convert_biases() (read from the solved model)
    """
    if method == 'admm':
        Ws, bs = model.consensus_Ws, model.consensus_bs
    elif method == 'mono':
        Ws, bs = model.convert_weights(), model.convert_biases()
    else:
        raise ValueError(f"\nUnknown model: {method}")
    return {
        "Ws": [np.asarray(W, dtype=float).tolist() for W in Ws],
        "bs": [np.asarray(b, dtype=float).tolist() for b in bs],
    }

def print_admm_hyperparams(admm_config, admm_training_options):
    """Print a table of EVERY ADMM hyperparameter (once, at the top of the run log).

    Passed to run_training as `hyperparam_table`; run_training invokes it with the config it built.
    Generic on purpose: it dumps every key in admm_config and admm_training_options (minus the bulky
    data arrays) so it can never silently drift out of sync when a hyperparameter is renamed/added.
    """
    # Bulky data arrays (not hyperparameters); mse_eval_options holds arrays too and is handled below.
    SKIP = {"Y_obs", "obs_times", "mse_eval_options"}
    rows = []
    # (1) training-loop options
    for k, v in admm_training_options.items():
        if k not in SKIP:
            rows.append((k, v))
    # (1b) MSE-eval tolerances (pull the scalars out of mse_eval_options; skip its arrays)
    mse_opts = admm_training_options.get("mse_eval_options", {})
    for k in ("rtol", "atol", "max_step"):
        if k in mse_opts:
            rows.append((f"mse_eval_{k}", mse_opts[k]))
    # (2) model / algorithm config (everything the ADMM constructor received)
    for k, v in admm_config.items():
        if k not in SKIP:
            rows.append((k, v))
    width = max(len(k) for k, _ in rows)
    bar = "=" * 64
    print(f"\n{bar}\nADMM HYPERPARAMETERS\n{bar}")
    for k, v in rows:
        print(f"  {k:<{width}} : {v}")
    print(f"{bar}\n")

def main():
    batch_sizes = list(range(args.B_start, args.B_end + 1))
    # One hidden layer of width args.hidden_layer_size; input/output pinned to the state dimension.
    nn_layer_sizes = [len(STATE_VARS), args.hidden_layer_size, len(STATE_VARS)]
    # ADMM wall-clock cap: <=0 means "no cap" (run to max_admm_iterations). Only affects the ADMM baseline.
    training_clock_budget = args.training_clock_budget if args.training_clock_budget > 0 else None
    print(f"Scaling sweep: B in {batch_sizes}, {args.num_training_repeats} repeat(s) per B, "
          f"test set size = {NUM_TEST}, nn_layer_sizes = {nn_layer_sizes}")
    print(f"results -> {args.results_path}")
    out = {
        "config": {
            "train_data_path": args.train_data_path,
            "test_data_path": args.test_data_path,
            "state vars": STATE_VARS,
            "nn_layer_sizes": nn_layer_sizes,
            "num_test_trajectories": NUM_TEST,
            "batch_sizes": batch_sizes,
            "split_seed": SPLIT_SEED,
            "num_training_repeats": args.num_training_repeats,
            "num_batches_per_subp": args.num_batches_per_subp,
            # subproblem-size rule (overrides num_batches_per_subp when set); None -> fixed K
            "subp_size_rule": ({"max_trajs_per_subp": args.max_trajs_per_subp, "min_num_subp": args.min_num_subp}
                               if args.subp_size_rule else None),
            # ADMM iteration-budget rule (overrides max_admm_iterations when set); None -> fixed budget
            "admm_iter_rule": ({"iter_const": args.admm_iter_const, "iter_floor": args.admm_iter_floor,
                                "s1_iter_per_traj": args.admm_s1_iter_per_traj, "s1_iter_floor": args.admm_s1_iter_floor}
                               if args.admm_iter_rule else None),
            "max_admm_iterations": None if args.admm_iter_rule else args.max_admm_iterations,
            # ADMM transcription: 'dc' | 'irrdc'; the IRR-DC hyperparams are None for 'dc'
            "transcription_method": args.transcription_method,
            "irrdc_param": args.irrdc_param,
            "num_res_eval_nodes": args.num_res_eval_nodes
        },
        # results[str(B)][str(repeat)] = { ...per-run metrics... }
        "results": {},
    }

    def save():
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_path, "w") as f:
            json.dump(out, f, indent=2)

    # load the train/test CSVs & build the split. With a single file (test path == train path) this is
    # exactly the original single-CSV split: the test ids are excluded from the training pool. With a
    # separate test file the two sets are independent, so the WHOLE training file is the training pool.
    def batch_ids_of(frame):
        return sorted(int(b.split('_')[1]) for b in frame['Batch'].unique())
    single_file = args.test_data_path == args.train_data_path
    train_df = pd.read_csv(args.train_data_path)
    test_df = train_df if single_file else pd.read_csv(args.test_data_path)
    train_file_ids, test_file_ids = batch_ids_of(train_df), batch_ids_of(test_df)
    if NUM_TEST > len(test_file_ids):
        raise ValueError(f"num_test = {NUM_TEST} exceeds the {len(test_file_ids)} trajectories in {args.test_data_path}.")
    # Test set is fixed across every B and repeat
    test_batch_ids = compute_test_batch_ids(test_file_ids, SPLIT_SEED, NUM_TEST)
    test_grid, Y_test_obs = build_trajectory_array(test_df, test_batch_ids, STATE_VARS)
    # ids to hold out of the training pool: only needed when train and test share one file
    train_exclude_ids = test_batch_ids if single_file else []
    train_pool_size = len(set(train_file_ids) - set(train_exclude_ids))
    max_B = batch_sizes[-1]
    if max_B > train_pool_size:
        raise ValueError(
            f"largest B = {max_B} exceeds the {train_pool_size} training trajectories available in "
            f"{args.train_data_path}" + (f" after excluding the {NUM_TEST} test batch ids." if single_file else ".")
        )

    for B in batch_sizes:
        out["results"][str(B)] = {}
        for r in range(args.num_training_repeats):
            print(f"\n{'#' * 70}\n# B = {B} training batches | repeat {r + 1}/{args.num_training_repeats}\n{'#' * 70}")
            try:
                # Build this run's training set + model-init seed 
                train_batch_ids = compute_train_batch_ids(train_file_ids, train_exclude_ids, SPLIT_SEED, r, B)
                train_grid, Y_obs = build_trajectory_array(train_df, train_batch_ids, STATE_VARS)
                model_init_seed = compute_model_init_seed(SPLIT_SEED, r)
                if args.subp_size_rule:
                    num_subproblems = compute_num_subproblems(B, args.max_trajs_per_subp, args.min_num_subp)
                    print(f"Subproblem-size rule: {num_subproblems} subproblems of "
                          f"{B // num_subproblems}-{-(-B // num_subproblems)} trajectories")
                else:
                    num_subproblems = None # ADMM uses ceil(B / num_batches_per_subp)
                if args.admm_iter_rule:
                    n_subp = num_subproblems if num_subproblems is not None else \
                        min(B, max(1, int(np.ceil(B / args.num_batches_per_subp))))
                    max_admm_iterations = compute_admm_iter_budget(
                        B, n_subp, args.admm_iter_const, args.admm_iter_floor,
                        args.admm_s1_iter_per_traj, args.admm_s1_iter_floor)
                    S = int(np.ceil(B / n_subp))
                    if S == 1:
                        print(f"ADMM iteration-budget rule (S=1): max(ADMM iters) = max({args.admm_s1_iter_floor}, "
                              f"ceil({args.admm_s1_iter_per_traj:g} * {B})) = {max_admm_iterations}")
                    else:
                        print(f"ADMM iteration-budget rule: max(ADMM iters) = max({args.admm_iter_floor}, "
                              f"ceil({args.admm_iter_const:g} / {S})) = {max_admm_iterations}")
                else:
                    max_admm_iterations = args.max_admm_iterations
                res = run_training(
                    Y_obs=Y_obs,
                    train_grid=train_grid,
                    Y_test_obs=Y_test_obs,
                    test_grid=test_grid,
                    nn_layer_sizes=nn_layer_sizes,
                    model_init_seed=model_init_seed,
                    admm_warm_init=args.admm_warm_init,
                    run_admm_baseline=args.run_admm_baseline,
                    consensus_aggregator=args.consensus_aggregator,
                    subproblem_warmstart=args.subproblem_warmstart,
                    relaxation_alpha=args.relaxation_alpha,
                    adaptive_admm_penalty=args.adaptive_penalty,
                    rho_max=args.rho_max,
                    late_mu_init=args.late_mu_init,
                    num_batches_per_subp=args.num_batches_per_subp,
                    num_subproblems=num_subproblems,
                    l2_reg_param=args.l2_reg_param,
                    transcription_method=args.transcription_method,
                    irrdc_param=args.irrdc_param,
                    num_res_eval_nodes=args.num_res_eval_nodes,
                    training_clock_budget=training_clock_budget,
                    max_admm_iterations=max_admm_iterations,
                    hyperparam_table=(print_admm_hyperparams if (B == batch_sizes[0] and r == 0) else None),
                    run_sequential_baseline=args.run_sequential_baseline,
                    run_monolithic_baseline=args.run_monolithic_baseline,
                    return_models=True
                )
                train_runtimes = {m: float(t) for m, t in res["train_runtimes"].items()}
                final_mses = {
                    m: {k: (float(v) if v is not None else None) for k, v in d.items()}
                    for m, d in res.get("final_mses", {}).items()
                } # final_mses[m] = {train, test, test_std, test_min, test_max}

                # The *_history objects are already built as dicts-of-lists => JSON-serialisable
                mse_histories = res.get("mse_histories", {})
                residual_histories = res.get("residual_histories", {})
                objective_histories = res.get("objective_histories", {})

                # Retain the fitted NN params per model so plot_predicted_trajectories.py can
                # forward-integrate test trajectories from the JSON without retraining
                trained_params = {
                    m: extract_trained_params(m, model)
                    for m, model in res.get("trained_models", {}).items()
                }

                # ADMM subproblem layout actually used for this B, read from the trained model (None if no ADMM run)
                admm_model = res.get("trained_models", {}).get("admm")
                subp_group_sizes = [len(g) for g in admm_model.subp_groups] if admm_model is not None else None

                out["results"][str(B)][str(r)] = {
                    "B": B,
                    "repeat_index": r,
                    "num_subproblems": len(subp_group_sizes) if subp_group_sizes is not None else None,
                    "max_admm_iterations": max_admm_iterations if admm_model is not None else None, # budget used for this B
                    "subp_group_sizes": subp_group_sizes, # trajectories per ADMM subproblem, in worker order
                    "train_batch_ids": train_batch_ids, # owned by the sweep (the split lives here)
                    "test_batch_ids": test_batch_ids,
                    "train_runtimes": train_runtimes,
                    "final_mses": final_mses,
                    "mse_histories": mse_histories,
                    "residual_histories": residual_histories,
                    "objective_histories": objective_histories,
                    "trained_params": trained_params
                }
            except Exception as e:  # keep the sweep alive; record the failure for this (B, repeat)
                out["results"][str(B)][str(r)] = {"B": B, "repeat_index": r, "error": f"{type(e).__name__}: {e}"}
                print(f"  B={B} repeat={r} FAILED: {type(e).__name__}: {e}")
            save()
            print(f" [saved results through B={B}, repeat={r}]")

    print(f"\nSweep complete. Results in {args.results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data_path", "--data_path", dest="train_data_path", type=str, required=True,
                        help="CSV of trajectories to draw the TRAINING sets from (e.g. noisy). "
                             "--data_path is kept as an alias for backwards compatibility.")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="CSV of trajectories to draw the fixed TEST set from (e.g. noise-free). "
                             "Defaults to --train_data_path, i.e. the original single-file train/test split. "
                             "With a separate test file the whole training file is used as the training pool.")
    parser.add_argument("--results_path", type=Path, required=True,
                        help="Path to write the results .json (parent dirs created as needed).")
    parser.add_argument("--B_start", type=int, metavar="B_START", required=True,
                        help="Start of training batch sizes to sweep over")
    parser.add_argument("--B_end", type=int, metavar="B_END", required=True,
                        help="End of training batch sizes to sweep over")
    parser.add_argument("--hidden_layer_size", type=int, default=16,
                        help="Width of the single hidden layer in the NN vector field (default 16). "
                             "The network is [state_dim, hidden_layer_size, state_dim].")
    parser.add_argument("--num_training_repeats", type=int, default=1,
                        help="Repeats per B (default 1). Each repeat uses a different training-subset "
                             "permutation and a different model initialisation, with a fixed test set, "
                             "so the spread across repeats gives a CI at each B.")
    parser.add_argument("--num_batches_per_subp", type=int, default=1,
                        help="Multi-block ADMM: number of trajectories (K) stitched block-diagonally "
                                "into each ADMM subproblem. K=1 (default) is classic consensus ADMM (one "
                                "subproblem per trajectory); K>=B is a single monolithic block. "
                                "num_subproblems = ceil(B / K), split evenly via np.array_split. "
                                "Ignored when --subp_size_rule is on.")
    parser.add_argument("--subp_size_rule", action=argparse.BooleanOptionalAction, default=False,
                        help="ADMM-only: set the subproblem count per B by the subproblem-size rule "
                             "num_subproblems = max(min_num_subp, ceil(B / max_trajs_per_subp)), split evenly, "
                             "instead of a fixed --num_batches_per_subp (default off).")
    parser.add_argument("--max_trajs_per_subp", type=int, default=20,
                        help="Subproblem-size rule: max trajectories per subproblem (default 20).")
    parser.add_argument("--min_num_subp", type=int, default=4,
                        help="Subproblem-size rule: min number of subproblems (default 4).")
    parser.add_argument("--rho_max", type=float, default=50,
                        help="ADMM-only: upper clamp on the adaptive penalty ρ (default 50). Lower clamp is 0.1.")
    parser.add_argument("--late_mu_init", type=float, default=0.1,
                        help="ADMM-only: IPOPT mu_init for the late subproblem solves (default 0.1 = IPOPT's "
                             "default). Early solves and the monolith are unaffected.")
    parser.add_argument("--l2_reg_param", type=float, default=1e-3,
                        help="L2 regularisation weight ν on the NN parameters (default 1e-3). "
                             "Applied to BOTH the ADMM and monolithic objectives.")
    parser.add_argument("--transcription_method", type=str, choices=("dc", "irrdc"), default="dc",
                        help="ADMM-only: collocation transcription. 'dc' (default) = direct collocation; "
                             "'irrdc' = DC plus the integrated-residual penalty "
                             "(1/(2 irrdc_param)) * sum_m w_m ||dY/dt(t_m) - NN(Y(t_m))||^2 over an "
                             "equidistant residual grid. Requires --irrdc_param and --num_res_eval_nodes, "
                             "and one trajectory per subproblem (--num_batches_per_subp 1, no "
                             "--subp_size_rule). The monolithic baseline is always plain DC.")
    parser.add_argument("--irrdc_param", type=float, default=None,
                        help="IRR-DC: residual penalty weight μ; the penalty is scaled by 1/(2μ), so a "
                             "LARGER μ gives a WEAKER residual term. Required with --transcription_method irrdc.")
    parser.add_argument("--num_res_eval_nodes", type=int, default=None,
                        help="IRR-DC: number of equidistant residual-evaluation nodes on [0, T]. "
                             "Required with --transcription_method irrdc.")
    parser.add_argument("--training_clock_budget", type=float, default=60,
                        help="ADMM-only: wall-clock cap PER (B, repeat) run, in minutes (default 60). "
                             "Pass 0 (or any value <=0) for no cap, i.e. run to --max_admm_iterations. "
                             "Ignored when the ADMM baseline is not trained.")
    parser.add_argument("--max_admm_iterations", type=int, default=2000,
                        help="ADMM-only: hard cap on the number of consensus iterations per run "
                             "(default 2000). Ignored when the ADMM baseline is not trained, or when "
                             "--admm_iter_rule is on.")
    parser.add_argument("--admm_iter_rule", action=argparse.BooleanOptionalAction, default=False,
                        help="ADMM-only: set the iteration cap per B as max(admm_iter_floor, "
                             "ceil(admm_iter_const / S)), S = largest subproblem size, instead of a fixed "
                             "--max_admm_iterations (default off). The wall-clock budget still applies.")
    parser.add_argument("--admm_iter_const", type=float, default=1500,
                        help="Iteration-budget rule: constant in ceil(const / S) (default 1500).")
    parser.add_argument("--admm_iter_floor", type=int, default=80,
                        help="Iteration-budget rule: minimum number of ADMM iterations (default 80).")
    parser.add_argument("--admm_s1_iter_per_traj", type=float, default=20,
                        help="Iteration-budget rule, S=1 case (one trajectory per subproblem): iterations per "
                             "training trajectory in max(admm_s1_iter_floor, ceil(admm_s1_iter_per_traj * B)) "
                             "(default 20).")
    parser.add_argument("--admm_s1_iter_floor", type=int, default=600,
                        help="Iteration-budget rule, S=1 case: minimum number of ADMM iterations (default 600).")
    parser.add_argument("--consensus_aggregator", type=str, required=True,
                        help="ADMM consensus aggregator: 'mean' | 'relaxed'")
    parser.add_argument("--subproblem_warmstart", type=str, required=True,
                        help="ADMM subproblem warmstart: 'consensus' | 'relaxed' | 'previous'")
    parser.add_argument("--relaxation_alpha", type=float, default=1.5,
                        help="Boyd relaxation factor α for the 'relaxed' aggregator: "
                             "θ̂ = α·θ + (1-α)·z. α<1 damps spikes, α=1 is plain mean, α>1 over-relaxes "
                             "(accelerates). Only used when --consensus_aggregator=relaxed (default 1.5).")
    parser.add_argument("--run_admm_baseline", action=argparse.BooleanOptionalAction, default=True,
                        help="Train the ADMM consensus model (default on; --no-run_admm_baseline to skip).")
    parser.add_argument("--run_monolithic_baseline", action=argparse.BooleanOptionalAction, default=True,
                        help="Train the monolithic collocation model (default on; --no-run_monolithic_baseline to skip).")
    parser.add_argument("--run_sequential_baseline", action=argparse.BooleanOptionalAction, default=False,
                        help="Train the sequential torchdiffeq baseline (default OFF; --run_sequential_baseline to enable).")
    parser.add_argument("--adaptive_penalty", action=argparse.BooleanOptionalAction, default=True,
                        help="Adaptive penalty via Boyd residual balancing (default on). Pass --no-adaptive-penalty "
                             "for a fixed penalty run (ρ stays at its initial value).")
    parser.add_argument("--admm_warm_init", action="store_true", default=False,
                        help="ADMM: warm-start every subproblem from a batch-0 warmup solve. Omit for "
                             "the default cold per-worker (seeded) Xavier init.")
    args = parser.parse_args()
    if args.test_data_path is None:
        args.test_data_path = args.train_data_path  # single-file split (original behaviour)
    if args.transcription_method == "irrdc":
        if args.irrdc_param is None or args.num_res_eval_nodes is None:
            parser.error("--transcription_method irrdc requires --irrdc_param and --num_res_eval_nodes.")
        if args.run_admm_baseline and (args.subp_size_rule or args.num_batches_per_subp > 1):
            parser.error("--transcription_method irrdc needs one trajectory per ADMM subproblem: "
                         "use --num_batches_per_subp 1 and omit --subp_size_rule.")
    main()
