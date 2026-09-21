"""
Script to run a training-set-size (B) scaling sweep and extract per-B training metrics.

Over mutliple runs (diff. seeds), sweeps the number of training trajectories B (with a fixed test set) 
and, for each B, trains three Neural-ODE models on the same split:
-- monolithic collocation model (Pyomo/IPOPT, block-diagonal DC)   -> 'mono'
-- ADMM consensus collocation model (Pyomo/IPOPT, DC)              -> 'admm'
-- sequential Neural ODE baseline (PyTorch / torchdiffeq)          -> 'seq'

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
NN_LAYER_SIZES = [len(STATE_VARS), 16, len(STATE_VARS)]
NUM_TEST = 50
SPLIT_SEED = 0 # fixes the test set (same for every B and repeat)

# Following funcs own the train/test split and the per-repeat seed derivation. run_training is a pure trainer
# and receives the resulting arrays + model_init_seed; it knows nothing about sweeps or repeats.

def compute_test_batch_ids(all_batch_ids, split_seed, num_test):
    """Held-out test ids, drawn ONCE from split_seed => identical for every B and every repeat."""
    test_perm = np.random.default_rng(split_seed).permutation(all_batch_ids)
    return sorted(int(x) for x in test_perm[:num_test])

def compute_train_batch_ids(all_batch_ids, test_batch_ids, split_seed, repeat_index, num_train):
    """Training ids: permute the non-test pool with a per-repeat seed (independent of B) and take the
    first num_train. => nested across B within a repeat; a different subset ordering across repeats."""
    pool = [b for b in all_batch_ids if b not in set(test_batch_ids)]
    train_perm = np.random.default_rng((split_seed, repeat_index)).permutation(pool)
    return sorted(int(x) for x in train_perm[:num_train])

def compute_model_init_seed(split_seed, repeat_index):
    """One well-mixed integer per (split_seed, repeat_index), independent of B. Shared by mono/admm/seq
    so all three start from the same initialisation within a repeat."""
    return int(np.random.SeedSequence((split_seed, repeat_index)).generate_state(1)[0])

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

def print_admm_hyperparams(admm_config, admm_training_options, fast_admm, fast_admm_restart_eta, fast_admm_start_iter):
    """Print a table of EVERY ADMM / Fast ADMM hyperparameter (once, at the top of the run log).

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
    # (3) Fast ADMM extras (only meaningful when enabled)
    rows.append(("fast_admm", fast_admm))
    if fast_admm:
        rows.append(("fast_admm_restart_eta", fast_admm_restart_eta))
        rows.append(("fast_admm_start_iter", fast_admm_start_iter))

    width = max(len(k) for k, _ in rows)
    bar = "=" * 64
    print(f"\n{bar}\nADMM HYPERPARAMETERS ({'Fast ADMM' if fast_admm else 'standard ADMM'})\n{bar}")
    for k, v in rows:
        print(f"  {k:<{width}} : {v}")
    print(f"{bar}\n")

def main():
    batch_sizes = list(range(args.B_start, args.B_end + 1))
    print(f"Scaling sweep: B in {batch_sizes}, {args.num_training_repeats} repeat(s) per B, test set size = {NUM_TEST}")
    print(f"results -> {args.results_path}")
    out = {
        "config": {
            "data_path": args.data_path,
            "state vars": STATE_VARS,
            "nn_layer_sizes": NN_LAYER_SIZES,
            "num_test_trajectories": NUM_TEST,
            "batch_sizes": batch_sizes,
            "split_seed": SPLIT_SEED,
            "num_training_repeats": args.num_training_repeats,
            "num_batches_per_subp": args.num_batches_per_subp
        },
        # results[str(B)][str(repeat)] = { ...per-run metrics... }
        "results": {},
    }

    def save():
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_path, "w") as f:
            json.dump(out, f, indent=2)

    # load the CSV & build the split
    df = pd.read_csv(args.data_path)
    all_batch_ids = sorted(int(b.split('_')[1]) for b in df['Batch'].unique())
    max_B = batch_sizes[-1]
    if max_B + NUM_TEST > len(all_batch_ids):
        raise ValueError(
            f"largest B + num_test = {max_B + NUM_TEST} exceeds the number of available "
            f"trajectories ({len(all_batch_ids)}) in {args.data_path}."
        )
    # Test set is fixed across every B and repeat 
    test_batch_ids = compute_test_batch_ids(all_batch_ids, SPLIT_SEED, NUM_TEST)
    test_grid, Y_test_obs = build_trajectory_array(df, test_batch_ids, STATE_VARS)

    for B in batch_sizes:
        out["results"][str(B)] = {}
        for r in range(args.num_training_repeats):
            print(f"\n{'#' * 70}\n# B = {B} training batches | repeat {r + 1}/{args.num_training_repeats}\n{'#' * 70}")
            try:
                # Build this run's training set + model-init seed 
                train_batch_ids = compute_train_batch_ids(all_batch_ids, test_batch_ids, SPLIT_SEED, r, B)
                train_grid, Y_obs = build_trajectory_array(df, train_batch_ids, STATE_VARS)
                model_init_seed = compute_model_init_seed(SPLIT_SEED, r)
                res = run_training(
                    Y_obs=Y_obs,
                    train_grid=train_grid,
                    Y_test_obs=Y_test_obs,
                    test_grid=test_grid,
                    nn_layer_sizes=NN_LAYER_SIZES,
                    model_init_seed=model_init_seed,
                    admm_warm_init=args.admm_warm_init,
                    run_admm_baseline=True,
                    consensus_aggregator=args.consensus_aggregator,
                    subproblem_warmstart=args.subproblem_warmstart,
                    fast_admm=args.fast_admm,
                    adaptive_admm_penalty=args.adaptive_penalty,
                    num_batches_per_subp=args.num_batches_per_subp,
                    hyperparam_table=(print_admm_hyperparams if (B == batch_sizes[0] and r == 0) else None),
                    run_sequential_baseline=False,
                    run_monolithic_baseline=True,
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

                out["results"][str(B)][str(r)] = {
                    "B": B,
                    "repeat_index": r,
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
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the CSV data file to train on.")
    parser.add_argument("--results_path", type=Path, required=True,
                        help="Path to write the results .json (parent dirs created as needed).")
    parser.add_argument("--B_start", type=int, metavar="B_START", required=True,
                        help="Start of training batch sizes to sweep over")
    parser.add_argument("--B_end", type=int, metavar="B_END", required=True,
                        help="End of training batch sizes to sweep over")
    parser.add_argument("--num_training_repeats", type=int, default=1,
                        help="Repeats per B (default 1). Each repeat uses a different training-subset "
                             "permutation and a different model initialisation, with a fixed test set, "
                             "so the spread across repeats gives a CI at each B.")
    parser.add_argument("--num_batches_per_subp", type=int, default=1,
                        help="Multi-block ADMM: number of trajectories (K) stitched block-diagonally "
                                "into each ADMM subproblem. K=1 (default) is classic consensus ADMM (one "
                                "subproblem per trajectory); K>=B is a single monolithic block. "
                                "num_subproblems = ceil(B / K), split evenly via np.array_split.")
    parser.add_argument("--fast_admm", action="store_true", 
                        help="Run Goldstein 2014 Fast ADMM (Alg 8: Nesterov momentum + restart, "
                                "fixed ρ, mean consensus) instead of standard ADMM. Pass the bare flag "
                                "to enable; omit it to run standard ADMM.")
    parser.add_argument("--consensus_aggregator", type=str, required=True,
                        help="ADMM consensus aggregator: 'mean' | 'under-relaxed'")
    parser.add_argument("--subproblem_warmstart", type=str, required=True, 
                        help="ADMM subproblem warmstart: 'consensus' | 'relaxed' | 'previous'")
    parser.add_argument("--adaptive_penalty", action=argparse.BooleanOptionalAction, default=True,
                        help="Adaptive penalty via Boyd residual balancing (default on). Pass --no-adaptive-penalty "
                             "for a fixed penalty=√B run. Forced off (fixed penalty) whenever --fast-admm is set.")
    parser.add_argument("--admm_warm_init", action="store_true", default=False,
                        help="ADMM: warm-start every subproblem from a batch-0 warmup solve. Omit for "
                             "the default cold per-worker (seeded) Xavier init.")
    args = parser.parse_args()
    main()
