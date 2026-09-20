"""
Script to run a training-set-size scaling sweep and extract per-batch size training metrics.

Sweeps the number of training trajectories B (with a fixed test set) 
and, for each B, trains three Neural-ODE models on the same split:
-- monolithic collocation model (Pyomo/IPOPT, block-diagonal DC)   -> 'mono'
-- ADMM consensus collocation model (Pyomo/IPOPT, DC)              -> 'admm'
-- sequential Neural ODE baseline (PyTorch / torchdiffeq)          -> 'seq'

Produces .json file containing, for each B in sweep: 
-- wall-clock training time (s) 
-- train and test MSEs
-- final train and test MSEs
-- primal and dual residuals at each iteration (for ADMM)
-- number of IPOPT iterations (for ADMM and monolithic)
-- trained NN params per model (for forward prediction)
"""
import json
import argparse 
import os
from pathlib import Path
import numpy as np

from data.bioreactor_simulation.run_training import run_training

# ------------------------------- CONFIG -------------------------------
STATE_COLS = ['G', 'O', 'X', 'Xd', 'P', 'L', 'CO2']
NN_LAYER_SIZES = [len(STATE_COLS), 16, len(STATE_COLS)]
NUM_TEST = 25
SEED = 0 # fixes the train/test split per B (reproducible)

def extract_trained_params(method: str, model) -> dict:
    """Pull a model's trained NN weights/biases as JSON-serialisable nested lists.

    The live model objects (Pyomo/IPOPT state) cannot be serialised, but the fitted
    NN parameters are small ndarrays. We want those so the plot scripts can
    forward-integrate trajectories without re-running training.
      - 'admm' -> consensus_Ws / consensus_bs (the converged consensus model)
      - 'mono' -> convert_weights() / convert_biases() (read back from the solved model)
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


def print_admm_hyperparams(admm_config, admm_training_options, fast_admm, restart_eta, accel_start_iter):
    """Print a table of every ADMM / Fast ADMM hyperparameter (once, at the top of the run log).

    Passed to run_training as `hyperparam_printer`; run_training invokes it with the config it built.
    """
    rows = []
    # (1) training-loop options (everything in admm_training_options except the MSE-eval settings)
    for k in ("early_solver_options", "late_solver_options", "admm_abs_tol", "admm_rel_tol",
              "max_admm_iterations", "num_early_iterations", "num_convergence_confirmations",
              "near_stopping_band", "training_clock_budget", "consensus_aggregator", "subproblem_warmstart"):
        rows.append((k, admm_training_options.get(k)))
    # (2) adaptive-ρ (penalty) settings (from admm_config). adaptive_rho shows the EFFECTIVE value
    # (fast ADMM forces it False), and mu is the residual-balancing factor.
    rows.append(("adaptive_rho", admm_config.get("adaptive_penalty_update")))
    rows.append(("mu", admm_config.get("mu")))
    # (3)+(4) under-relaxation weight/schedule and warm-start schedule (from admm_config)
    for k in ("relaxation_alpha", "relaxation_start_iter", "warmstart_start_iter"):
        rows.append((k, admm_config.get(k)))
    # (4) Fast ADMM hyperparameters
    rows.append(("fast_admm", fast_admm))
    if fast_admm:
        rows.append(("restart_eta", restart_eta))
        rows.append(("accel_start_iter", accel_start_iter))

    width = max(len(k) for k, _ in rows)
    bar = "=" * 64
    print(f"\n{bar}\nADMM HYPERPARAMETERS ({'Fast ADMM' if fast_admm else 'standard ADMM'})\n{bar}")
    for k, v in rows:
        print(f"  {k:<{width}} : {v}")
    print(f"{bar}\n")


def main():
    batch_sizes = list(range(args.B_start, args.B_end + 1))
    print(f"Scaling sweep: B in {batch_sizes}, test set size = {NUM_TEST}")
    print(f"results -> {args.results_path}")
    out = {
        "config": {
            "data_path": args.data_path,
            "state vars": STATE_COLS,
            "nn_layer_sizes": NN_LAYER_SIZES,
            "num_test_trajectories": NUM_TEST,
            "batch_sizes": batch_sizes,
            "seed": SEED
        },
        "results": {},
    }

    def save():
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_path, "w") as f:
            json.dump(out, f, indent=2)

    for B in batch_sizes:
        print(f"\n{'#' * 70}\n# B = {B} training batches\n{'#' * 70}")
        try:
            res = run_training(
                data_path=args.data_path,
                state_cols=STATE_COLS,
                nn_layer_sizes=NN_LAYER_SIZES,
                num_train_trajectories=B,
                num_test_trajectories=NUM_TEST,
                seed=SEED,
                run_admm_baseline=True,
                consensus_aggregator=args.consensus_aggregator,
                subproblem_warmstart=args.subproblem_warmstart,
                fast_admm=args.fast_admm,
                adaptive_rho=args.adaptive_rho,
                mu=args.mu,
                # print the hyperparameter table once, at the top of the log (first B only)
                hyperparam_printer=(print_admm_hyperparams if B == batch_sizes[0] else None),
                run_sequential_baseline=False,
                run_monolithic_baseline=True,
                return_models=True
            )
            train_runtimes = {m: float(t) for m, t in res["train_runtimes"].items()}
            # final_mses[m] = {train, test, test_std, test_min, test_max}; std/min/max may be None (e.g. seq).
            final_mses = {
                m: {k: (float(v) if v is not None else None) for k, v in d.items()}
                for m, d in res.get("final_mses", {}).items()
            }
            
            # The *_history objects are already built as dicts-of-lists => JSON-serialisable
            mse_histories = res.get("mse_histories", {})
            residual_histories = res.get("residual_histories", {})
            objective_histories = res.get("objective_histories", {})

            # Retain the fitted NN params per model so plot_predicted_trajectories.py can
            # forward-integrate test trajectories from the JSON without retraining.
            trained_params = {
                m: extract_trained_params(m, model)
                for m, model in res.get("trained_models", {}).items()
            }

            out["results"][str(B)] = {
                "B": B,
                "train_batch_ids": res.get("train_batch_ids"),
                "test_batch_ids": res.get("test_batch_ids"),
                "train_runtimes": train_runtimes,
                "final_mses": final_mses,
                "mse_histories": mse_histories,
                "residual_histories": residual_histories,
                "objective_histories": objective_histories,
                "trained_params": trained_params,
            }
        except Exception as e:  # keep the sweep alive; record the failure for this B
            out["results"][str(B)] = {"B": B, "error": f"{type(e).__name__}: {e}"}
            print(f"  B={B} FAILED: {type(e).__name__}: {e}")
        save()
        print(f" [saved results through B={B}]")

    print(f"\nSweep complete. Results in {args.results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the CSV data file to train on.")
    parser.add_argument("--results-path", type=Path, required=True,
                        help="Path to write the results .json (parent dirs created as needed).")
    parser.add_argument("--B_start", type=int, metavar="B_START", required=True,
                        help="Start of training batch sizes to sweep over")
    parser.add_argument("--B_end", type=int, metavar="B_END", required=True,
                        help="End of training batch sizes to sweep over")
    parser.add_argument("--consensus-aggregator", type=str, required=True, 
                        help="ADMM consensus aggregator: 'mean' | 'under-relaxed'")
    parser.add_argument("--subproblem-warmstart", type=str, required=True,
                        help="ADMM subproblem warmstart: 'consensus' | 'relaxed' | 'previous'")
    parser.add_argument("--fast-admm", action="store_true",
                        help="Run Goldstein 2014 Fast ADMM (Alg 8: Nesterov momentum + restart, "
                             "fixed ρ, mean consensus) instead of standard ADMM. Pass the bare flag "
                             "to enable; omit it to run standard ADMM.")
    parser.add_argument("--adaptive-rho", action=argparse.BooleanOptionalAction, default=True,
                        help="Adaptive ρ via Boyd residual balancing (default on). Pass --no-adaptive-rho "
                             "for a fixed ρ=√B run. Forced off (fixed ρ) whenever --fast-admm is set.")
    parser.add_argument("--mu", type=float, default=3.0,
                        help="Adaptive-ρ residual-balancing factor μ: increase ρ when the primal/dual "
                             "ratio-to-tolerance exceeds μ, decrease when below 1/μ (default 3.0).")
    args = parser.parse_args()
    main()
