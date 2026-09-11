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
B_MIN, B_MAX = 2, 25 # sweep B = 2 .. 45 training trajectories
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


def main():
    batch_sizes = list(range(B_MIN, B_MAX + 1))
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
                run_sequential_baseline=False,
                run_monolithic_baseline=True,
                return_models=True 
            )
            train_runtimes = {m: float(t) for m, t in res["train_runtimes"].items()}
            # final_mses[m] = {train, test, test_min, test_max}; test_min/max may be None (e.g. seq).
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
    args = parser.parse_args()
    main()
