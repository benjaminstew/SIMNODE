"""
Training-set-size scaling sweep.

Sweeps the number of training trajectories B = 2 .. 45 (with a fixed 5-trajectory test
set) and, for each B, trains three Neural-ODE models on the SAME split.

  - monolithic collocation model (Pyomo/IPOPT, block-diagonal DC)   -> 'mono'
  - ADMM consensus collocation model (Pyomo/IPOPT, DC)              -> 'dc'
  - sequential Neural ODE baseline (PyTorch / torchdiffeq)          -> 'seq'

Produces one plot -- wall-clock training time (s) vs number of training batches B, one
line per model -- and saves the raw timings to ``scaling_sweep_results.json''

Run from the repo root:  python -m data.bioreactor_simulation.run_scaling_sweep
"""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data.bioreactor_simulation.training_run import training_run as training_main

# ------------------------------- CONFIG -------------------------------
DATA_PATH = "./data/bioreactor_simulation/raw_data/noise_level_01_eps0.01.csv"
STATE_COLS = ['G', 'O', 'X', 'Xd', 'P', 'L', 'CO2']
NN_LAYER_SIZES = [len(STATE_COLS), 16, len(STATE_COLS)]
NUM_TEST = 5
B_MIN, B_MAX = 2, 8 # sweep B = 2 .. 45 training trajectories
SEED = 0 # fixes the train/test split per B (reproducible)
RESULTS_PATH = Path(os.environ.get("CNODE_RESULTS_PATH", 
                    str(Path(__file__).with_name("scaling_sweep_results.json"))))
WALLCLOCK_PLOT_PATH = Path(__file__).with_name("scaling_walltime_vs_batches.png")
MSE_PLOT_PATH = Path(__file__).with_name("scaling_test_mse_vs_batches.png")

# plotting identity (methods in the desired legend order)
METHOD_ORDER = ('mono', 'dc', 'seq')
METHOD_LABELS = {
    'mono': 'Monolithic (DC)',
    'dc': 'ADMM consensus (DC)',
    'seq': 'Sequential (torchdiffeq)',
}
METHOD_COLORS = {
    'mono': (200 / 255, 120 / 255, 120 / 255),
    'dc': (160 / 255, 190 / 255, 200 / 255),
    'seq': (150 / 255, 200 / 255, 150 / 255),
}

def plot_test_mse(out, path):
    """Plot final test-set MSE vs number of training batches (one line per model)."""
    results = out["results"]
    Bs = sorted(int(b) for b in results.keys())

    fig, ax = plt.subplots(figsize=(7.5, 5))
    any_data = False
    for method in METHOD_ORDER:
        xs, ys = [], []
        for B in Bs:
            fm = results[str(B)].get("final_mses", {})
            pair = fm.get(method)  # [final_train_mse, final_test_mse]
            if pair is not None and len(pair) == 2 and np.isfinite(pair[1]):
                xs.append(B)
                ys.append(float(pair[1]))  # index 1 = final TEST MSE
        if xs:
            any_data = True
            ax.plot(xs, ys, marker='o', markersize=4, linewidth=1.6,
                    color=METHOD_COLORS[method], label=METHOD_LABELS[method])

    ax.set_xlabel("Number of training batches, B")
    ax.set_ylabel("Final test-set MSE")
    ax.set_title("Final test MSE vs training-set size")
    if any_data:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", color="lightgrey", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

def plot(out, path):
    """Plot wall-clock training time vs number of training batches (one line per model)."""
    results = out["results"]
    Bs = sorted(int(b) for b in results.keys())

    fig, ax = plt.subplots(figsize=(7.5, 5))
    any_data = False
    for method in METHOD_ORDER:
        xs, ys = [], []
        for B in Bs:
            rt = results[str(B)].get("train_runtimes", {})
            t = rt.get(method)
            if t is not None and np.isfinite(t):
                xs.append(B)
                ys.append(float(t))
        if xs:
            any_data = True
            ax.plot(xs, ys, marker='o', markersize=4, linewidth=1.6,
                    color=METHOD_COLORS[method], label=METHOD_LABELS[method])

    ax.set_xlabel("Number of training batches, B")
    ax.set_ylabel("Wall-clock training time (s)")
    ax.set_title("Training time vs training-set size")
    if any_data:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", color="lightgrey", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

def main():
    batch_sizes = list(range(B_MIN, B_MAX + 1))

    print(f"Scaling sweep: B in {batch_sizes}, test set = {NUM_TEST}")
    print(f"results -> {RESULTS_PATH}")

    out = {
        "config": {
            "data_path": DATA_PATH,
            "state_cols": STATE_COLS,
            "nn_layer_sizes": NN_LAYER_SIZES,
            "num_test": NUM_TEST,
            "batch_sizes": batch_sizes,
            "seed": SEED,
        },
        "results": {},
    }

    def save():
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(out, f, indent=2)

    for B in batch_sizes:
        print(f"\n{'#' * 70}\n# B = {B} training batches\n{'#' * 70}")
        try:
            res = training_main(
                data_path=DATA_PATH,
                state_cols=STATE_COLS,
                nn_layer_sizes=NN_LAYER_SIZES,
                num_train_trajectories=B,
                num_test_trajectories=NUM_TEST,
                run_admm_baseline=True,
                run_sequential_baseline=False,
                run_monolithic_baseline=True,
                seed=SEED,
            )
            # keep JSON-friendly per-method runtimes and final (train, test) MSEs
            runtimes = {m: float(t) for m, t in res["train_runtimes"].items()}
            final_mses = {m: [float(tr), float(te)] for m, (tr, te) in res.get("final_mses", {}).items()}
            out["results"][str(B)] = {
                "B": B,
                "train_batch_ids": res.get("train_batch_ids"),
                "test_batch_ids": res.get("test_batch_ids"),
                "train_runtimes": runtimes,
                "final_mses": final_mses,
            }
        except Exception as e:  # keep the sweep alive; record the failure for this B
            out["results"][str(B)] = {"B": B, "error": f"{type(e).__name__}: {e}"}
            print(f"  B={B} FAILED: {type(e).__name__}: {e}")
        save()
        print(f" [saved results through B={B}]")

    plot(out, WALLCLOCK_PLOT_PATH)
    plot_test_mse(out, MSE_PLOT_PATH)
    print(f"\nSweep complete. Results in {RESULTS_PATH}")


if __name__ == "__main__":
    main()
