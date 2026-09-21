"""
Plot predicted vs true state trajectories for a test batch, using the fitted NN
parameters stored in a scaling-sweep .json by run_scaling_sweep.py.

No retraining and no Pyomo dependency: the trained weights/biases are read straight
from the JSON (results[<B>]["trained_params"][<model>]) and forward-integrated with a
small standalone tanh-MLP RHS (identical to the models' nn_prediction). The ground-truth
test trajectory is re-read from the CSV named in the JSON config, using the per-B
test_batch_ids that were saved alongside the params.

Produces one PNG per call, tagged with the model(s) overlaid:
    predicted_vs_true_B=<B>_batch=<test_batch_id>_<models>.png
    (e.g. ..._mono-admm.png for both, ..._admm.png for a single model)

Examples:
    python -m data.bioreactor_simulation.plot_predicted_trajectories \
        --results-path data/bioreactor_simulation/scaling_sweep_results.json --B 20
    python -m data.bioreactor_simulation.plot_predicted_trajectories \
        --results-path .../results.json --B 8 --test-index 2 --models admm
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------- STYLE -------------------------------
MODEL_ORDER = ('mono', 'admm')
MODEL_LABELS = {'mono': 'Monolithic', 'admm': 'ADMM consensus'}
MODEL_COLOURS = {'mono': '#c87878', 'admm': '#6e96c8'}

# Tight integration tolerances for prediction (match the training MSE eval).
PRED_OPTIONS = dict(method='RK45', rtol=1e-7, atol=1e-9)

def nn_rhs(t, y, Ws, bs):
    """Standalone tanh-MLP RHS, identical to the models' nn_prediction(t, y, Ws, bs)."""
    out = np.tanh(np.dot(y.reshape(1, -1), Ws[0]) + bs[0])
    for W, b in zip(Ws[1:-1], bs[1:-1]):
        out = np.tanh(np.dot(out, W) + b)
    out = np.dot(out, Ws[-1]) + bs[-1]
    return out.flatten()

def load_params(model_params):
    """Turn the JSON {'Ws': [...], 'bs': [...]} nested lists into arrays."""
    Ws = [np.asarray(W, dtype=float) for W in model_params["Ws"]]
    bs = [np.asarray(b, dtype=float) for b in model_params["bs"]]
    return Ws, bs

def predict(Ws, bs, y0, t_grid):
    """Forward-integrate the fitted NN; return (N, D) prediction or None on failure."""
    dt = float(t_grid[1] - t_grid[0])
    sol = solve_ivp(
        fun=nn_rhs, t_span=(t_grid[0], t_grid[-1]), y0=np.asarray(y0, float),
        t_eval=t_grid, args=(Ws, bs), max_step=dt, **PRED_OPTIONS,
    )
    if not sol.success:
        print(f"  WARNING: prediction failed (status={sol.status}: {sol.message}); skipping.")
        return None
    return sol.y.T

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-path", type=Path, required=True,
        help="Path to the scaling-sweep .json produced by run_scaling_sweep.py.",
    )
    parser.add_argument(
        "--plots-path", type=Path, default="./data/bioreactor_simulation",
        help="Directory to write the PNG into (default: same directory as --results-path).",
    )
    parser.add_argument(
        "--B", type=int, required=True,
        help="Which training-set size B in the sweep to plot (must exist in the JSON).",
    )
    parser.add_argument(
        "--test-index", type=int, default=5,
        help="Which test trajectory to plot (0-based, into that B's test_batch_ids).",
    )
    parser.add_argument(
        "--repeat", type=int, default=0,
        help="Which repeat of this B to plot (0-based; default 0). Each repeat has its own fitted "
             "params and its own training subset (the test set is fixed across repeats).",
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_ORDER), default=None, metavar="MODEL",
        help=f"Which models to overlay (default: all present). Choose from: {', '.join(MODEL_ORDER)}.",
    )
    args = parser.parse_args()

    if not args.results_path.is_file():
        parser.error(f"--results-path does not exist or is not a file: {args.results_path}")
    if args.plots_path is None:
        args.plots_path = args.results_path.parent
    args.plots_path.mkdir(parents=True, exist_ok=True)
    return args

def main():
    args = parse_args()

    with open(args.results_path) as f:
        data = json.load(f)

    config = data.get("config", {})
    data_path = config.get("data_path")
    state_cols = config.get("state vars")
    if data_path is None or state_cols is None:
        raise ValueError(
            f"{args.results_path} is missing config['data_path'] / config['state vars']; "
            "re-run run_scaling_sweep.py to regenerate it."
        )

    results = data.get("results", {})
    B_key = str(args.B)
    if B_key not in results:
        available = ", ".join(sorted((k for k in results if k.lstrip('-').isdigit()), key=int))
        raise SystemExit(f"B={args.B} not in results. Available B: {available}")

    reps = results[B_key]
    rep_key = str(args.repeat)
    if not isinstance(reps, dict) or rep_key not in reps:
        avail = ", ".join(sorted((r for r in reps if str(r).lstrip('-').isdigit()), key=int)) \
            if isinstance(reps, dict) else "(none)"
        raise SystemExit(f"repeat {args.repeat} not found for B={args.B}. Available repeats: {avail}")
    entry = reps[rep_key]
    if "error" in entry:
        raise SystemExit(f"B={args.B} repeat={args.repeat} recorded a training error, "
                         f"nothing to plot: {entry['error']}")

    trained_params = entry.get("trained_params", {})
    if not trained_params:
        raise SystemExit(
            f"B={args.B} has no 'trained_params' in the JSON. Re-run run_scaling_sweep.py "
            "(it now persists the fitted NN weights needed for prediction)."
        )

    test_batch_ids = entry.get("test_batch_ids")
    if not test_batch_ids:
        raise SystemExit(f"B={args.B} has no 'test_batch_ids' in the JSON.")
    ti = args.test_index
    if not (0 <= ti < len(test_batch_ids)):
        raise SystemExit(f"--test-index {ti} out of range [0, {len(test_batch_ids) - 1}] for B={args.B}.")
    test_bid = test_batch_ids[ti]

    # Which models to plot.
    models_to_plot = [m for m in MODEL_ORDER if m in trained_params]
    if args.models is not None:
        missing = [m for m in args.models if m not in trained_params]
        if missing:
            print(f"WARNING: requested model(s) absent for B={args.B}, skipping: {', '.join(missing)}")
        models_to_plot = [m for m in args.models if m in trained_params]
    if not models_to_plot:
        raise SystemExit(f"No requested models available for B={args.B} (have: {', '.join(trained_params)}).")

    # --- ground truth from the CSV (re-read, using the saved test batch id) ---
    if not os.path.isfile(data_path):
        raise SystemExit(f"CSV data file from JSON config not found: {data_path}")
    df = pd.read_csv(data_path)
    batch_df = df[df['Batch'] == f"Batch_{test_bid}"].reset_index(drop=True)
    if batch_df.empty:
        raise SystemExit(f"Batch_{test_bid} not found in {data_path}.")
    t_grid = batch_df['Time'].values
    true_traj = batch_df[state_cols].values # (N, D)
    y0 = true_traj[0]

    print(f"Plotting B={args.B}, repeat={args.repeat}, test Batch_{test_bid} (test index {ti}); "
          f"models: {', '.join(models_to_plot)}")

    # --- predictions from stored params ---
    preds = {}
    for m in models_to_plot:
        Ws, bs = load_params(trained_params[m])
        preds[m] = predict(Ws, bs, y0, t_grid)

    # -------------------------------- PLOT --------------------------------
    n = len(state_cols)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for i, sp in enumerate(state_cols):
        ax = axes[i // ncols][i % ncols]
        ax.plot(t_grid, true_traj[:, i], 'k.', ms=4, label="True")
        for m in models_to_plot:
            pred = preds.get(m)
            if pred is not None:
                ax.plot(t_grid, pred[:, i], '-', lw=1.6,
                        color=MODEL_COLOURS[m], label=MODEL_LABELS[m])
        ax.set_title(sp)
        ax.set_xlabel("Time")
        ax.set_ylabel(sp)
        ax.grid(True, alpha=0.3)

    # Shared legend placed in the spare subplot slot (or below the figure).
    handles, labels = axes[0][0].get_legend_handles_labels()
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if n < nrows * ncols:
        axes[n // ncols][n % ncols].legend(handles, labels, loc="center", fontsize=12)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels))

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    model_tag = "-".join(models_to_plot)
    out_path = args.plots_path / f"predicted_vs_true_B={args.B}_r={args.repeat}_batch={test_bid}_{model_tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
