"""
Plot predicted vs true state trajectories for a test batch, using the fitted NN
parameters stored in a scaling-sweep .json by run_scaling_sweep.py.

No retraining and no Pyomo dependency: the trained weights/biases are read straight
from the JSON (results[<B>]["trained_params"][<model>]) and forward-integrated with a
small standalone tanh-MLP RHS (identical to the models' nn_prediction). The ground-truth
test trajectory is re-read from the CSV named in the JSON config, using the per-B
test_batch_ids that were saved alongside the params.

Produces one PNG per call, tagged with the method(s) overlaid:
    predicted_vs_true_T=<T>_r=<repeat>_batch=<test_batch_id>_<methods>.png
    (e.g. ..._mono-mb-s1.png). Methods: mono (monolithic), mb (multi-block ADMM) and s1 (ADMM
    with B=T, i.e. one subproblem per trajectory), coloured as in plot_scaling_sweep.py.

Examples:
    python -m data.bioreactor_simulation.plot_predicted_trajectories \
        --results-path data/bioreactor_simulation/scaling_sweep_results.json --B 20
    python -m data.bioreactor_simulation.plot_predicted_trajectories \
        --results-path .../results.json --B 8 --test-index 2 --models admm
    # overlay models stored in separate sweep files (e.g. a mono sweep and a multi-block ADMM sweep)
    python -m data.bioreactor_simulation.plot_predicted_trajectories \
        --results-path .../mono.json .../mb.json --T 50 --labels "mNLP, multi-block ADMM"
    # ... plus the B=T (single-trajectory-per-subproblem) ADMM run, drawn in its scaling-plot colour
    python -m data.bioreactor_simulation.plot_predicted_trajectories \
        --results-path .../mono.json .../mb.json .../s1.json --T 50 \
        --labels "mNLP, ADMM (5), ADMM (5) with B=T"
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.integrate import solve_ivp

from data.bioreactor_simulation.plot_scaling_sweep import FONTSIZE, AXIS_LABEL_PAD, SERIES_COLOURS, parse_labels

# ------------------------------- STYLE -------------------------------
MODEL_ORDER = ('mono', 'admm')
MODEL_LABELS = {'mono': 'Monolithic', 'admm': 'ADMM consensus'}
# Each fitted model is drawn as one of these method groups, in this order, with the same colour as
# the group's line in plot_scaling_sweep.py: monolithic, multi-block ADMM and single-trajectory
# (B=T, one subproblem per trajectory) ADMM. An ADMM run is the B=T group when it has as many
# subproblems as training trajectories.
GROUP_ORDER = ('mono', 'mb', 's1')
GROUP_LABELS = {'mono': 'Monolithic', 'mb': 'ADMM consensus', 's1': 'ADMM consensus (B=T)'}
GROUP_COLOURS = {'mono': SERIES_COLOURS[0], 'mb': SERIES_COLOURS[1], 's1': SERIES_COLOURS[2]}

def method_group(model, entry):
    """'mono', 'mb' or 's1' for model `model` of a repeat entry."""
    if model == 'mono':
        return 'mono'
    return 's1' if entry.get("num_subproblems") == entry.get("B") else 'mb'

# States whose y-axis gets a tick at every integer (matplotlib otherwise picks every 2).
INTEGER_TICK_STATES = ('O', 'Xd')
# The rotated y-label's box has ~1 pt more blank space beside the glyphs than the x-label's,
# so trim it by that much to make the visible label-to-tick gap match on both axes.
YLABEL_PAD = AXIS_LABEL_PAD - 1

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
        "--results-path", type=Path, nargs="+", required=True,
        help="One or more scaling-sweep .json files produced by run_scaling_sweep.py. With several "
             "files (e.g. a mono sweep and a multi-block ADMM sweep) their models are overlaid; "
             "the files must share the same test set, and a model may appear in only one of them.",
    )
    parser.add_argument(
        "--plots-path", type=Path, default="./data/bioreactor_simulation",
        help="Directory to write the PNG into (default: same directory as --results-path).",
    )
    parser.add_argument(
        "--T", type=int, required=True,
        help="Which training-set size T in the sweep to plot (must exist in the JSON).",
    )
    parser.add_argument(
        "--test-index", type=int, default=5,
        help="Which test trajectory to plot (0-based, into that T's test_batch_ids).",
    )
    parser.add_argument(
        "--repeat", type=int, default=0,
        help="Which repeat of this T to plot (0-based; default 0). Each repeat has its own fitted "
             "params and its own training subset (the test set is fixed across repeats).",
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_ORDER), default=None, metavar="MODEL",
        help=f"Which models to overlay (default: all present). Choose from: {', '.join(MODEL_ORDER)}.",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None, metavar="LABEL",
        help="Legend label for each --results-path, in the same order (default: the model names). "
             "A file holding one model is labelled by its label alone, otherwise '<model> (<label>)'. "
             "Multi-word labels can be given as a comma-separated list, e.g. --labels \"mNLP, multi-block "
             "ADMM\", or as separate quoted words, e.g. --labels \"mNLP\" \"multi-block ADMM\".",
    )
    args = parser.parse_args()
    if args.labels is not None:
        args.labels = parse_labels(args.labels)
        if len(args.labels) != len(args.results_path):
            parser.error(f"--labels has {len(args.labels)} entries {args.labels} but --results-path has "
                         f"{len(args.results_path)}; give one label per results file.")

    for p in args.results_path:
        if not p.is_file():
            parser.error(f"--results-path does not exist or is not a file: {p}")
    if args.plots_path is None:
        args.plots_path = args.results_path[0].parent
    args.plots_path.mkdir(parents=True, exist_ok=True)
    return args

def main():
    args = parse_args()

    # group ('mono' / 'mb' / 's1') -> (model, fitted params); one fitted model per group
    trained_params, test_batch_ids, data_path, state_cols = {}, None, None, None
    legend_labels = dict(GROUP_LABELS)  # group -> legend label, overridden per file by --labels
    for i, path in enumerate(args.results_path):
        with open(path) as f:
            data = json.load(f)

        config = data.get("config", {})
        # test ground truth: 'test_data_path' in new JSONs; older single-file JSONs only have 'data_path'
        f_data_path = config.get("test_data_path", config.get("data_path"))
        f_state_cols = config.get("state vars")
        if f_data_path is None or f_state_cols is None:
            raise ValueError(
                f"{path} is missing config['test_data_path'] / config['state vars']; "
                "re-run run_scaling_sweep.py to regenerate it."
            )

        results = data.get("results", {})
        T_key = str(args.T)
        if T_key not in results:
            available = ", ".join(sorted((k for k in results if k.lstrip('-').isdigit()), key=int))
            raise SystemExit(f"T={args.T} not in {path}. Available T: {available}")

        reps = results[T_key]
        rep_key = str(args.repeat)
        if not isinstance(reps, dict) or rep_key not in reps:
            avail = ", ".join(sorted((r for r in reps if str(r).lstrip('-').isdigit()), key=int)) \
                if isinstance(reps, dict) else "(none)"
            raise SystemExit(f"repeat {args.repeat} not found for T={args.T} in {path}. Available repeats: {avail}")
        entry = reps[rep_key]
        if "error" in entry:
            raise SystemExit(f"T={args.T} repeat={args.repeat} in {path} recorded a training error, "
                             f"nothing to plot: {entry['error']}")

        f_params = entry.get("trained_params", {})
        if not f_params:
            raise SystemExit(
                f"T={args.T} in {path} has no 'trained_params'. Re-run run_scaling_sweep.py "
                "(it now persists the fitted NN weights needed for prediction)."
            )
        f_test_ids = entry.get("test_batch_ids")
        if not f_test_ids:
            raise SystemExit(f"T={args.T} in {path} has no 'test_batch_ids'.")

        # overlaying only makes sense if every file was evaluated on the same test set
        if data_path is None:
            data_path, state_cols, test_batch_ids = f_data_path, f_state_cols, f_test_ids
        elif (f_data_path, f_state_cols, f_test_ids) != (data_path, state_cols, test_batch_ids):
            raise SystemExit(f"{path} uses a different test set / state vars from {args.results_path[0]}; "
                             "cannot overlay their predictions.")
        f_groups = {method_group(m, entry): (m, params) for m, params in f_params.items()}
        dup = set(f_groups) & set(trained_params)
        if dup:
            raise SystemExit(f"method(s) {', '.join(sorted(dup))} appear in more than one --results-path file.")
        trained_params.update(f_groups)
        if args.labels is not None:
            for g, (m, _) in f_groups.items():
                legend_labels[g] = (args.labels[i] if len(f_groups) == 1
                                    else f"{MODEL_LABELS.get(m, m)} ({args.labels[i]})")

    ti = args.test_index
    if not (0 <= ti < len(test_batch_ids)):
        raise SystemExit(f"--test-index {ti} out of range [0, {len(test_batch_ids) - 1}] for T={args.T}.")
    test_bid = test_batch_ids[ti]

    # Which models to plot (as method groups, in GROUP_ORDER).
    have_models = {m for m, _ in trained_params.values()}
    if args.models is not None:
        missing = [m for m in args.models if m not in have_models]
        if missing:
            print(f"WARNING: requested model(s) absent for T={args.T}, skipping: {', '.join(missing)}")
    models_to_plot = [g for g in GROUP_ORDER if g in trained_params
                      and (args.models is None or trained_params[g][0] in args.models)]
    if not models_to_plot:
        raise SystemExit(f"No requested models available for T={args.T} (have: {', '.join(sorted(have_models))}).")

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

    print(f"Plotting T={args.T}, repeat={args.repeat}, test Batch_{test_bid} (test index {ti}); "
          f"models: {', '.join(models_to_plot)}")

    # --- predictions from stored params ---
    preds = {}
    for g in models_to_plot:
        Ws, bs = load_params(trained_params[g][1])
        preds[g] = predict(Ws, bs, y0, t_grid)

    # -------------------------------- PLOT --------------------------------
    n = len(state_cols)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for i, sp in enumerate(state_cols):
        ax = axes[i // ncols][i % ncols]
        ax.plot(t_grid, true_traj[:, i], 'k.', ms=4, label="True observations")
        for g in models_to_plot:
            pred = preds.get(g)
            if pred is not None:
                ax.plot(t_grid, pred[:, i], '-', lw=1.6,
                        color=GROUP_COLOURS[g], label=legend_labels[g])
        ax.set_xlabel("Time", fontsize=FONTSIZE, labelpad=AXIS_LABEL_PAD)
        ax.set_ylabel(sp, fontsize=FONTSIZE, labelpad=YLABEL_PAD)
        if sp in INTEGER_TICK_STATES:
            ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.tick_params(axis="both", which="both", labelsize=FONTSIZE)
        ax.grid(True, alpha=0.15)

    # Shared legend placed in the spare subplot slot (or below the figure).
    handles, labels = axes[0][0].get_legend_handles_labels()
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if n < nrows * ncols:
        axes[n // ncols][n % ncols].legend(handles, labels, loc="center", fontsize=FONTSIZE, frameon=False)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=FONTSIZE, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    model_tag = "-".join(models_to_plot)
    out_path = args.plots_path / f"predicted_vs_true_T={args.T}_r={args.repeat}_batch={test_bid}_{model_tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
