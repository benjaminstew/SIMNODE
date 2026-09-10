"""
Plot the results in a scaling-sweep .json produced by run_scaling_sweep.py.

Two families of plot, each individually selectable via --plots:

  Across-B (scaling) plots — use every B in the sweep:
    test_mse_scaling   -> scaling_test_mse_vs_batches.png    (final test MSE vs B, per model)
    train_mse_scaling  -> scaling_train_mse_vs_batches.png   (final train MSE vs B, per model)
    walltime_scaling   -> scaling_walltime_vs_batches.png    (wall-clock train time vs B, per model)

  Per-B, per-iteration plots — ADMM only, one PNG per selected B:
    training_loss    -> training_loss_vs_iter_B=<B>.png      (sum of per-batch objective vs iter)
    test_mse         -> test_mse_vs_iter_B=<B>.png
    train_mse        -> train_mse_vs_iter_B=<B>.png
    primal_residual  -> primal_residual_vs_iter_B=<B>.png     (with adaptive eps_pri band)
    dual_residual    -> dual_residual_vs_iter_B=<B>.png        (with adaptive eps_dual band)

Examples:
    python -m data.bioreactor_simulation.plot_scaling_sweep \
        --results-path data/bioreactor_simulation/scaling_sweep_results.json \
        --plots test_mse_scaling walltime_scaling
    python -m data.bioreactor_simulation.plot_scaling_sweep \
        --results-path .../results.json --plots primal_residual dual_residual --batch-sizes 8 20
    python -m data.bioreactor_simulation.plot_scaling_sweep --results-path .../results.json --plots all
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------- STYLE -------------------------------
MODEL_LABELS = {
    'mono': 'Monolithic',
    'admm': 'ADMM consensus',
    'seq': 'Sequential',
}
MODEL_COLOURS = {
    'mono': (200 / 255, 120 / 255, 120 / 255),
    'admm': (110 / 255, 150 / 255, 200 / 255),
    'seq': (150 / 255, 200 / 255, 150 / 255),
}
MODEL_ORDER = ('mono', 'admm', 'seq')

# Which plot keys are across-B vs per-B-per-iteration.
SCALING_PLOTS = ('test_mse_scaling', 'train_mse_scaling', 'walltime_scaling')
ITER_PLOTS = ('training_loss', 'test_mse', 'train_mse', 'primal_residual', 'dual_residual')
ALL_PLOTS = SCALING_PLOTS + ITER_PLOTS


# ------------------------------- IO / PARSING -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-path", type=Path, required=True,
        help="Path to the scaling-sweep .json produced by run_scaling_sweep.py.",
    )
    parser.add_argument(
        "--plots-path", type=Path, default=None,
        help="Directory to write PNGs into (default: same directory as --results-path).",
    )
    parser.add_argument(
        "--plots", nargs="+", required=True, choices=list(ALL_PLOTS) + ["all"],
        metavar="PLOT",
        help="Which plot(s) to produce. Choose any of: "
             + ", ".join(ALL_PLOTS) + "; or 'all'.",
    )
    parser.add_argument(
        "--batch-sizes", nargs="*", type=int, default=None, metavar="B",
        help="For the per-iteration plots, which B to plot (default: every B in the sweep). "
             "Ignored by the across-B scaling plots.",
    )
    args = parser.parse_args()

    # --- error handling on the user-supplied paths ---
    if not args.results_path.is_file():
        parser.error(f"--results-path does not exist or is not a file: {args.results_path}")
    if args.plots_path is None:
        args.plots_path = args.results_path.parent
    args.plots_path.mkdir(parents=True, exist_ok=True)

    # Expand 'all'.
    selected = set(ALL_PLOTS) if "all" in args.plots else set(args.plots)
    args.selected = selected
    return args


def load_results(results_path):
    with open(results_path) as f:
        data = json.load(f)
    if "results" not in data or not data["results"]:
        raise ValueError(f"No 'results' block found in {results_path}.")
    return data


def sorted_B_keys(results):
    """Return the results keys sorted by integer B (skipping malformed ones)."""
    return sorted((k for k in results if str(k).lstrip("-").isdigit()), key=int)


def models_present(results):
    """Union of model keys that appear in any B's final_mses, in canonical order."""
    seen = set()
    for entry in results.values():
        seen.update(entry.get("final_mses", {}).keys())
    return [m for m in MODEL_ORDER if m in seen]


# ------------------------------- SCALING PLOTS -------------------------------
def _scaling_series(results, keys, kind):
    """Collect per-model (B, value) series. kind in {'test_mse','train_mse','walltime'}."""
    series = {}
    for m in models_present(results):
        xs, ys = [], []
        for k in keys:
            entry = results[k]
            if kind == "walltime":
                v = entry.get("train_runtimes", {}).get(m)
            else:  # final_mses: [train, test]
                pair = entry.get("final_mses", {}).get(m)
                if pair is None or len(pair) != 2:
                    v = None
                else:
                    v = pair[1] if kind == "test_mse" else pair[0]
            if v is not None and np.isfinite(v):
                xs.append(entry["B"])
                ys.append(float(v))
        if xs:
            series[m] = (np.array(xs, float), np.array(ys, float))
    return series


def plot_scaling(results, kind, ylabel, title, out_path):
    keys = sorted_B_keys(results)
    series = _scaling_series(results, keys, kind)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m, (xs, ys) in series.items():
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.6,
                color=MODEL_COLOURS[m], label=MODEL_LABELS[m])
    ax.set_xlabel("Number of training batches, B")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if series:  # log y only makes sense with data present and positive
        if all(np.all(ys > 0) for _, ys in series.values()):
            ax.set_yscale("log")
        ax.legend()
    ax.grid(True, which="both", linestyle="--", color="lightgrey", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ------------------------------- PER-ITERATION PLOTS -------------------------------
def _admm_hist(entry, group, key):
    """Fetch entry[group]['admm'][key] or None."""
    return entry.get(group, {}).get("admm", {}).get(key)


def plot_iter_metric(entry, B, kind, out_path):
    """One per-iteration ADMM plot for a single B. Returns True if a PNG was written."""
    fig, ax = plt.subplots(figsize=(7, 5))
    col = MODEL_COLOURS["admm"]

    if kind == "training_loss":
        obj = _admm_hist(entry, "objective_histories", "obj_per_batch")
        if not obj:
            plt.close(fig); return False
        y = [float(np.sum(per_batch)) for per_batch in obj]
        ax.plot(range(len(y)), y, color=col, linewidth=1.4)
        ax.set_ylabel("Total training objective (sum over batches)")
        ax.set_yscale("log")

    elif kind in ("test_mse", "train_mse"):
        key = "test_mse" if kind == "test_mse" else "train_mse"
        y = _admm_hist(entry, "mse_histories", key)
        if not y:
            plt.close(fig); return False
        ax.plot(range(len(y)), y, color=col, linewidth=1.4)
        ax.set_ylabel(f"{'Test' if kind == 'test_mse' else 'Train'} MSE")
        ax.set_yscale("log")

    elif kind in ("primal_residual", "dual_residual"):
        if kind == "primal_residual":
            y = _admm_hist(entry, "residual_histories", "primal_avg")
            band = _admm_hist(entry, "residual_histories", "eps_pri")
            ylabel, band_label = "Primal residual", r"$\epsilon_{pri}$"
        else:
            y = _admm_hist(entry, "residual_histories", "dual")
            band = _admm_hist(entry, "residual_histories", "eps_dual")
            ylabel, band_label = "Dual residual", r"$\epsilon_{dual}$"
        if not y:
            plt.close(fig); return False
        ax.plot(range(len(y)), y, color=col, linewidth=1.4, label=ylabel)
        if band:
            ax.plot(range(len(band)), band, color="grey", linestyle="--",
                    linewidth=1.0, label=band_label)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.legend()
    else:
        plt.close(fig); return False

    ax.set_xlabel("ADMM iteration")
    ax.set_title(f"{ylabel_for(kind)} vs iteration (B = {B})")
    ax.grid(True, which="both", linestyle="--", color="lightgrey", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return True


def ylabel_for(kind):
    return {
        "training_loss": "Training objective",
        "test_mse": "Test MSE",
        "train_mse": "Train MSE",
        "primal_residual": "Primal residual",
        "dual_residual": "Dual residual",
    }[kind]


# ------------------------------- MAIN -------------------------------
def main():
    args = parse_args()
    data = load_results(args.results_path)
    results = data["results"]
    out_dir = args.plots_path
    selected = args.selected

    # --- across-B scaling plots ---
    if "test_mse_scaling" in selected:
        plot_scaling(results, "test_mse", "Final test-set MSE",
                     "Final test MSE vs training-set size",
                     out_dir / "scaling_test_mse_vs_batches.png")
    if "train_mse_scaling" in selected:
        plot_scaling(results, "train_mse", "Final train-set MSE",
                     "Final train MSE vs training-set size",
                     out_dir / "scaling_train_mse_vs_batches.png")
    if "walltime_scaling" in selected:
        plot_scaling(results, "walltime", "Wall-clock training time (s)",
                     "Training time vs training-set size",
                     out_dir / "scaling_walltime_vs_batches.png")

    # --- per-B per-iteration ADMM plots ---
    iter_selected = [p for p in ITER_PLOTS if p in selected]
    if iter_selected:
        all_B = sorted_B_keys(results)
        if args.batch_sizes is None:
            want = all_B
        else:
            want = [str(b) for b in args.batch_sizes]
            missing = [b for b in want if b not in results]
            if missing:
                print(f"WARNING: requested B not in results, skipping: "
                      f"{', '.join(missing)} (available: {', '.join(all_B)})")
            want = [b for b in want if b in results]

        for k in want:
            entry = results[k]
            B = entry.get("B", k)
            if "error" in entry:
                print(f"skipping B={B}: recorded training error ({entry['error']})")
                continue
            for kind in iter_selected:
                out_path = out_dir / f"{kind}_vs_iter_B={B}.png"
                wrote = plot_iter_metric(entry, B, kind, out_path)
                if not wrote:
                    print(f"skipping {kind} for B={B}: no ADMM history for this metric")

    print("Done.")


if __name__ == "__main__":
    main()
