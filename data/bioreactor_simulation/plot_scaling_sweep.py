"""
Plot the results in a scaling-sweep .json produced by run_scaling_sweep.py.

Two families of plot, each individually selectable via --plots:

  Across-B (scaling) plots — use every B in the sweep:
    test_mse_scaling   -> scaling_test_mse_vs_batches.png    (final test MSE vs B, per model)
    train_mse_scaling  -> scaling_train_mse_vs_batches.png   (final train MSE vs B, per model)
    walltime_scaling   -> scaling_walltime_vs_batches.png    (wall-clock train time vs B, per model)

  Per-B, per-iteration plots — ADMM only, one PNG per selected B:
    training_loss    -> training_loss_vs_iter_B=<B>.png     
    test_mse         -> test_mse_vs_iter_B=<B>.png
    train_mse        -> train_mse_vs_iter_B=<B>.png
    primal_residual  -> primal_residual_vs_iter_B=<B>.png
    dual_residual    -> dual_residual_vs_iter_B=<B>.png
    primal_dual_residual -> primal_dual_residual_vs_iter_B=<B>.png  (both residuals + both acceptance regions)

  The primal/dual residual plots shade the convergence acceptance region (residual <= --stop-band * eps).
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

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

# Convergence acceptance region is (residual <= NEAR_STOPPING_BAND * eps). This factor mirrors
# `near_stopping_band` in the ADMM solver and is NOT stored in the sweep JSON, so it is exposed
# via --stop-band (defaulting to this value) in case a run used a different band.
NEAR_STOPPING_BAND = 1.5

# Which plot keys are across-B vs per-B-per-iteration.
SCALING_PLOTS = ('test_mse_scaling', 'train_mse_scaling', 'walltime_scaling')
ITER_PLOTS = ('training_loss', 'test_mse', 'train_mse',
              'primal_residual', 'dual_residual', 'primal_dual_residual')
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
    parser.add_argument(
        "--repeat", type=int, default=0, metavar="R",
        help="For the per-iteration ADMM plots, which repeat's history to plot (default 0). "
             "Iteration counts differ across repeats, so these plots show one repeat rather than "
             "an aggregate. Ignored by the across-B scaling plots (those aggregate over all repeats).",
    )
    parser.add_argument(
        "--stop-band", type=float, default=NEAR_STOPPING_BAND, metavar="MULT",
        help="Convergence acceptance region shaded on the residual plots is (residual <= MULT * eps). "
             "Mirrors the ADMM solver's near_stopping_band; it is not stored in the sweep JSON, so set "
             f"it to match your run (default {NEAR_STOPPING_BAND}).",
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

def repeat_entries(results, B_key):
    """Per-repeat entry dicts for a given B, sorted by repeat index, skipping error/malformed ones.

    New schema: results[<B>][<repeat>] = {entry}. This helper is the single place that descends
    into the repeat level, so the rest of the module stays repeat-agnostic.
    """
    reps = results.get(B_key, {})
    if not isinstance(reps, dict):
        return []
    out = []
    for rk in sorted((k for k in reps if str(k).lstrip("-").isdigit()), key=int):
        e = reps[rk]
        if isinstance(e, dict) and "error" not in e:
            out.append(e)
    return out

def mean_std_ci(vals, confidence=0.95):
    """(mean, std, lo, hi, n) across a list of per-repeat scalars, or None if none are finite.

    std is the sample standard deviation (ddof=1); (lo, hi) is the two-sided `confidence` CI on the
    mean using the Student-t distribution (appropriate for the small n of a repeat sweep). With n=1
    the std is 0 and the CI collapses to the mean (no ribbon drawn).
    """
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    n = a.size
    if n == 0:
        return None
    mean = float(a.mean())
    if n == 1:
        return mean, 0.0, mean, mean, 1
    std = float(a.std(ddof=1))
    sem = std / np.sqrt(n)
    half = float(student_t.ppf(0.5 + confidence / 2.0, df=n - 1)) * sem
    return mean, std, mean - half, mean + half, n

def models_present(results):
    """Union of model keys that appear in any repeat's final_mses, in canonical order."""
    seen = set()
    for B_key in results:
        for entry in repeat_entries(results, B_key):
            seen.update(entry.get("final_mses", {}).keys())
    return [m for m in MODEL_ORDER if m in seen]


# ------------------------------- SCALING PLOTS -------------------------------
def _scaling_series(results, keys, kind):
    """Per-model (xs, ys, lo, hi, sd, nmax) aggregated ACROSS REPEATS. kind in
    {'test_mse','train_mse','walltime'}.

    For each B we collect one scalar per repeat (final test/train MSE, or wall-clock time), then
    report ys=mean across repeats, (lo,hi)=95% CI on that mean (Student-t), sd=sample std across
    repeats. nmax is the largest per-B repeat count seen (for the legend / title). With a single
    repeat the CI collapses to the mean, so no ribbon is drawn.
    """
    series = {}
    nmax = 0
    for m in models_present(results):
        xs, ys, los, his, sds = [], [], [], [], []
        for k in keys:
            entries = repeat_entries(results, k)
            vals, B = [], None
            for entry in entries:
                B = entry.get("B", int(k))
                if kind == "walltime":
                    v = entry.get("train_runtimes", {}).get(m)
                else:
                    d = entry.get("final_mses", {}).get(m)
                    v = (d.get("test") if kind == "test_mse" else d.get("train")) \
                        if isinstance(d, dict) else None
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
            stats = mean_std_ci(vals)
            if stats is not None:
                mean, sd, lo, hi, n = stats
                nmax = max(nmax, n)
                xs.append(int(k) if B is None else B)
                ys.append(mean); los.append(lo); his.append(hi); sds.append(sd)
        if xs:
            series[m] = (np.array(xs, float), np.array(ys, float),
                         np.array(los, float), np.array(his, float), np.array(sds, float))
    return series, nmax

def plot_scaling(results, kind, ylabel, title, out_path):
    keys = sorted_B_keys(results)
    series, nmax = _scaling_series(results, keys, kind)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ci_label_used = False
    for m, (xs, ys, lo, hi, sd) in series.items():
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.6,
                color=MODEL_COLOURS[m], label=MODEL_LABELS[m])
        # Shade the 95% CI (across repeats) ribbon wherever it is non-degenerate (n >= 2).
        if np.any(hi > lo):
            pos = ys[ys > 0]
            floor = float(pos.min()) * 1e-2 if pos.size else 1e-12  # keep lower edge positive for log axis
            ax.fill_between(xs, np.clip(lo, floor, None), hi,
                            color=MODEL_COLOURS[m], alpha=0.15, linewidth=0,
                            label=(None if ci_label_used else "95% CI across repeats"))
            ci_label_used = True
    ax.set_xlabel("Number of training trajectories, B")
    ax.set_ylabel(ylabel)
    suffix = f"  (mean of {nmax} repeats)" if nmax > 1 else ""
    ax.set_title(title + suffix)
    if series:  # log y only makes sense with data present and positive
        if all(np.all(ys > 0) for _, ys, _, _, _ in series.values()):
            ax.set_yscale("log")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ------------------------------- PER-ITERATION PLOTS -------------------------------
def _admm_hist(entry, group, key):
    """Fetch entry[group]['admm'][key] or None."""
    return entry.get(group, {}).get("admm", {}).get(key)

def _shade_acceptance(ax, x, y, band, stop_band, colour, label):
    """Shade the acceptance region between the residual line and the acceptance ceiling (stop_band * eps).

    Only the area above the residual that lies within the acceptance band is shaded: visible only
    when (or as) the residual enters the band. `y` is the residual array, `band` is eps per iteration.
    Returns a y-floor for the caller to set ax.set_ylim(bottom=floor).
    """
    y = np.asarray(y, dtype=float)
    band = np.asarray(band, dtype=float)
    upper = stop_band * band
    pos = band[band > 0]
    floor = float(pos.min()) * 0.3 if pos.size else 1e-12
    ax.fill_between(x, y, upper, where=(y < upper), interpolate=True,
                    color=colour, alpha=0.3, linewidth=0, label=label)
    return floor

def plot_iter_metric(entry, B, kind, out_path, stop_band=NEAR_STOPPING_BAND):
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
        mean = np.asarray(y, dtype=float)
        x = range(len(mean))
        ax.plot(x, mean, color=col, linewidth=1.4, label="mean over trajectories")
        # ±1 std across trajectories (test: NUM_TEST trajs, train: B trajs), if recorded.
        std_series = _admm_hist(entry, "mse_histories", key + "_std")
        if std_series and len(std_series) == len(mean):
            std = np.asarray(std_series, dtype=float)
            upper = mean + std
            lower = mean - std
            # log axis can't render a non-positive lower edge, so clip it to a small positive
            # floor (2 decades below the smallest positive mean) and cap ylim to the means so
            # the clipped band doesn't blow out the view.
            pos = mean[mean > 0]
            floor = float(pos.min()) * 1e-2 if pos.size else 1e-12
            ax.fill_between(x, np.clip(lower, floor, None), upper,
                            color=col, alpha=0.2, linewidth=0,
                            label=r"$\pm 1$ std across trajectories")
            if pos.size:
                ax.set_ylim(bottom=float(pos.min()) * 0.5)
            ax.legend()
        ax.set_ylabel(f"{'Test' if kind == 'test_mse' else 'Train'} MSE")
        ax.set_yscale("log")

    elif kind in ("primal_residual", "dual_residual"):
        if kind == "primal_residual":
            y = _admm_hist(entry, "residual_histories", "primal_avg")
            band = _admm_hist(entry, "residual_histories", "eps_pri")
            line_label, band_label = "Primal residual", r"$\epsilon_{pri}$"
            ylabel = r"$\|r\|_2$"
            accept_label = fr"Accepted band ($\epsilon_{{pri}} \leq {stop_band:g}$)"
        else:
            y = _admm_hist(entry, "residual_histories", "dual")
            band = _admm_hist(entry, "residual_histories", "eps_dual")
            line_label, band_label = "Dual residual", r"$\epsilon_{dual}$"
            ylabel = r"$\|s\|_2$"
            accept_label = fr"Accepted band ($\epsilon_{{dual}} \leq {stop_band:g}$)"
        if not y:
            plt.close(fig); return False
        y = np.asarray(y, dtype=float)
        ax.plot(range(len(y)), y, color=col, linewidth=1.4, label=line_label)
        floor = None
        if band:
            band = np.asarray(band, dtype=float)
            m = min(len(y), len(band))
            ax.plot(range(m), band[:m], color="grey", linestyle="--",
                    linewidth=1.0, label=band_label)
            floor = _shade_acceptance(
                ax, np.arange(m), y[:m], band[:m], stop_band,
                (150 / 255, 200 / 255, 150 / 255), accept_label)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        if floor is not None:
            ax.set_ylim(bottom=floor)
        ax.legend()

    elif kind == "primal_dual_residual":
        yp = _admm_hist(entry, "residual_histories", "primal_avg")
        yd = _admm_hist(entry, "residual_histories", "dual")
        if not yp or not yd:
            plt.close(fig); return False
        bp = _admm_hist(entry, "residual_histories", "eps_pri")
        bd = _admm_hist(entry, "residual_histories", "eps_dual")
        c_pri, c_dual = MODEL_COLOURS["admm"], MODEL_COLOURS["mono"]
        yp = np.asarray(yp, dtype=float)
        yd = np.asarray(yd, dtype=float)
        ax.plot(range(len(yp)), yp, color=c_pri, linewidth=1.4, label="Primal residual")
        ax.plot(range(len(yd)), yd, color=c_dual, linewidth=1.4, label="Dual residual")
        floors = []
        for band, base, colour, tex, sub in (
            (bp, yp, c_pri, r"\epsilon_{pri}", "pri"),
            (bd, yd, c_dual, r"\epsilon_{dual}", "dual"),
        ):
            if not band:
                continue
            band = np.asarray(band, dtype=float)
            m = min(len(base), len(band))
            ax.plot(range(m), band[:m], color=colour, linestyle="--",
                    linewidth=0.9, alpha=0.7, label=fr"${tex}$")
            floors.append(_shade_acceptance(
                ax, np.arange(m), base[:m], band[:m], stop_band, colour,
                fr"Accepted band ($\epsilon_{{{sub}}} \leq {stop_band:g}$)"))
        ax.set_ylabel(r"$\|r\|_2,\;\|s\|_2$")
        ax.set_yscale("log")
        if floors:
            ax.set_ylim(bottom=min(floors))
        ax.legend(fontsize=8)

    else:
        plt.close(fig); return False

    ax.set_xlabel("ADMM iteration count")
    ax.set_title(f"{ylabel_for(kind)} vs iteration (B = {B})")
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
        "primal_dual_residual": "Primal & dual residual",
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

        rep_key = str(args.repeat)
        for k in want:
            reps = results[k]
            if not isinstance(reps, dict) or rep_key not in reps:
                avail = ", ".join(sorted((r for r in reps if str(r).lstrip("-").isdigit()), key=int)) \
                    if isinstance(reps, dict) else "(none)"
                print(f"skipping B={k}: repeat {args.repeat} not found (available: {avail})")
                continue
            entry = reps[rep_key]
            B = entry.get("B", k)
            if "error" in entry:
                print(f"skipping B={B} repeat={args.repeat}: recorded training error ({entry['error']})")
                continue
            for kind in iter_selected:
                out_path = out_dir / f"{kind}_vs_iter_B={B}_r={args.repeat}.png"
                wrote = plot_iter_metric(entry, B, kind, out_path, stop_band=args.stop_band)
                if not wrote:
                    print(f"skipping {kind} for B={B} repeat={args.repeat}: no ADMM history for this metric")

    print("Done.")


if __name__ == "__main__":
    main()
