"""
Plot the results in scaling-sweep .json files produced by run_scaling_sweep.py.

Inputs come in up to three method groups, each taking one or more .json files:
    --mono-input-data  monolithic NLP runs              (model 'mono')
    --mb-input-data    multi-block ADMM runs            (model 'admm')
    --s1-input-data    single-trajectory-per-subproblem ADMM runs (model 'admm'; optional)
The files of a group are merged into one series (e.g. a T=2..150 sweep plus a set of large-T runs),
so each method is one line with one colour, marker and legend label (--mono-label / --mb-label /
--s1-label) everywhere it appears, including both panels of a --break-at plot. Only the group's own
model is read from its files. The per-iteration ADMM plots are written once per ADMM group.

Example:
    python -m data.bioreactor_simulation.plot_scaling_sweep \
        --mono-input-data mono_2to150.json mono_200to1000.json \
        --mb-input-data mb_10to100.json mb_101to1000.json \
        --plots test_mse_scaling walltime_scaling --break-at 150

Two families of plot, each individually selectable via --plots:

  Across-B (scaling) plots — use every B in the sweep:
    test_mse_scaling   -> scaling_test_mse_vs_batches.png    (final test MSE vs B, per model)
    train_mse_scaling  -> scaling_train_mse_vs_batches.png   (final train MSE vs B, per model)
    walltime_scaling   -> scaling_walltime_vs_batches.png    (wall-clock train time vs B, per model)
  With --combined_plots True, the selected scaling plots are instead drawn as one figure,
  scaling_combined_vs_batches.png: a column of subplots (train MSE, test MSE, wall-clock, top to
  bottom) sharing the x-axis and a single legend.

  Per-B, per-iteration plots — ADMM only, one PNG per selected B:
    training_loss    -> training_loss_vs_iter_B=<B>.png     
    test_mse         -> test_mse_vs_iter_B=<B>.png
    train_mse        -> train_mse_vs_iter_B=<B>.png
    primal_residual  -> primal_residual_vs_iter_B=<B>.png
    dual_residual    -> dual_residual_vs_iter_B=<B>.png
    primal_dual_residual -> primal_dual_residual_vs_iter_B=<B>.png  (both residuals + both acceptance regions)

  The primal/dual residual plots shade the convergence acceptance region (residual <= --stop-band * eps).

The scaling plots draw each method's mean over its successful repeats, with a band showing their
min-max range. Failed runs are left out of both. A run counts as failed if it
recorded an error, if its final train MSE is non-finite or above --fail-train-mse (a crashed solve
leaves the untrained model, MSE ~10), or if it is missing altogether (fewer repeats than the file's
num_training_repeats, e.g. a job killed for running out of memory or time). A T with any failed runs
is drawn as a red cross, at the mean of its successful runs (or along the top edge if none
succeeded), with a note saying how many runs failed.

--break-at X splits the x-axis into a wide linear left panel (T <= X) and a narrow right panel
(T > X, log scale by default, see --right-xscale), so a dense small-T sweep and a few large-T runs
fit on one figure. Each panel gets its own y-range, fitted to the data it shows; the right panel's
y tick labels sit on its right-hand side.
"""
import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, FixedLocator, ScalarFormatter, MaxNLocator
from matplotlib.lines import Line2D

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

# With several results files, each (file, model) line takes the next colour/marker in this fixed
# order (single-file plots keep MODEL_COLOURS). Categorical palette, never cycled.
SERIES_COLOURS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948')
SERIES_MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*')

# Font size for the axis labels, ticks and legend of the across-B scaling plots.
FONTSIZE = 14
AXIS_LABEL_PAD = 8   # gap (points) between the tick numbers and the axis title

# Method groups: (key, CLI name, model read from its files, default legend label, colour, marker).
# Fixed per method, so a method looks the same in every plot and in both --break-at panels.
METHOD_GROUPS = (
    ("mono", "mono", "mono", "mNLP", SERIES_COLOURS[0], "o"),
    ("mb", "mb", "admm", "multi-block ADMM", SERIES_COLOURS[1], "s"),
    ("s1", "s1", "admm", "single-trajectory ADMM", SERIES_COLOURS[2], "^"),
)

# Failed-run markers on the scaling plots.
FAIL_COLOUR = '#d62728'
FAIL_TRAIN_MSE = 1.0      # default --fail-train-mse: successful runs sit at ~1e-2, crashed solves at ~10
# Candidate tick values for the right (large-T) panel of a --break-at plot on a log x-axis.
RIGHT_LOG_TICKS = (100, 200, 500, 1000, 2000, 5000, 10000)  # 1-2-5 steps keep 14 pt labels apart

# Convergence acceptance region is (residual <= NEAR_STOPPING_BAND * eps). This factor mirrors
# `near_stopping_band` in the ADMM solver and is NOT stored in the sweep JSON, so it is exposed
# via --stop-band (defaulting to this value) in case a run used a different band.
NEAR_STOPPING_BAND = 1

# Which plot keys are across-B vs per-B-per-iteration.
SCALING_PLOTS = ('test_mse_scaling', 'train_mse_scaling', 'walltime_scaling')
ITER_PLOTS = ('training_loss', 'test_mse', 'train_mse',
              'primal_residual', 'dual_residual', 'primal_dual_residual')
ALL_PLOTS = SCALING_PLOTS + ITER_PLOTS


# ------------------------------- IO / PARSING -------------------------------
def parse_labels(tokens):
    """--labels tokens -> list of labels. If any comma is present, the tokens are re-joined with
    spaces and split on commas (so multi-word labels work quoted or not); otherwise each token is one
    label, as before. Surrounding whitespace / quotes are stripped and empty entries dropped."""
    joined = " ".join(tokens)
    parts = joined.split(",") if "," in joined else tokens
    return [p.strip().strip("\"'").strip() for p in parts if p.strip().strip("\"'").strip()]

def str_to_bool(s):
    """'True' / 'False' style CLI value -> bool."""
    v = str(s).strip().lower()
    if v in ("true", "t", "yes", "y", "1"):
        return True
    if v in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got '{s}'")

def file_safe(label):
    """Label -> filename-safe string (spaces and other punctuation become underscores)."""
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", label).strip("_")

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for key, cli, model, label, _, _ in METHOD_GROUPS:
        what = {"mono": "monolithic NLP", "mb": "multi-block ADMM",
                "s1": "single-trajectory-per-subproblem ADMM (optional)"}[key]
        parser.add_argument(
            f"--{cli}-input-data", type=Path, nargs="+", default=None, metavar="JSON",
            help=f"One or more scaling-sweep .json files of {what} runs (model '{model}'), merged into "
                 "one series. Each T may appear in only one of them.",
        )
        parser.add_argument(
            f"--{cli}-label", default=label, metavar="LABEL",
            help=f"Legend label for the --{cli}-input-data series (default '{label}').",
        )
    parser.add_argument(
        "--plots-path", type=Path, default=None,
        help="Directory to write PNGs into (default: directory of the first input file).",
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
    parser.add_argument(
        "--fail-train-mse", type=float, default=FAIL_TRAIN_MSE, metavar="MSE",
        help="Scaling plots: a run whose final train MSE is non-finite or above this counts as a failed "
             f"solve and is left out of the mean / range (default {FAIL_TRAIN_MSE:g}).",
    )
    parser.add_argument(
        "--break-at", type=float, default=None, metavar="T",
        help="Scaling plots: split the x-axis at this T into a wide linear left panel (T <= value) and "
             "a narrow right panel (T > value). Default: one continuous axis.",
    )
    parser.add_argument(
        "--right-xscale", choices=("log", "linear"), default="log",
        help="x scale of the right panel when --break-at is set (default log).",
    )
    parser.add_argument(
        "--hide-fail-markers", nargs="+", default=[], choices=[g[0] for g in METHOD_GROUPS],
        metavar="GROUP",
        help="Scaling plots: method group(s) (mono / mb / s1) drawn without red failed-run crosses. "
             "Their failed or missing runs are still left out of the mean / range.",
    )
    parser.add_argument(
        "--combined_plots", "--combined-plots", type=str_to_bool, nargs="?", const=True, default=False,
        metavar="BOOL",
        help="True: draw the selected scaling plots (train MSE, test MSE, wall-clock, top to bottom) as "
             "one figure, scaling_combined_vs_batches.png, of vertically stacked subplots with a shared "
             "x-axis and legend, instead of one PNG each (default False).",
    )
    args = parser.parse_args()

    # --- error handling on the user-supplied paths / labels ---
    all_paths = []
    for key, cli, *_ in METHOD_GROUPS:
        for p in getattr(args, f"{cli}_input_data") or []:
            if not p.is_file():
                parser.error(f"--{cli}-input-data file does not exist or is not a file: {p}")
            all_paths.append(p)
    if not all_paths:
        parser.error("give at least one of --mono-input-data / --mb-input-data / --s1-input-data.")
    labels = [getattr(args, f"{cli}_label") for _, cli, *_ in METHOD_GROUPS
              if getattr(args, f"{cli}_input_data")]
    if len(set(labels)) != len(labels):
        parser.error(f"the method legend labels must differ (got {labels}).")
    if args.plots_path is None:
        args.plots_path = all_paths[0].parent
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

def mean_range(vals):
    """(mean, lo, hi, n) across a list of per-repeat scalars, or None if none are finite:
    the mean over repeats and the range (lo = min, hi = max) of the repeats themselves.
    With n=1 the range collapses to the value (no band drawn)."""
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return None
    return float(a.mean()), float(a.min()), float(a.max()), int(a.size)

def run_failed(entry, m, fail_train_mse):
    """True if model m's run in this repeat entry failed: a recorded error, or a non-finite /
    implausibly large final train MSE (a crashed solve keeps the untrained parameters)."""
    if not isinstance(entry, dict) or "error" in entry:
        return True
    d = entry.get("final_mses", {}).get(m)
    tr = d.get("train") if isinstance(d, dict) else None
    return tr is None or not np.isfinite(tr) or tr > fail_train_mse

def load_method_groups(args):
    """Load and merge the files of each method group given on the command line.

    Returns [(key, label, model, colour, marker, results, expected)] in METHOD_GROUPS order, where
    results is the merged results[<B>][<repeat>] dict and expected[B_key] is the number of repeats
    that file's config asked for (used to count runs missing from the JSON, e.g. killed for
    out-of-memory or time limit)."""
    groups = []
    for key, cli, model, _, colour, marker in METHOD_GROUPS:
        paths = getattr(args, f"{cli}_input_data") or []
        if not paths:
            continue
        results, expected = {}, {}
        for path in paths:
            data = load_results(path)
            n_rep = data.get("config", {}).get("num_training_repeats")
            for B_key, reps in data["results"].items():
                if B_key in results:
                    raise SystemExit(f"T={B_key} appears in more than one --{cli}-input-data file "
                                     f"(second one: {path}); give the files distinct T ranges.")
                results[B_key] = reps
                if n_rep is not None:
                    expected[B_key] = int(n_rep)
        if model not in models_present(results):
            raise SystemExit(f"--{cli}-input-data files hold no '{model}' runs.")
        groups.append((key, getattr(args, f"{cli}_label"), model, colour, marker, results, expected))
    return groups

def models_present(results):
    """Union of model keys that appear in any repeat's final_mses, in canonical order."""
    seen = set()
    for B_key in results:
        for entry in repeat_entries(results, B_key):
            seen.update(entry.get("final_mses", {}).keys())
    return [m for m in MODEL_ORDER if m in seen]


# ------------------------------- SCALING PLOTS -------------------------------
def _scaling_series(results, expected, keys, kind, m, fail_train_mse):
    """Model m's (xs, ys, lo, hi, nfail) aggregated ACROSS SUCCESSFUL REPEATS, or None if m has no
    runs. kind in {'test_mse','train_mse','walltime'}.

    For each B we collect one scalar per successful repeat (final test/train MSE, or wall-clock
    time), then report ys=mean across them and (lo,hi)=min/max over those repeats. Failed runs
    (see run_failed) and runs missing from the JSON (fewer than expected[B] repeats) are left out of
    the statistics and counted in nfail. A B where every run failed gets ys=lo=hi=nan.
    """
    xs, ys, los, his, nfails = [], [], [], [], []
    for k in keys:
        reps = results.get(k, {})
        reps = reps if isinstance(reps, dict) else {}
        entries = [reps[r] for r in sorted((r for r in reps if str(r).lstrip("-").isdigit()), key=int)]
        # a repeat entry that never ran model m (e.g. a mono-only file) is not a failure of m
        entries = [e for e in entries if not isinstance(e, dict) or "error" in e
                   or m in e.get("final_mses", {})]
        if not entries:
            continue
        vals, nfail = [], 0
        for entry in entries:
            if run_failed(entry, m, fail_train_mse):
                nfail += 1
                continue
            if kind == "walltime":
                v = entry.get("train_runtimes", {}).get(m)
            else:
                d = entry["final_mses"][m]
                v = d.get("test") if kind == "test_mse" else d.get("train")
            if v is not None and np.isfinite(v):
                vals.append(float(v))
            else:
                nfail += 1
        nfail += max(0, expected.get(k, 0) - len(entries))   # repeats that never reached the JSON
        stats = mean_range(vals)
        mean, lo, hi = stats[:3] if stats is not None else (np.nan,) * 3
        xs.append(int(k)); ys.append(mean); los.append(lo); his.append(hi); nfails.append(nfail)
    if not xs:
        return None
    return tuple(np.array(a, float) for a in (xs, ys, los, his, nfails))

def _scaling_lines(datasets, kind, fail_train_mse, hide_fail_markers=()):
    """One (legend label, colour, marker, xs, ys, lo, hi, nfail) line per method group with data.
    Groups whose key is in hide_fail_markers get nfail = 0 everywhere, so they are drawn with normal
    markers and no red crosses (their failed runs are still left out of the mean / range)."""
    lines = []
    for key, label, model, colour, marker, results, expected in datasets:
        series = _scaling_series(results, expected, sorted_B_keys(results), kind, model, fail_train_mse)
        if series is not None:
            if key in hide_fail_markers:
                series = series[:4] + (np.zeros_like(series[4]),)
            lines.append((label, colour, marker) + series)
    return lines

def _format_break_axes(axl, axr, left_x, right_x, break_at, right_xscale):
    """x-limits / ticks of a --break-at left / right panel pair, plus the break styling."""
    pad = 0.03 * (break_at - left_x.min())
    axl.set_xlim(left_x.min() - pad, break_at + pad)
    lo, hi = right_x.min(), right_x.max()
    if right_xscale == "log":
        axr.set_xscale("log")
        axr.set_xlim(lo / 1.15, hi * 1.15)
        ticks = [t for t in RIGHT_LOG_TICKS if lo / 1.15 <= t <= hi * 1.15]
        axr.xaxis.set_major_locator(FixedLocator(ticks))
        axr.xaxis.set_major_formatter(ScalarFormatter())
        axr.xaxis.set_minor_locator(NullLocator())
    else:
        rpad = 0.06 * (hi - lo) if hi > lo else 0.1 * hi
        axr.set_xlim(lo - rpad, hi + rpad)
        axr.xaxis.set_major_locator(MaxNLocator(3))
    # hide the facing spines and draw "//" break marks on the top and bottom edges
    axl.spines["right"].set_visible(False)
    axr.spines["left"].set_visible(False)
    # independent y-range on the right panel: its tick labels go on the far (right) side
    axr.yaxis.tick_right()
    axr.tick_params(axis="y", which="both", left=False)
    mark = dict(marker=[(-1, -1.5), (1, 1.5)], markersize=10, linestyle="none",
                color="k", mec="k", mew=1, clip_on=False)
    axl.plot([1, 1], [0, 1], transform=axl.transAxes, **mark)
    axr.plot([0, 0], [0, 1], transform=axr.transAxes, **mark)

def _make_scaling_axes(lines, break_at, right_xscale, nrows=1):
    """nrows rows of one axis each, or (with break_at) of a wide left / narrow right pair sharing y,
    with x-limits set. Rows share the x-axis (per panel column). Returns (fig, rows), where each row
    is [ax] or [axl, axr]."""
    all_x = np.concatenate([xs for *_, xs, _, _, _, _ in lines]) if lines else np.array([])
    # Each row gets the axes size of a single-plot figure (5 in tall): the figure grows by one plot
    # height per row plus the inter-row gaps (hspace is a fraction of the axes height).
    hspace = 0.12
    height = 5 * (nrows + (nrows - 1) * hspace)
    if break_at is None:
        fig, axs = plt.subplots(nrows, 1, figsize=(7.5, height), sharex=True, squeeze=False,
                                gridspec_kw={"hspace": hspace})
        return fig, [list(row) for row in axs]
    left_x, right_x = all_x[all_x <= break_at], all_x[all_x > break_at]
    if left_x.size == 0 or right_x.size == 0:
        raise SystemExit(f"--break-at {break_at:g} needs data on both sides "
                         f"(T range in the files: {all_x.min():g}..{all_x.max():g}).")
    fig, axs = plt.subplots(nrows, 2, figsize=(9, height), sharex="col", squeeze=False,
                            gridspec_kw={"width_ratios": [0.7, 0.3], "wspace": 0.06, "hspace": hspace})
    for axl, axr in axs:
        _format_break_axes(axl, axr, left_x, right_x, break_at, right_xscale)
    return fig, [list(row) for row in axs]

def _draw_scaling_row(axes, lines, kind, right_xscale):
    """Draw every method's line onto one row of panels (one axis, or a --break-at pair). Failed runs
    are excluded from the mean / range and marked with a red cross plus a count of the failed runs.
    Returns True if any failed-run cross was drawn."""
    log_y = kind != "walltime" and bool(lines) and all(
        np.all(ys[np.isfinite(ys)] > 0) for _, _, _, _, ys, _, _, _ in lines)
    any_fail = False
    for ax in axes:
        x0, x1 = ax.get_xlim()
        shown_y = []   # every y drawn on this panel (means, band edges), to fit its own y-range
        for name, colour, marker, xs, ys, lo, hi, nfail in lines:
            ok = np.isfinite(ys)
            inx = ok & (xs >= x0) & (xs <= x1)
            shown_y.append(ys[inx])
            failed = nfail > 0
            # line through every T with a mean; normal markers only where no run failed
            ax.plot(xs[ok], ys[ok], marker=marker, markersize=4, linewidth=1.6, color=colour,
                    label=name, markevery=[int(j) for j in np.flatnonzero(~failed[ok])])
            # Shade the range (min-max across successful repeats) wherever it is non-degenerate (n >= 2).
            if np.any(hi[ok] > lo[ok]):
                if kind == "walltime":
                    floor = 0.0  # linear axis starting at 0 s; nothing below is shown
                else:
                    pos = ys[ok & (ys > 0)]
                    floor = float(pos.min()) * 1e-2 if pos.size else 1e-12  # keep lower edge positive for log axis
                ax.fill_between(xs[ok], np.clip(lo[ok], floor, None), hi[ok],
                                color=colour, alpha=0.15, linewidth=0)
                shown_y += [np.clip(lo[inx], floor, None), hi[inx]]
            # red cross at every T with failed runs (on the panel that shows that T), with the
            # number of failed runs as a small note (a bare count: closely spaced large-T points
            # leave no room for words; the legend entry explains it)
            for x, y, n in zip(xs[failed], ys[failed], nfail[failed]):
                if not (x0 <= x <= x1):
                    continue
                any_fail = True
                if np.isfinite(y):   # at the mean of the successful runs
                    pt, kw = (x, y), {}
                else:                # every run failed: along the top edge
                    pt, kw = (x, 0.95), dict(transform=ax.get_xaxis_transform(), clip_on=False)
                ax.plot(*([v] for v in pt), "x", color=FAIL_COLOUR, markersize=8, mew=2, zorder=5, **kw)
                # count directly above the cross on the wall-time plot (below it, it would run into
                # the ADMM line under the mNLP points), directly below it on the MSE plots
                above = kind == "walltime"
                ax.annotate(f"{int(n)}", pt, xycoords=kw.get("transform", "data"),
                            xytext=(0, 7 if above else -7), textcoords="offset points",
                            ha="center", va="bottom" if above else "top",
                            fontsize=FONTSIZE - 3, color=FAIL_COLOUR, fontweight="bold")
        ax.tick_params(axis="both", which="both", labelsize=FONTSIZE)
        ax.grid(True, alpha=0.15)
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_minor_locator(NullLocator())  # only the major 10^k ticks
        if len(axes) == 2:
            # fit this panel's y-range to what it shows (autoscaling would use both panels' data)
            v = np.concatenate(shown_y) if shown_y else np.array([])
            v = v[np.isfinite(v) & ((v > 0) if log_y else True)]
            if v.size:
                y_lo, y_hi = v.min(), v.max()
                if log_y:
                    m = 10 ** (0.05 * max(np.log10(y_hi / y_lo), 0.5))   # 5% margin, >= 0.025 decades
                    ax.set_ylim(y_lo / m, y_hi * m)
                else:
                    ax.set_ylim(y_lo - 0.05 * (y_hi - y_lo), y_hi + 0.05 * (y_hi - y_lo) or 1.0)
            if ax is axes[1] and right_xscale == "log":
                ax.xaxis.set_minor_locator(NullLocator())
        if kind == "walltime":
            ax.set_ylim(bottom=0)
    return any_fail

def _set_xlabel(axes):
    """x-axis title under a row of panels, centred under both panels of a --break-at pair."""
    axes[0].set_xlabel("Number of training trajectories, T", fontsize=FONTSIZE, labelpad=AXIS_LABEL_PAD)
    if len(axes) == 2:
        # only the label's x is moved; labelpad still sets y
        pl, pr = axes[0].get_position(), axes[1].get_position()
        axes[0].xaxis.label.set_x(((pl.x0 + pr.x1) / 2 - pl.x0) / pl.width)

def _legend_entries(axes_list, any_fail):
    """Legend handles / labels gathered from the given axes (one entry per label), plus the
    failed-run cross if any was drawn."""
    entries = {}
    for ax in axes_list:
        for h, l in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(l, h)
    labels, handles = list(entries), list(entries.values())
    if any_fail:
        handles.append(Line2D([], [], marker="x", linestyle="none", color=FAIL_COLOUR,
                              markersize=8, mew=2))
        labels.append("Failed runs (count)")
    return handles, labels

def plot_scaling(datasets, kind, ylabel, out_path, fail_train_mse=FAIL_TRAIN_MSE,
                 break_at=None, right_xscale="log", hide_fail_markers=()):
    """datasets: method groups from load_method_groups. One line per method, with the method's fixed
    colour, marker and legend label (the same in both --break-at panels)."""
    lines = _scaling_lines(datasets, kind, fail_train_mse, hide_fail_markers)
    fig, (axes,) = _make_scaling_axes(lines, break_at, right_xscale)
    any_fail = _draw_scaling_row(axes, lines, kind, right_xscale)
    # same tick-numbers-to-title gap on both axes
    axes[0].set_ylabel(ylabel, fontsize=FONTSIZE, labelpad=AXIS_LABEL_PAD)
    _set_xlabel(axes)
    if lines:
        axes[0].legend(*_legend_entries([axes[0]], any_fail), frameon=False, fontsize=FONTSIZE)
    if len(axes) == 1:
        fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

def plot_scaling_combined(datasets, kinds, out_path, fail_train_mse=FAIL_TRAIN_MSE,
                          break_at=None, right_xscale="log", hide_fail_markers=()):
    """kinds: [(kind, ylabel)], drawn top to bottom as one column of subplots in a single figure.
    The rows share the x-axis (only the bottom row is labelled) and one legend above the figure."""
    rows_lines = [_scaling_lines(datasets, kind, fail_train_mse, hide_fail_markers) for kind, _ in kinds]
    fig, rows = _make_scaling_axes([l for lines in rows_lines for l in lines], break_at,
                                   right_xscale, nrows=len(kinds))
    any_fail = False
    for axes, (kind, ylabel), lines in zip(rows, kinds, rows_lines):
        any_fail |= _draw_scaling_row(axes, lines, kind, right_xscale)
        axes[0].set_ylabel(ylabel, fontsize=FONTSIZE, labelpad=AXIS_LABEL_PAD)
    fig.align_ylabels([axes[0] for axes in rows])
    if break_at is None:
        fig.tight_layout()
    _set_xlabel(rows[-1])
    handles, labels = _legend_entries([ax for axes in rows for ax in axes], any_fail)
    if handles:
        # centred above the top row (over both panels of a --break-at pair)
        pl, pr = rows[0][0].get_position(), rows[0][-1].get_position()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=((pl.x0 + pr.x1) / 2, pl.y1 + 0.005),
                   ncol=len(handles) if len(handles) <= 3 else 2, frameon=False, fontsize=FONTSIZE)
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

def plot_iter_metric(entry, B, kind, out_path, stop_band=NEAR_STOPPING_BAND, tag=None):
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
    ax.grid(True, alpha=0.15)
    ax.set_title(f"{ylabel_for(kind)} vs iteration (B = {B})" + (f" — {tag}" if tag else ""))
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
    datasets = load_method_groups(args)
    out_dir = args.plots_path
    selected = args.selected

    # --- across-B scaling plots (all files on the same axes) ---
    scaling_kw = dict(fail_train_mse=args.fail_train_mse, break_at=args.break_at,
                      right_xscale=args.right_xscale, hide_fail_markers=set(args.hide_fail_markers))
    # (plot key, series kind, y-label, file name), in the top-to-bottom order of the combined figure
    scaling_specs = (
        ("train_mse_scaling", "train_mse", "Final train-set MSE", "scaling_train_mse_vs_batches.png"),
        ("test_mse_scaling", "test_mse", "Final test-set MSE", "scaling_test_mse_vs_batches.png"),
        ("walltime_scaling", "walltime", "Wall-clock training time (s)", "scaling_walltime_vs_batches.png"),
    )
    chosen = [s for s in scaling_specs if s[0] in selected]
    if args.combined_plots and chosen:
        plot_scaling_combined(datasets, [(kind, ylabel) for _, kind, ylabel, _ in chosen],
                              out_dir / "scaling_combined_vs_batches.png", **scaling_kw)
    else:
        for _, kind, ylabel, fname in chosen:
            plot_scaling(datasets, kind, ylabel, out_dir / fname, **scaling_kw)

    # --- per-B per-iteration ADMM plots (one set per file) ---
    iter_selected = [p for p in ITER_PLOTS if p in selected]
    admm_groups = [g for g in datasets if g[2] == "admm"]
    multi = len(admm_groups) > 1
    for key, label, _, _, _, results, _ in (admm_groups if iter_selected else []):
        file_tag = f"_{key}" if multi else ""   # keep file names distinct when mb and s1 are both plotted
        all_B = sorted_B_keys(results)
        if args.batch_sizes is None:
            want = all_B
        else:
            want = [str(b) for b in args.batch_sizes]
            missing = [b for b in want if b not in results]
            if missing:
                print(f"WARNING [{label}]: requested B not in results, skipping: "
                      f"{', '.join(missing)} (available: {', '.join(all_B)})")
            want = [b for b in want if b in results]

        rep_key = str(args.repeat)
        for k in want:
            reps = results[k]
            if not isinstance(reps, dict) or rep_key not in reps:
                avail = ", ".join(sorted((r for r in reps if str(r).lstrip("-").isdigit()), key=int)) \
                    if isinstance(reps, dict) else "(none)"
                print(f"[{label}] skipping B={k}: repeat {args.repeat} not found (available: {avail})")
                continue
            entry = reps[rep_key]
            B = entry.get("B", k)
            if "error" in entry:
                print(f"[{label}] skipping B={B} repeat={args.repeat}: recorded training error ({entry['error']})")
                continue
            for kind in iter_selected:
                out_path = out_dir / f"{kind}_vs_iter_B={B}_r={args.repeat}{file_tag}.png"
                wrote = plot_iter_metric(entry, B, kind, out_path, stop_band=args.stop_band,
                                         tag=(label if multi else None))
                if not wrote:
                    print(f"[{label}] skipping {kind} for B={B} repeat={args.repeat}: no ADMM history for this metric")

    print("Done.")


if __name__ == "__main__":
    main()
