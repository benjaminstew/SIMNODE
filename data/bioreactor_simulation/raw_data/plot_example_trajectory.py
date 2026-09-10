"""
Overlay all state variables for several randomly-chosen trajectories from a generated CSV.

Each state variable keeps one colour (so variables stay distinguishable across the overlay),
and each trajectory is drawn at a different opacity so the individual trajectories separate
out visually. Useful for eyeballing the spread/variability across training trajectories.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

STATE_COLS = ["G", "O", "X", "Xd", "P", "L", "CO2"]

def plot_trajectory(csv_path: str, num_trajs: int = 5, save: bool = False, seed: int = 0):
    """
    Overlay `num_trajs` randomly-chosen trajectories from the CSV on one shared axes.

    Args:
        csv_path: path to the generated CSV (columns: Batch, Time, and the state vars).
        num_trajs: number of trajectories to draw (clamped to the number available).
        save: save a PNG next to the CSV instead of showing the figure.
        seed: seeds the random trajectory choice for reproducibility (None = nondeterministic).
    """
    if num_trajs < 1:
        raise ValueError(f"num_trajs must be >= 1; got {num_trajs}.")

    df = pd.read_csv(csv_path)
    batches = df["Batch"].unique()
    n_avail = len(batches)
    n = min(num_trajs, n_avail)
    if num_trajs > n_avail:
        print(f"Requested {num_trajs} trajectories but only {n_avail} available; plotting {n}.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(batches, size=n, replace=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10.colors
    # One opacity per trajectory, spread from strong to faint so overlaid trajectories separate.
    alphas = np.linspace(1.0, 0.35, n) if n > 1 else [1.0]

    for t_idx, batch in enumerate(chosen):
        traj = df[df["Batch"] == batch].sort_values("Time")
        for i, sp in enumerate(STATE_COLS):
            ax.plot(
                traj["Time"], traj[sp],
                lw=1.4, color=colors[i % len(colors)], alpha=alphas[t_idx],
                # Label each state variable once (on the first trajectory) to keep the legend small.
                label=sp if t_idx == 0 else None,
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, framealpha=0.9, title="State variable")

    fig.suptitle(f"{n} random trajectories — {os.path.basename(csv_path)}", fontsize=12)
    fig.tight_layout()

    if save:
        out_path = os.path.splitext(csv_path)[0] + f"_overlay_{n}traj.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="Path to the generated CSV.")
    parser.add_argument(
        "--num-trajs", type=int, default=5,
        help="Number of randomly-chosen trajectories to overlay (default: 5).",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save PNG next to the CSV instead of showing."
    )
    args = parser.parse_args()

    plot_trajectory(args.csv, num_trajs=args.num_trajs, save=args.save, seed=args.seed)
