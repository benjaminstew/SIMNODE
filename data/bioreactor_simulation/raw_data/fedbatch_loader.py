"""
Loader for fed-batch trajectories with bolus feeds, plus the phase-block structure that a
knotted (multi-phase) collocation model is built from.

A CSV written by FedBatchSim.simulate(with_feed=True) interleaves two kinds of row per batch:
  - observation rows: Time + state columns filled, G_Feed NaN
  - feed rows:        Time + G_Feed filled, state columns NaN
build_trajectory_array() in run_scaling_sweep.py cannot read these (it would pick up the feed
rows as NaN observations), so this module replaces it for feed data. It also works for CSVs
without a G_Feed column, in which case every trajectory is a single phase.

Conventions (match FedBatchSim._integrate_batch and Ross & Fahroo's knots):
  - Phase p of a trajectory with feed times tau_1 < ... < tau_K is the closed interval
    [tau_p, tau_{p+1}], with tau_0 = t_0 and tau_{K+1} = T. Phases overlap only at the feed times.
  - A state observed exactly at a feed time is the PRE-feed value x^- (that is what the simulator
    records), so it belongs to the phase that ENDS there: obs t in phase p iff tau_p < t <= tau_{p+1}
    (phase 0 also takes t == t_0).
  - The jump at the knot tau_p is x^+ = x^- + a_p e_G, i.e. the feed amount is added to the glucose
    concentration only; every other state is continuous.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATE_VARS = ['G', 'O', 'X', 'Xd', 'P', 'L', 'CO2']  # Scale is constant and not learned


def load_feed_trajectories(
    source: str | pd.DataFrame,
    batch_ids: list[int | str] | None = None,
    state_vars: list[str] = STATE_VARS,
) -> list[dict]:
    """
    Read a FedBatchSim CSV (or its DataFrame) into one dict per trajectory.

    Args:
        source: path to the CSV, or the DataFrame itself
        batch_ids: batches to load, as ints (85) or names ('Batch_85'); default: all, in file order
        state_vars: state columns to keep, in this order

    Returns:
        list of dicts, one per batch, each with
            'batch':        batch name, e.g. 'Batch_85'
            'obs_times':    (n,) observation times, ascending
            'Y_obs':        (n, D) observed states at those times
            'y0':           (D,) initial state, Y_obs[0]
            'feed_times':   (K,) feed times, ascending (K = 0 if no feeds)
            'feed_amounts': (K,) glucose added at each feed
    """
    df = pd.read_csv(source) if isinstance(source, str) else source
    if batch_ids is None:
        names = list(dict.fromkeys(df['Batch']))  # unique, in file order
    else:
        # accept 85, '85' or 'Batch_85'
        names = [b if isinstance(b, str) and not b.isdigit() else f"Batch_{b}" for b in batch_ids]
    has_feed = 'G_Feed' in df.columns

    trajs = []
    for name in names:
        b = df[df['Batch'] == name].sort_values('Time')
        if b.empty:
            raise ValueError(f"No rows for batch {name!r}.")
        obs = b[b[state_vars[0]].notna()]
        feeds = b[b['G_Feed'].notna()] if has_feed else b.iloc[0:0]
        obs_times = obs['Time'].to_numpy(dtype=float)
        trajs.append({
            'batch': name,
            'obs_times': obs_times,
            'Y_obs': obs[state_vars].to_numpy(dtype=float),
            'y0': obs[state_vars].to_numpy(dtype=float)[0],
            'feed_times': feeds['Time'].to_numpy(dtype=float) if has_feed else np.empty(0),
            'feed_amounts': feeds['G_Feed'].to_numpy(dtype=float) if has_feed else np.empty(0),
        })
    return trajs


def build_phase_blocks(trajs: list[dict], T: float | None = None) -> list[dict]:
    """
    Flatten trajectories into the list of phase blocks a knotted collocation model indexes over.
    Blocks are ordered trajectory by trajectory, phase by phase, so the block before a phase-p
    block (p >= 1) is always phase p-1 of the same trajectory.

    Args:
        trajs: output of load_feed_trajectories()
        T: end of the horizon (default: each trajectory's last observation time)

    Returns:
        list of dicts, one per (trajectory, phase), each with
            'traj':     trajectory index into trajs
            'phase':    phase index p (0 = before the first feed)
            't0', 't1': phase interval [tau_p, tau_{p+1}]
            'jump':     glucose added at t0 (None for phase 0): x^+ = x^- + jump e_G
            'prev':     index of the previous phase's block (None for phase 0)
            'obs_idx':  indices into trajs[traj]['obs_times'] of the observations in this phase
    """
    blocks = []
    for b, tr in enumerate(trajs):
        t_end = tr['obs_times'][-1] if T is None else T
        ft, fa = tr['feed_times'], tr['feed_amounts']
        if len(ft) and (ft[0] <= tr['obs_times'][0] or ft[-1] >= t_end):
            raise ValueError(f"{tr['batch']}: feed times must lie strictly inside (t0, T).")
        edges = np.concatenate([[tr['obs_times'][0]], ft, [t_end]])
        first_block = len(blocks)
        for p, (t0, t1) in enumerate(zip(edges[:-1], edges[1:])):
            t = tr['obs_times']
            in_phase = (t > t0) & (t <= t1)
            if p == 0:
                in_phase |= np.isclose(t, t0)
            blocks.append({
                'traj': b,
                'phase': p,
                't0': float(t0),
                't1': float(t1),
                'jump': float(fa[p - 1]) if p > 0 else None,
                'prev': first_block + p - 1 if p > 0 else None,
                'obs_idx': np.flatnonzero(in_phase),
            })
    return blocks


def summarise_blocks(trajs: list[dict], blocks: list[dict]) -> str:
    """One line per trajectory: number of phases, phase lengths and observations per phase."""
    lines = []
    for b, tr in enumerate(trajs):
        mine = [blk for blk in blocks if blk['traj'] == b]
        lens = ', '.join(f"{blk['t1'] - blk['t0']:.2f}" for blk in mine)
        nobs = ', '.join(str(len(blk['obs_idx'])) for blk in mine)
        lines.append(f"{tr['batch']}: {len(mine)} phases | lengths {lens} | obs per phase {nobs}")
    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--batches', nargs='*', default=None, help="e.g. 85 3 Batch_7 (default: all)")
    args = parser.parse_args()
    trajs = load_feed_trajectories(args.csv, args.batches)
    blocks = build_phase_blocks(trajs)
    print(summarise_blocks(trajs, blocks))
    print(f"{len(trajs)} trajectories, {len(blocks)} phase blocks")
