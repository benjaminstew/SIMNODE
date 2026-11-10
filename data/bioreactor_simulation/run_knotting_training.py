"""
Preliminary PS-knotting experiment on the fed-batch bioreactor data with bolus feeds.

Trains the knotted monolithic collocation NODE (KnottedNODEPyomo: one Chebyshev-Lobatto polynomial
per phase, a FIXED HARD KNOT G^+ = G^- + a_k at every feed) on a train split and evaluates it by
piecewise forward rollout of the trained network (restart at each feed) on the train and test sets.
Results, solver info and the trained weights go to one JSON file.

Examples:
    OMP_NUM_THREADS=1 python -m data.bioreactor_simulation.run_knotting_training \
        --results_path results/knotting/prelim_N8_B10.json --num_train 10 --num_test 10 --num_colloc_nodes 8

    OMP_NUM_THREADS=1 python -m data.bioreactor_simulation.run_knotting_training \
        --results_path results/knotting/irrdc.json --transcription_method irrdc --irrdc_param 1.0 --num_res_eval_nodes 11

(OMP_NUM_THREADS=1 avoids the libHSL/OpenMP crash on exit of the local /usr/local/bin/ipopt.)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from data.bioreactor_simulation.feed_data import STATE_VARS, load_feed_trajectories, summarise_blocks, build_phase_blocks
from src.models.simultaneous.node_pyomo_mono_knotting import KnottedNODEPyomo

RAW = Path(__file__).resolve().parent / "raw_data"
DEFAULT_TRAIN = RAW / "trainset_60trajectories_30points_0.0noise_feed.csv"
DEFAULT_TEST = RAW / "testset_40trajectories_30points_0.0noise_feed.csv"
JUMP_STATE_IDX = STATE_VARS.index('G')



def choose_batches(all_names, num, seed):
    """Random subset of batch names (all of them if num is None or >= len)."""
    if num is None or num >= len(all_names):
        return list(all_names)
    rng = np.random.default_rng(seed)
    return [all_names[i] for i in sorted(rng.choice(len(all_names), size=num, replace=False))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_data_path", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--test_data_path", type=Path, default=DEFAULT_TEST)
    p.add_argument("--results_path", type=Path, required=True, help="Output .json (parent dirs are created).")
    p.add_argument("--num_train", type=int, default=None, help="Training trajectories to use (default: all in the file).")
    p.add_argument("--num_test", type=int, default=None, help="Test trajectories to use (default: all in the file).")
    p.add_argument("--split_seed", type=int, default=0, help="Seed for the random choice of train/test subsets.")
    p.add_argument("--model_init_seed", type=int, default=0, help="Seed for the NN parameter initialisation.")
    p.add_argument("--hidden_layer_size", type=int, default=16)
    p.add_argument("--num_hidden_layers", type=int, default=1)
    p.add_argument("--num_colloc_nodes", type=int, default=8, help="Chebyshev-Lobatto nodes PER PHASE.")
    p.add_argument("--l2_reg_param", type=float, default=1e-3)
    p.add_argument("--param_bound", type=float, default=10.0)
    p.add_argument("--state_bounds", type=float, nargs=2, default=(-5.0, 50.0), metavar=("LO", "HI"))
    p.add_argument("--transcription_method", choices=("dc", "irrdc"), default="dc")
    p.add_argument("--irrdc_param", type=float, default=None, help="IRR-DC penalty weight mu; term is (1/(2 mu)) * integral ||r||^2.")
    p.add_argument("--num_res_eval_nodes", type=int, default=None, help="IRR-DC residual-evaluation nodes PER PHASE.")
    p.add_argument("--max_iter", type=int, default=3000)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--acceptable_tol", type=float, default=1e-7)
    p.add_argument("--acceptable_iter", type=int, default=15)
    p.add_argument("--print_level", type=int, default=5)
    args = p.parse_args()
    if args.transcription_method == "irrdc" and (args.irrdc_param is None or args.num_res_eval_nodes is None):
        p.error("--transcription_method irrdc needs --irrdc_param and --num_res_eval_nodes")

    # ---------------- data ----------------
    train_all = load_feed_trajectories(str(args.train_data_path))
    test_all = load_feed_trajectories(str(args.test_data_path))
    train_names = choose_batches([t['batch'] for t in train_all], args.num_train, args.split_seed)
    test_names = choose_batches([t['batch'] for t in test_all], args.num_test, args.split_seed + 1)
    train_trajs = [t for t in train_all if t['batch'] in set(train_names)]
    test_trajs = [t for t in test_all if t['batch'] in set(test_names)]
    state_dim = train_trajs[0]['Y_obs'].shape[1]
    layer_sizes = [state_dim] + [args.hidden_layer_size] * args.num_hidden_layers + [state_dim]
    dt = float(np.min(np.diff(train_trajs[0]['obs_times'])))
    eval_opts = {"rtol": 1e-7, "atol": 1e-9, "max_step": dt}

    blocks = build_phase_blocks(train_trajs)
    print(f"Train: {len(train_trajs)} trajectories from {args.train_data_path.name} | "
          f"Test: {len(test_trajs)} from {args.test_data_path.name}")
    print(summarise_blocks(train_trajs, blocks))
    print(f"Knotted model: {len(blocks)} phases x {args.num_colloc_nodes} nodes = {len(blocks) * args.num_colloc_nodes} nodes, "
          f"network {layer_sizes}\n")

    solver_options = {"max_iter": args.max_iter, "tol": args.tol, "acceptable_tol": args.acceptable_tol,
                      "acceptable_iter": args.acceptable_iter, "print_level": args.print_level}
    results = {
        "config": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                   "layer_sizes": layer_sizes, "eval_opts": eval_opts, "ipopt_solver_options": solver_options,
                   "omp_num_threads": os.environ.get("OMP_NUM_THREADS")},
        "train_batches": train_names, "test_batches": test_names,
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- train ----------------
    print("=" * 80 + "\nTraining KNOTTED monolithic NODE (hard knot at every feed)\n" + "=" * 80)
    t0 = time.perf_counter()
    model = KnottedNODEPyomo(
        train_trajs, num_colloc_nodes=args.num_colloc_nodes, layer_sizes=layer_sizes,
        state_lower_bound=args.state_bounds[0], state_upper_bound=args.state_bounds[1],
        param_lower_bound=-args.param_bound, param_upper_bound=args.param_bound,
        l2_reg_param=args.l2_reg_param, jump_state_idx=JUMP_STATE_IDX, seed=args.model_init_seed,
        transcription_method=args.transcription_method, irrdc_param=args.irrdc_param,
        num_res_eval_nodes=args.num_res_eval_nodes,
    )
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    info = model.solve_model(solver_options)
    solve_s = time.perf_counter() - t0

    # ---------------- evaluate (piecewise rollout with feeds) ----------------
    train_mse, train_std, train_min, train_max = model.compute_prediction_mse(train_trajs, return_range=True, **eval_opts)
    test_mse, test_std, test_min, test_max = model.compute_prediction_mse(test_trajs, return_range=True, **eval_opts)
    results["knotted"] = {
        "num_nodes_total": model.total_nodes, "num_phase_blocks": model.num_blocks,
        "build_time_s": build_s, "train_runtime_s": solve_s, "solve_info": info,
        "final_mses": {"train": train_mse, "train_std": train_std, "train_min": train_min, "train_max": train_max,
                       "test": test_mse, "test_std": test_std, "test_min": test_min, "test_max": test_max},
        "trained_params": {"Ws": [W.tolist() for W in model.convert_weights()],
                           "bs": [b.tolist() for b in model.convert_biases()]},
    }
    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- summary ----------------
    print("\n" + "=" * 90)
    print(f"Summary ({len(train_trajs)} train / {len(test_trajs)} test trajectories; MSE by piecewise rollout with feeds)")
    print("-" * 90)
    print(f"{'nodes':>8}{'solve (s)':>11}{'termination':>18}{'train MSE':>15}{'test MSE':>15}{'test std':>12}{'test max':>12}")
    print(f"{model.total_nodes:>8}{solve_s:>11.1f}{info.get('termination_condition', '?'):>18}"
          f"{train_mse:>15.3e}{test_mse:>15.3e}{test_std:>12.2e}{test_max:>12.2e}")
    print("=" * 90)
    print(f"Saved {args.results_path}")


if __name__ == "__main__":
    main()
