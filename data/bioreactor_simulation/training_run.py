"""
Parent method for running batched NODE training on a set of multiple, independent
trajectories, with three interchangeable training algorithms:

  - ADMM-based consensus collocation model (Pyomo/IPOPT, DC)        -> method key 'dc'
  - sequential Neural ODE baseline (PyTorch / torchdiffeq)          -> method key 'seq'
  - monolithic collocation model (Pyomo/IPOPT, block-diagonal DC)   -> method key 'mono'

main() trains each enabled model on the SAME train/test split and RETURNS the per-model
wall-clock training times (and MSE histories), so a sweep driver
(run_scaling_sweep.py) can compare them across training-set sizes.

Run from repo root with 'python -m data.bioreactor_simulation.training_run'.
"""
import time
import numpy as np
import pandas as pd
import torch

from src.models.simultaneous.admm_mp import ADMM
from src.models.simultaneous.node_pyomo_monolithic import NODEPyomo as MonolithicNODEPyomo
from src.models.sequential.node_torch_multitraj import train_sequential_multitraj

def training_run(
    data_path: str,
    state_cols: list,
    nn_layer_sizes: list,
    num_train_trajectories: int,
    num_test_trajectories: int = 5,
    run_admm_baseline: bool = True,
    run_sequential_baseline: bool = False,
    run_monolithic_baseline: bool = True,
    seed: int = None
) -> dict:
    """
    Train the enabled models on one train/test split and return their runtimes.

    Args:
        seed: seeds the train/test split (and the torch baseline) for reproducibility.

    Returns:
        dict {
          'train_runtimes': {method: wall_clock_seconds}, # methods that ran
          'mse_histories':  {method: history dict}, # 'dc' / 'seq' only
          'train_batch_ids': [...], 'test_batch_ids': [...],
        }
    """
    print(f"Using layer sizes: {nn_layer_sizes} (num_train={num_train_trajectories}, num_test={num_test_trajectories})")

    rng = np.random.default_rng(seed)


    # -------------------- SETUP --------------------
    df = pd.read_csv(data_path)
    all_batch_ids = sorted(int(b.split('_')[1]) for b in df['Batch'].unique())
    if num_train_trajectories + num_test_trajectories > len(all_batch_ids):
        raise ValueError(
            f"num_train + num_test = {num_train_trajectories + num_test_trajectories} exceeds the number of available "
            f"trajectories ({len(all_batch_ids)}) in {data_path}."
        )
    chosen = rng.choice(all_batch_ids, size=num_train_trajectories + num_test_trajectories, replace=False)
    train_batch_ids = sorted(int(x) for x in chosen[:num_train_trajectories])
    test_batch_ids = sorted(int(x) for x in chosen[num_train_trajectories:num_train_trajectories + num_test_trajectories])
    train_batches = {bid: df[df['Batch'] == f"Batch_{bid}"].reset_index(drop=True) for bid in train_batch_ids}
    test_batches = {bid: df[df['Batch'] == f"Batch_{bid}"].reset_index(drop=True) for bid in test_batch_ids}

    train_grid = train_batches[train_batch_ids[0]]['Time'].values
    Y_obs = np.array([train_batches[bid][state_cols].values for bid in train_batch_ids]) # (B, N, D)
    test_grid = test_batches[test_batch_ids[0]]['Time'].values
    Y_test_obs = np.array([test_batches[bid][state_cols].values for bid in test_batch_ids]) # (num_test, N, D)
    end_time = train_grid[-1]

    # per-iteration MSE recording for the ADMM consensus model
    dt = train_grid[1] - train_grid[0]
    mse_eval_options = {
        "train_grid": train_grid,
        "Y_obs": Y_obs,
        "test_grid": test_grid,
        "Y_test_obs": Y_test_obs,
        "rtol": 1e-7,
        "atol": 1e-9,
        "max_step": dt,
    }

    train_runtimes = {} # {method: training wall-clock seconds}
    mse_histories = {}
    final_mses = {} # {method: (final_train_mse, final_test_mse)} after convergence / budget elapsed
    method_labels = {
        'mono': 'monolithic (DC)',
        'dc': 'ADMM consensus (DC)',
        'seq': 'sequential (torchdiffeq)',
    }


    # ---------------- 1. ADMM consensus collocation model (DC) ----------------
    if run_admm_baseline:
        # softer tolerances / fewer IPOPT iterations for the first `num_early_iterations` ADMM
        # iterations (subproblems are far from consensus early on -> tight solves are wasted)
        admm_early_solver_options = {
            "max_iter": 100,
            "tol": 1e-3,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 5,
        }
        # standard tolerances for the remaining ADMM iterations
        admm_late_solver_options = {
            "max_iter": 500,
            "tol": 1e-5,
            "acceptable_tol": 1e-3,
            "acceptable_iter": 5,
        }
        admm_config = dict(
            Y_obs=Y_obs,                   
            layer_sizes=nn_layer_sizes,
            end_time=end_time,
            obs_times=train_grid,               
            state_lower_bound=-5,
            state_upper_bound=50,
            param_lower_bound=-50,
            param_upper_bound=50,
            l2_reg_param=1e-4,
            admm_penalty_param=1,               
            use_pool=True,                           
            transcription_method='dc',
            irrdc_param=None,                        
            num_res_eval_nodes=None                  
        )

        admm_training_options = dict(
            early_solver_options=admm_early_solver_options,
            late_solver_options=admm_late_solver_options,
            admm_primal_res_tol=0.1,                  
            admm_dual_res_tol=0.1,                   
            max_admm_iterations=150,
            num_early_iterations=10, # opening iters using early_solver_options
            training_clock_budget=30, # minutes; None = no wall-clock cap
            mse_eval_options=mse_eval_options
        )
        print("\n\nTraining ADMM consensus model (DC) ...\n\n")
        admm = ADMM(**admm_config)
        t0 = time.perf_counter()
        admm.run_admm_training(**admm_training_options)
        train_runtimes['dc'] = time.perf_counter() - t0
        mse_histories['dc'] = admm.mse_history
        final_mses['dc'] = (admm.mse_history['train_mse'][-1], admm.mse_history['test_mse'][-1])
        print(f"  {method_labels['dc']} training time: {train_runtimes['dc']:.1f}s")


    # ---------------- 2. Sequential Neural ODE baseline (PyTorch) ----------------
    if run_sequential_baseline:
        print("\n\nTraining sequential Neural ODE baseline ...\n\n")
        Y_obs_t = torch.tensor(Y_obs, dtype=torch.float32)
        train_grid_t = torch.tensor(train_grid, dtype=torch.float32)
        Y_test_t = torch.tensor(Y_test_obs, dtype=torch.float32)
        test_grid_t = torch.tensor(test_grid, dtype=torch.float32)
        t0 = time.perf_counter()
        torch_model, mse_histories['seq'] = train_sequential_multitraj(
            Y_obs=Y_obs_t,
            t_grid=train_grid_t,
            layer_sizes=nn_layer_sizes,
            num_epochs=20000,
            mini_batch_size=None, # None -> full batch over all B trajectories
            lambda_reg=1e-4,
            learning_rate=1e-4,
            pretrain_iterations=0, # >0 enables supervised derivative pre-training
            pretrain_fraction=0.2,
            Y_obs_test=Y_test_t,
            test_grid=test_grid_t,
            record_mse_every=10,
            time_budget_min=30, # minutes; None = no wall-clock cap
            solver_method="dopri5",
            rtol=1e-3,
            atol=1e-6,
            max_step=None,
            eval_rtol=1e-3,
            eval_atol=1e-6,
            device="cpu",
            seed=seed,
            verbose=True,
            log_every=50
        )
        train_runtimes['seq'] = time.perf_counter() - t0
        final_mses['seq'] = (mse_histories['seq']['train_mse'][-1], mse_histories['seq']['test_mse'][-1])
        print(f"{method_labels['seq']} training time: {train_runtimes['seq']:.1f}s")

    # ---------------- 3. Monolithic collocation model (block-diagonal DC) ----------------
    if run_monolithic_baseline:
        monolithic_solver_options = {
            "max_iter": 3000,
            "tol": 1e-6,
            "acceptable_tol": 1e-4,
            "acceptable_iter": 15
        }
        print("\n\nTraining monolithic Pyomo baseline ...\n\n")
        mono = MonolithicNODEPyomo(
            Y_obs=Y_obs,
            obs_times=train_grid,
            layer_sizes=nn_layer_sizes,
            end_time=end_time,
            state_lower_bound=-5, state_upper_bound=50,
            param_lower_bound=-50, param_upper_bound=50,
            l2_reg_param=1e-4,
            seed=seed
        )
        t0 = time.perf_counter()
        mono.solve_model(monolithic_solver_options)
        train_runtimes['mono'] = time.perf_counter() - t0

        # monolithic has no per-iteration MSE history, so compute its final MSE directly
        mono_train_mse, _ = mono.compute_prediction_mse(
            train_grid, Y_obs,
            rtol=mse_eval_options['rtol'], atol=mse_eval_options['atol'], max_step=mse_eval_options['max_step']
        )
        mono_test_mse, _ = mono.compute_prediction_mse(
            test_grid, Y_test_obs,
            rtol=mse_eval_options['rtol'], atol=mse_eval_options['atol'], max_step=mse_eval_options['max_step']
        )
        final_mses['mono'] = (mono_train_mse, mono_test_mse)
        print(f"{method_labels['mono']} training time: {train_runtimes['mono']:.1f}s")


    # ---------------- TRAINING SUMMARY TABLE ----------------
    n_pred = len(train_batch_ids) + len(test_batch_ids)
    print("\n" + "=" * 92)
    print(f"Training summary (num_train={num_train_trajectories}, num_test={num_test_trajectories}, {n_pred} trajectories)")
    print("Final MSE = model MSE after convergence or wall-clock budget elapsed")
    print("-" * 92)
    print(f"{'Method':<32}{'Train (s)':>12}{'Final train MSE':>22}{'Final test MSE':>22}")
    for m in ('mono', 'dc', 'seq'):
        if m in train_runtimes:
            train_mse, test_mse = final_mses.get(m, (float('nan'), float('nan')))
            print(f"{method_labels[m]:<32}{train_runtimes[m]:>12.2f}{train_mse:>22.6e}{test_mse:>22.6e}")
    print("=" * 92 + "\n")

    return {
        "train_runtimes": train_runtimes,
        "mse_histories": mse_histories,
        "final_mses": final_mses,
        "train_batch_ids": train_batch_ids,
        "test_batch_ids": test_batch_ids,
    }

