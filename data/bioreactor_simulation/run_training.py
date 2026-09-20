"""
Parent method for running batched NODE training on a set of multiple, independent
trajectories, with three interchangeable training algorithms:

-- ADMM-based consensus collocation model (Pyomo/IPOPT, DC)        -> method key 'admm'
-- sequential Neural ODE baseline (PyTorch / torchdiffeq)          -> method key 'seq'
-- monolithic collocation model (Pyomo/IPOPT, block-diagonal DC)   -> method key 'mono'

main() trains each enabled model on the same train/test split and returns the per-model
wall-clock training times, MSE histories and residuals. 
"""
import time
import numpy as np
import pandas as pd
import torch

from src.models.simultaneous.admm_mp import ADMM
from src.models.simultaneous.node_pyomo_monolithic import NODEPyomo as MonolithicNODEPyomo
from src.models.sequential.node_torch_multitraj import train_sequential_multitraj

def run_training(
    data_path: str,
    state_cols: list,
    nn_layer_sizes: list,
    num_train_trajectories: int,
    num_test_trajectories: int,
    seed: int,
    run_admm_baseline: bool = True,
    consensus_aggregator="under-relaxed", # "mean" | "under-relaxed"
    subproblem_warmstart="consensus", # "consensus" | "relaxed" | "previous"
    fast_admm: bool = False, # run Goldstein 2014 Fast ADMM (Alg 8) instead of standard ADMM
    restart_eta: float = 0.9, # Fast ADMM restart threshold η (only used when fast_admm)
    accel_start_iter: int = 25, # Fast ADMM: iter at which momentum turns on (only used when fast_admm)
    relaxation_alpha: float = 0.5, # under-relaxation weight α (θ̂ = α·θ + (1-α)·z); α<1 damps spikes
    relaxation_start_iter: int = 10, # iter after which under-relaxed aggregation activates
    warmstart_start_iter: int = 25, # iter after which subproblem warm start activates
    adaptive_rho: bool = True, # adaptive ρ via Boyd residual balancing; fast_admm forces fixed ρ
    mu: float = 3.0, # adaptive-ρ residual-balancing factor μ
    hyperparam_printer=None,
    run_sequential_baseline: bool = False,
    run_monolithic_baseline: bool = True,
    return_models: bool = True
) -> dict:
    """
    Train the models on one train/test split and return their metrics.

    Args:
        seed: seeds the train/test split (and the torch baseline) for reproducibility.

    Returns:
        dict {
          'train_batch_ids': [...],
          'test_batch_ids': [...],
          'train_runtimes': {method: wall_clock_seconds},
          'mse_histories':  {method: {train_mse: [...], test_mse: [...], ...}},
          'residual_histories': {admm: {primal_avg: [...], dual: [...], rho: [...], ...}},
          'objective_histories': {admm: {obj_per_batch: [...], ...}},
          'final_mses': {method: {train, test, test_std, test_min, test_max}},  # std/range = per-test-traj spread
          # the following are added only when return_models=True:
          'trained_models': {method: fitted model object},
          'test_grid': [...], 'Y_test_obs': ndarray, 'train_grid': [...], 'Y_obs': ndarray,
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

    # per-iteration training metric recording 
    dt = train_grid[1] - train_grid[0]
    mse_eval_options = {
        "train_grid": train_grid,
        "Y_obs": Y_obs,
        "test_grid": test_grid, 
        "Y_test_obs": Y_test_obs,
        "rtol": 1e-7,
        "atol": 1e-9,
        "max_step": dt
    }
    train_runtimes = {}
    mse_histories = {}
    residual_histories = {}
    final_mses = {}
    objective_histories = {}
    trained_models = {}
    method_labels = {
        'mono': 'monolithic (DC)',
        'admm': 'ADMM consensus (DC)',
        'seq': 'sequential (torchdiffeq)',
    }


    # ---------------- 1. ADMM consensus collocation model (DC) ----------------
    if run_admm_baseline:
        # softer tolerances / fewer IPOPT iterations for the first `num_early_iterations` ADMM
        # iterations (subproblems are far from consensus early on -> tight solves are wasted)
        admm_early_solver_options = {
            "max_iter": 200,
            "tol": 1e-4,
            "acceptable_tol": 1e-3,
            "acceptable_iter": 10
        }
        # standard tolerances for the remaining ADMM iterations
        admm_late_solver_options = {
            "max_iter": 1000,
            "tol": 1e-6,
            "acceptable_tol": 1e-5,
            "acceptable_iter": 15
        }
        admm_config = dict(
            Y_obs=Y_obs,                   
            layer_sizes=nn_layer_sizes,
            end_time=end_time,
            obs_times=train_grid,               
            state_lower_bound=-5,
            state_upper_bound=50,
            param_lower_bound=-10,
            param_upper_bound=10,
            l2_reg_param=1e-4,
            use_pool=True,
            warm_init=True,
            adaptive_penalty_update=(adaptive_rho and not fast_admm), # fast ADMM forces fixed ρ; else honour adaptive_rho
            mu=mu,                                                    # adaptive-ρ residual-balancing factor μ
            adaptive_tol_update=True,
            transcription_method='dc',
            irrdc_param=None,
            num_res_eval_nodes=None,
            relaxation_alpha=relaxation_alpha,           # under-relaxation weight α
            relaxation_start_iter=relaxation_start_iter, # iter under-relaxation activates
            warmstart_start_iter=warmstart_start_iter    # iter subproblem warm start activates
        )

        admm_training_options = dict(
            early_solver_options=admm_early_solver_options,
            late_solver_options=admm_late_solver_options,
            admm_abs_tol=5e-4,   # ε_abs (per-parameter absolute floor)
            admm_rel_tol=1e-3,   # ε_rel (Boyd-recommended 1e-3..1e-4)
            max_admm_iterations=2000,
            num_early_iterations=20, # opening iters using early_solver_options
            num_convergence_confirmations=2, # consecutive sub-tol iters before stopping (Boyd: 1)
            near_stopping_band=1.5,  # early stop when residuals sit within this x eps
            training_clock_budget=60, # minutes; None = no wall-clock cap
            consensus_aggregator=consensus_aggregator,
            subproblem_warmstart=subproblem_warmstart,
            mse_eval_options=mse_eval_options,
            mse_eval_every=10 # evaluate/record prediction MSE every N ADMM iters (raise to cut eval wall-clock)
        )
        if hyperparam_printer is not None:
            hyperparam_printer(admm_config, admm_training_options, fast_admm, restart_eta, accel_start_iter)
        admm = ADMM(**admm_config)
        if fast_admm:
            # Fast ADMM keeps the chosen consensus_aggregator (mean or under-relaxed) and adds the restart params
            fast_admm_training_options = dict(admm_training_options)
            fast_admm_training_options["restart_eta"] = restart_eta
            fast_admm_training_options["accel_start_iter"] = accel_start_iter
            print("\n\nTraining Fast ADMM consensus model (DC, Goldstein 2014 Alg 8) ...\n\n")
            admm.run_fast_admm_training(**fast_admm_training_options)
        else:
            print("\n\nTraining ADMM consensus model (DC) ...\n\n")
            admm.run_admm_training(**admm_training_options)
        train_runtimes['admm'] = admm.train_runtime
        mse_histories['admm'] = admm.mse_history
        residual_histories['admm'] = admm.residual_history
        objective_histories['admm'] = admm.objective_history
        # Final MSEs from the definitive final eval (mean + per-test-trajectory range).
        final_mses['admm'] = {
            "train": admm.final_train_mse,
            "test": admm.final_test_mse,
            "test_std": admm.final_test_mse_std,
            "test_min": admm.final_test_mse_range[0],
            "test_max": admm.final_test_mse_range[1],
        }
        if return_models:
            trained_models['admm'] = admm
        print(f"{method_labels['admm']} training time: {train_runtimes['admm']:.1f}s")


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
            time_budget_min=60, # minutes; None = no wall-clock cap
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
        # seq baseline records only mean MSEs (no per-trajectory range); keep schema uniform.
        final_mses['seq'] = {
            "train": mse_histories['seq']['train_mse'][-1],
            "test": mse_histories['seq']['test_mse'][-1],
            "test_std": None,
            "test_min": None,
            "test_max": None,
        }
        print(f"{method_labels['seq']} training time: {train_runtimes['seq']:.1f}s")

    # ---------------- 3. Monolithic collocation model (block-diagonal DC) ----------------
    if run_monolithic_baseline:
        monolithic_solver_options = {
            "max_iter": 1000,
            "tol": 1e-8,
            "acceptable_tol": 1e-7,
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
        mono_test_mse, mono_test_std, mono_test_min, mono_test_max = mono.compute_prediction_mse(
            test_grid, Y_test_obs, return_range=True,
            rtol=mse_eval_options['rtol'], atol=mse_eval_options['atol'], max_step=mse_eval_options['max_step']
        )
        final_mses['mono'] = {
            "train": mono_train_mse,
            "test": mono_test_mse,
            "test_std": mono_test_std,
            "test_min": mono_test_min,
            "test_max": mono_test_max,
        }
        if return_models:
            trained_models['mono'] = mono
        print(f"{method_labels['mono']} training time: {train_runtimes['mono']:.1f}s")


    # ---------------- TRAINING SUMMARY TABLE ----------------
    n_pred = len(train_batch_ids) + len(test_batch_ids)
    print("\n" + "=" * 92)
    print(f"Training summary (num_train={num_train_trajectories}, num_test={num_test_trajectories}, {n_pred} trajectories)")
    print("Final MSE = model MSE after convergence or wall-clock budget elapsed")
    print("-" * 92)
    print(f"{'Method':<32}{'Train (s)':>12}{'Final train MSE':>22}{'Final test MSE':>22}")
    for m in ('mono', 'admm', 'seq'):
        if m in train_runtimes:
            mses = final_mses.get(m, {})
            train_mse = mses.get('train', float('nan'))
            test_mse = mses.get('test', float('nan'))
            print(f"{method_labels[m]:<32}{train_runtimes[m]:>12.2f}{train_mse:>22.6e}{test_mse:>22.6e}")
    print("=" * 92 + "\n")

    result = {
        "train_batch_ids": train_batch_ids,
        "test_batch_ids": test_batch_ids,
        "train_runtimes": train_runtimes,
        "mse_histories": mse_histories,
        "residual_histories": residual_histories,
        "final_mses": final_mses,
        "objective_histories": objective_histories,
    }
    
    if return_models:
        # fitted model objects + the arrays needed to plot predicted vs true trajectories
        result["trained_models"] = trained_models
        result["test_grid"] = test_grid
        result["Y_test_obs"] = Y_test_obs
        result["train_grid"] = train_grid
        result["Y_obs"] = Y_obs

    return result

