"""
Parent method for running batched NODE training on a set of multiple, independent
trajectories, with three interchangeable training algorithms:

-- ADMM-based consensus collocation model (Pyomo/IPOPT, DC)        -> method key 'admm'
-- sequential Neural ODE baseline (PyTorch / torchdiffeq)          -> method key 'seq'
-- monolithic collocation model (Pyomo/IPOPT, block-diagonal DC)   -> method key 'mono'

run_training() trains each enabled model on a single pre-built train/test split (passed in as arrays)
and returns the per-model training metrics.
"""
import time
import numpy as np
import torch

from src.models.simultaneous.admm_mp import ADMM
from src.models.simultaneous.node_pyomo_monolithic import NODEPyomo as MonolithicNODEPyomo
from src.models.sequential.node_torch_multitraj import train_sequential_multitraj

def run_training(
    Y_obs: np.ndarray, # training trajectories, (B, N, D)
    train_grid: np.ndarray, # training time grid, (N,)
    Y_test_obs: np.ndarray, # test trajectories, (num_test, N, D)
    test_grid: np.ndarray, # test time grid, (N,)
    nn_layer_sizes: list,
    model_init_seed: int, # per-(split_seed, repeat) seed; shared by mono/admm/seq for identical init
    run_admm_baseline: bool = True,
    consensus_aggregator="under-relaxed", # "mean" | "under-relaxed"
    relaxation_alpha: float = 0.5, # relaxation weight α (θ̂ = α·θ + (1-α)·z)
    relaxation_start_iter: int = 25, # iter after which under-relaxed aggregation activates
    subproblem_warmstart="consensus", # "consensus" | "relaxed" | "previous"
    fast_admm: bool = False, # run Goldstein Fast ADMM instead of standard Boyd ADMM
    fast_admm_restart_eta: float = None, # Fast ADMM restart threshold η
    fast_admm_start_iter: int = None, # Fast ADMM: iter at which momentum turns on
    warmstart_start_iter: int = 25, # iter after which subproblem warm start activates
    adaptive_admm_penalty: bool = False, # adaptive ρ via Boyd residual balancing; fast_admm forces fixed ρ
    num_batches_per_subp: int = 1, # num trajectories stitched block-diagonally into each ADMM subproblem (K)
    admm_warm_init: bool = False, # ADMM: False -> cold per-worker Xavier init (seeded); True -> batch-0 warmup solve
    hyperparam_table=None,
    run_sequential_baseline: bool = False,
    run_monolithic_baseline: bool = True,
    return_models: bool = True
) -> dict:
    """
    Train the models on one pre-built train/test split and return their metrics.

    Args:
        Y_obs, train_grid: training trajectories (B, N, D) and their shared time grid (N,).
        Y_test_obs, test_grid: held-out test trajectories (num_test, N, D) and their time grid (N,).
        model_init_seed: seeds every model's weight init so mono/admm/seq start from the same point.

    Returns:
        dict {
          'train_runtimes': {method: wall_clock_seconds},
          'mse_histories':  {method: {train_mse: [...], test_mse: [...], ...}},
          'residual_histories': {admm: {primal_avg: [...], dual: [...], rho: [...], ...}},
          'objective_histories': {admm: {obj_per_batch: [...], ...}},
          'final_mses': {method: {train, test, test_std, test_min, test_max}},  # std/range = per-test-traj spread
          # when return_models=True:
          'trained_models': {method: fitted model object},
          'test_grid': [...], 'Y_test_obs': ndarray, 'train_grid': [...], 'Y_obs': ndarray,
        }
    """
    num_train_trajectories = Y_obs.shape[0]
    num_test_trajectories = Y_test_obs.shape[0]
    print(f"Using layer sizes: {nn_layer_sizes} (num_train_trajs={num_train_trajectories}, num_test_trajs={num_test_trajectories})")
    end_time = train_grid[-1]
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


    # ---------------------------- 1. ADMM consensus collocation model (DC) -------------------------------
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
            admm_warm_init=admm_warm_init,
            seed=model_init_seed, # master seed -> per-worker cold-init child seeds
            adaptive_admm_penalty=(adaptive_admm_penalty and not fast_admm), # fast ADMM forces fixed ρ; else honour adaptive_admm_penalty
            adaptive_admm_penalty_mu=3.0, # adaptive-ρ residual balancing factor 
            num_batches_per_subp=num_batches_per_subp, # trajectories per ADMM subproblem (K)
            adaptive_admm_tol=True,
            transcription_method='dc',
            irrdc_param=None,
            num_res_eval_nodes=None,
            relaxation_alpha=relaxation_alpha, # relaxation weight α
            relaxation_start_iter=relaxation_start_iter, # iter relaxation activates
            warmstart_start_iter=warmstart_start_iter # iter subproblem warm start activates
        )
        admm_training_options = dict(
            early_solver_options=admm_early_solver_options,
            late_solver_options=admm_late_solver_options,
            admm_abs_tol=5e-4, # ε_abs (per-parameter absolute floor)
            admm_rel_tol=1e-3, # ε_rel (Boyd-recommended 1e-3..1e-4)
            max_admm_iterations=2000,
            num_early_iterations=20, # opening iters using early_solver_options
            num_convergence_confirmations=2, # consecutive sub-tol iters before stopping (Boyd: 1)
            near_stopping_band=1.5, # early stop when residuals sit within this x eps
            training_clock_budget=60, # minutes; None = no wall-clock cap
            consensus_aggregator=consensus_aggregator,
            subproblem_warmstart=subproblem_warmstart,
            mse_eval_options=mse_eval_options,
            mse_eval_every=10 # evaluate/record prediction MSE every N ADMM iters (raise to cut eval wall-clock)
        )
        if hyperparam_table is not None:
            hyperparam_table(admm_config, admm_training_options, fast_admm, fast_admm_restart_eta, fast_admm_start_iter)
        admm = ADMM(**admm_config)
        if fast_admm:
            # Fast ADMM keeps the chosen consensus_aggregator (mean or under-relaxed) and adds the restart params
            fast_admm_training_options = dict(admm_training_options)
            # only override the solver defaults when explicitly provided (avoid passing None)
            if fast_admm_restart_eta is not None:
                fast_admm_training_options["fast_admm_restart_eta"] = fast_admm_restart_eta
            if fast_admm_start_iter is not None:
                fast_admm_training_options["fast_admm_start_iter"] = fast_admm_start_iter
            print("\n\nTraining Fast ADMM consensus model (DC, Goldstein 2014 Alg 8) ...\n\n")
            admm.run_fast_admm_training(**fast_admm_training_options)
        else:
            print("\n\nTraining ADMM consensus model (DC) ...\n\n")
            admm.run_admm_training(**admm_training_options)
        train_runtimes['admm'] = admm.train_runtime
        mse_histories['admm'] = admm.mse_history
        residual_histories['admm'] = admm.residual_history
        objective_histories['admm'] = admm.objective_history
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


    # ---------------------------- 2. Sequential Neural ODE baseline (PyTorch) -------------------------------
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
            seed=model_init_seed,
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


    # ---------------------------- 3. Monolithic collocation model (block-diagonal DC) ----------------
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
            seed=model_init_seed
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


    # ------------------------------------- TRAINING OUTPUT SUMMARY TABLE -----------------------------------
    n_pred = num_train_trajectories + num_test_trajectories
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
