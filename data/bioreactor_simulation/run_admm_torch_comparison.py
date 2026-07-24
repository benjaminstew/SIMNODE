"""
Compare two Neural-ODE training paradigms on the synthetic bioreactor dataset
under a common wall-clock budget and matched settings:

  * ADMM consensus model (simultaneous collocation, transcription 'dc'), Pyomo/IPOPT
  * PyTorch multi-trajectory model (sequential solver-in-the-loop mini-batch Adam)

Both use the same network size and L2 regularisation, and are evaluated for
inference by a standard forward ODE integration at identical (model-default)
tolerances. Produces test-set predicted-vs-observed trajectory plots and
train/test MSE-vs-wall-clock-time convergence plots.

Run from repo root with 'python -m data.bioreactor_simulation.run_admm_torch_comparison'
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.models.simultaneous.admm_mp import ADMM
from src.models.sequential.node_torch_multitraj import train_sequential_multitraj

def main():
    # ------------------------------- CONFIG -------------------------------
    data_path = "./data/bioreactor_simulation/noise_level_01_eps0.01.csv"
    state_cols = ['G', 'O', 'X', 'Xd', 'P', 'L', 'CO2']
    state_dim = len(state_cols)

    layer_sizes = [state_dim, 32, state_dim]   # same NN for both models
    L2_REG = 1e-4                                  # same L2 for both models
    TIME_BUDGET_MIN = 5.0                         # sequential wall-clock training budget

    # torch multi-trajectory (sequential mini-batch Adam) settings
    MINI_BATCH_SIZE = 10
    LEARNING_RATE = 1e-3
    PRETRAIN_ITERATIONS = 0  # raise (e.g. 1000) to warm-start the torch model
    NUM_EPOCHS = 100000      # large; the 30-min budget is what stops training

    # ADMM settings
    MAX_ADMM_ITERATIONS = 500      # large; the budget/convergence is what stops training

    # identical evaluation-solver tolerances for BOTH models (the model defaults);
    # used for inference and for the recorded MSE convergence histories
    EVAL_RTOL = 1e-3
    EVAL_ATOL = 1e-6

    NUM_TRAIN = 45
    NUM_TEST = 5
    SEED = 0

    print(f"Using layer sizes: {layer_sizes}")

    # -------------------------- DATA LOADING & SPLIT --------------------------
    np.random.seed(SEED)
    all_batch_ids = np.arange(0, 50)
    train_batch_ids = sorted(np.random.choice(all_batch_ids, size=NUM_TRAIN, replace=False).tolist())
    remaining_batch_ids = np.setdiff1d(all_batch_ids, train_batch_ids)
    test_batch_ids = sorted(np.random.choice(remaining_batch_ids, size=NUM_TEST, replace=False).tolist())

    df = pd.read_csv(data_path)
    train_batches = {
        batch_id: df[df['Batch'] == f"Batch_{batch_id}"].reset_index(drop=True)
        for batch_id in train_batch_ids
    }
    test_batches = {
        batch_id: df[df['Batch'] == f"Batch_{batch_id}"].reset_index(drop=True)
        for batch_id in test_batch_ids
    }
    train_grid = train_batches[train_batch_ids[0]]['Time'].values
    Y_obs = np.array(
        [train_batches[batch_id][state_cols].values for batch_id in train_batch_ids]
    )  # (45, N, 7)
    test_grid = test_batches[test_batch_ids[0]]['Time'].values
    Y_test_obs = np.array(
        [test_batches[batch_id][state_cols].values for batch_id in test_batch_ids]
    )  # (5, N, 7)
    end_time = train_grid[-1]

    # float32 torch tensors for the torch model
    Y_obs_t = torch.tensor(Y_obs, dtype=torch.float32)
    train_grid_t = torch.tensor(train_grid, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test_obs, dtype=torch.float32)
    test_grid_t = torch.tensor(test_grid, dtype=torch.float32)

    # plotting identity for the two methods
    admm_color = (160/255, 190/255, 200/255)
    torch_color = (150/255, 200/255, 150/255)
    methods = ('admm', 'torch')
    method_colors = {'admm': admm_color, 'torch': torch_color}
    method_labels = {
        'admm': 'ADMM consensus (DC)',
        'torch': 'sequential mini-batch Adam',
    }

    mse_histories = {}
    predicted_test = {}
    train_runtimes = {}

   # ============================ SECTION 1 -- PyTorch multi-trajectory model (sequential mini-batch Adam) ================================

    print("\n=== Training PyTorch multi-trajectory model (mini-batch Adam) ===")
    t0 = time.perf_counter()
    model, mse_histories['torch'] = train_sequential_multitraj(
        Y_obs=Y_obs_t,
        t_grid=train_grid_t,
        layer_sizes=layer_sizes,
        num_epochs=NUM_EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        lambda_reg=L2_REG,
        learning_rate=LEARNING_RATE,
        pretrain_iterations=PRETRAIN_ITERATIONS,
        Y_obs_test=Y_test_t,
        test_grid=test_grid_t,
        record_mse_every=1,
        time_budget_min=TIME_BUDGET_MIN,
        eval_rtol=EVAL_RTOL,
        eval_atol=EVAL_ATOL,
        seed=SEED,
        verbose=True,
        log_every=1000
    )
    train_runtimes['torch'] = time.perf_counter() - t0
    print(f"torch training time: {train_runtimes['torch']:.1f}s")

    # predict all test trajectories in one batched solve: (5, D) -> (5, N, D)
    torch_test_pred = model.get_predicted_trajectory(
        Y_test_t[:, 0, :], test_grid_t, rtol=EVAL_RTOL, atol=EVAL_ATOL
    ).cpu().numpy()
    predicted_test['torch'] = {
        batch_id: torch_test_pred[i]
        for i, batch_id in enumerate(test_batch_ids)
    }

    # ============================ SECTION 2 -- ADMM consensus model (simultaneous collocation, DC) ===============================

    ipopt_options = {
        "max_iter": 300,
        "nlp_scaling_method": "gradient-based",
        "mu_strategy": "adaptive",
        "tol": 1e-5,
        "acceptable_tol": 1e-4,
        "acceptable_iter": 5,
    }

    # record train/test MSE of the consensus model at the SAME eval tolerances as torch
    mse_eval_options = {
        "train_grid": train_grid,
        "Y_obs": Y_obs,
        "test_grid": test_grid,
        "Y_test_obs": Y_test_obs,
        "rtol": EVAL_RTOL,
        "atol": EVAL_ATOL,
        "max_step": np.inf,
    }

    dc_config = dict(
        Y_obs=Y_obs,
        layer_sizes=layer_sizes,
        end_time=end_time,
        state_lower_bound=-2,
        state_upper_bound=50,
        param_lower_bound=-50,
        param_upper_bound=50,
        l2_reg_param=L2_REG,
        admm_penalty_param=10,
        use_pool=True,
        transcription_method='dc',
        max_admm_iterations=MAX_ADMM_ITERATIONS,
        admm_primal_residual_tol=1,
        admm_train_mse_tol=0.5,
        obs_times=train_grid
    )

    print("\n=== Training ADMM consensus model (DC) ===")
    print(f"({NUM_TRAIN} training batches -> {NUM_TRAIN} IPOPT subproblems per ADMM "
          f"iteration; use_pool spawns ~{NUM_TRAIN} workers)")
    admm = ADMM(**dc_config)
    t0 = time.perf_counter()
    admm.run_admm_training(
        ipopt_options, mse_eval_options=mse_eval_options
    )
    train_runtimes['admm'] = time.perf_counter() - t0
    print(f"ADMM training time: {train_runtimes['admm']:.1f}s")
    mse_histories['admm'] = admm.mse_history

    predicted_test['admm'] = {
        batch_id: admm.get_predicted_trajectory(
            Y_test_obs[i, 0, :], test_grid,
            method='RK45', rtol=EVAL_RTOL, atol=EVAL_ATOL, max_step=np.inf
        )
        for i, batch_id in enumerate(test_batch_ids)
    }

    # =========================== COMPARISON PLOTS ==================================

    # ---- test-set predicted vs observed trajectories (both models overlaid) ----
    def plot_overlaid_trajectories(grid, Y_data, batch_ids, predicted, domain_name, out_name):
        fig, axes = plt.subplots(4, 2, figsize=(13, 11))
        axes = axes.flatten()
        for i, col in enumerate(state_cols):
            ax = axes[i]
            for j, batch_id in enumerate(batch_ids):
                ax.scatter(
                    grid, Y_data[j][:, i],
                    color='lightgrey', marker='o', s=22, alpha=0.6, zorder=5,
                    label='Observed' if j == 0 else None
                )
            for method in methods:
                c = method_colors[method]
                for j, batch_id in enumerate(batch_ids):
                    ax.plot(
                        grid, predicted[method][batch_id][:, i],
                        color=c, linewidth=1.6, linestyle='-',
                        label=f'Predicted ({method_labels[method]})' if j == 0 else None
                    )
            ax.set_title(col, fontsize=13)
            ax.set_xlabel('Time (h)')
            ax.set_ylabel('Concentration (g/L)')
            ax.grid(True, linestyle='--', color='lightgrey', alpha=0.8)
        axes[7].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=10, loc='lower right', bbox_to_anchor=(0.98, 0.04))
        plt.suptitle(f'Predicted vs Observed Trajectories ({domain_name}; ADMM vs torch)', fontsize=14)
        plt.tight_layout()
        fig.savefig(Path(__file__).with_name(out_name), dpi=200, bbox_inches="tight")
        plt.show()

    plot_overlaid_trajectories(test_grid, Y_test_obs, test_batch_ids, predicted_test,
                               'Test Batches', 'admm_vs_torch_test_trajectories.png')

    # ---- interpolate each run's MSE-vs-time curve onto a common time grid ----
    def interp_history(split, num_points=200):
        """Interpolate both models' {split}_mse (and std) onto a shared time grid."""
        per_method = {}
        t_maxes = []
        for method in methods:
            h = mse_histories[method]
            times = np.asarray(h["time_min"], dtype=float)
            mse = np.asarray(h[f"{split}_mse"], dtype=float)
            std = np.asarray(h[f"{split}_mse_std"], dtype=float)
            good = np.isfinite(times) & np.isfinite(mse)
            per_method[method] = (times[good], mse[good], std[good])
            if good.sum() >= 1:
                t_maxes.append(times[good].max())
        # common grid up to the shortest run so neither curve is extrapolated
        t_max = min(t_maxes) if t_maxes else 0.0
        t_common = np.linspace(0.0, t_max, num_points)
        interp = {}
        for method in methods:
            times, mse, std = per_method[method]
            if times.size >= 2:
                mse_i = np.interp(t_common, times, mse)
                std_i = np.interp(t_common, times, np.nan_to_num(std))
            else:
                mse_i = np.full_like(t_common, np.nan)
                std_i = np.full_like(t_common, np.nan)
            interp[method] = (mse_i, std_i)
        return t_common, interp

    # ---- train/test MSE vs wall-clock convergence (interpolated, both models) ----
    for split in ('train', 'test'):
        t_common, interp = interp_history(split)
        fig, ax = plt.subplots(figsize=(7, 5))
        finite_means = []
        for method in methods:
            mse_i, std_i = interp[method]
            c = method_colors[method]
            ax.plot(t_common, mse_i, color=c, marker='o', markersize=3, linewidth=1.4,
                    label=f'{method_labels[method]} mean MSE')
            lower = np.clip(mse_i - std_i, 1e-12, None)
            ax.fill_between(t_common, lower, mse_i + std_i, color=c, alpha=0.2,
                            label=f'{method_labels[method]} ±1 std')
            finite_means.append(mse_i[np.isfinite(mse_i) & (mse_i > 0)])
        ax.set_yscale('log')
        finite = np.concatenate(finite_means) if finite_means else np.array([])
        if finite.size > 0:
            ax.set_ylim(finite.min() / 3.0, finite.max() * 3.0)
        ax.set_xlabel('Training time (min)')
        ax.set_ylabel(f'{split.capitalize()}-set MSE')
        ax.grid(False)
        ax.legend()
        plt.tight_layout()
        out = Path(__file__).with_name(f"admm_vs_torch_{split}_convergence.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.show()

    # ---- runtime summary ----
    print("\n" + "=" * 60)
    print(f"{'Method':<32}{'Train (s)':>12}{'Budget (min)':>16}")
    print("-" * 60)
    for method in methods:
        print(f"{method_labels[method]:<32}{train_runtimes[method]:>12.1f}{TIME_BUDGET_MIN:>16.1f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
