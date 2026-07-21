"""
Run ADMM training for the synthetic bioreactor dataset with multiprocessing.
Run from repo root with 'python -m data.bioreactor_simulation.run_admm_mp'
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.simultaneous.admm_mp import ADMM
from src.models.simultaneous.node_pyomo_inference import CollocationInference
from src.models.sequential.node_torch_base import train_sequential_multitraj

def main():
    # import training data
    data_path = "./data/bioreactor_simulation/noise_level_01_eps0.01.csv"
    state_cols = ['G', 'O', 'X', 'Xd', 'P', 'L', 'CO2']
    state_dim = len(state_cols)

    layer_sizes = [state_dim, 18, 18, state_dim]
    print(f"Using layer sizes: {layer_sizes}")

    # inference engine for evaluating the consensus model:
    # 'sequential'  -> forward IVP solve (scipy solve_ivp)
    # 'collocation' -> collocation NLP with theta* fixed (RK45 warm-start, IPOPT)
    inference_method = 'sequential'

    # sequential (solver-in-the-loop) Neural ODE baseline settings (torchdiffeq)
    # set run_sequential_baseline = False for a DC/IRR-DC-only comparison
    run_sequential_baseline = True
    seq_num_iterations = 30000

    # split train and test batch ids
    all_batch_ids = np.arange(0, 50)
    train_batch_ids = np.random.choice(all_batch_ids, size=10, replace=False).tolist()
    remaining_batch_ids = np.setdiff1d(all_batch_ids, train_batch_ids)
    test_batch_ids = sorted(np.random.choice(remaining_batch_ids, size=2, replace=False).tolist())

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
    )
    test_grid = test_batches[test_batch_ids[0]]['Time'].values
    Y_test_obs = np.array(
        [test_batches[batch_id][state_cols].values for batch_id in test_batch_ids]
    )
    end_time = train_grid[-1]
    #print(f"Time grid shape: {train_grid.shape}")
    #print(f"Y_obs shape: {Y_obs.shape}")
    #print(f"Test grid shape: {test_grid.shape}")
    #print(f"Y_test_obs shape: {Y_test_obs.shape}")

    ipopt_options = {
        "max_iter": 300,
        "print_level": 0,
        "nlp_scaling_method": "gradient-based",
        "mu_strategy": "adaptive",
        "tol": 1e-5,
        "acceptable_tol": 1e-4,
        "acceptable_iter": 5
    }

    # record train/test MSE of the consensus model for the wall-clock convergence plots 
    dt = train_grid[1] - train_grid[0]
    dt_test = test_grid[1] - test_grid[0]
    mse_eval_options = {
        "train_grid": train_grid,
        "Y_obs": Y_obs,
        "test_grid": test_grid,
        "Y_test_obs": Y_test_obs,
        "rtol": 1e-7,
        "atol": 1e-9,
        "max_step": dt
    }

    # shared ADMM constructor args 
    shared_config = dict(
        Y_obs=Y_obs,
        layer_sizes=layer_sizes,
        end_time=end_time,
        state_lower_bound=-2,
        state_upper_bound=35,
        param_lower_bound=-50,
        param_upper_bound=50,
        l2_reg_param=1e-4,
        admm_penalty_param=10,
        use_pool=True
    )

    # per-method configs 
    dc_config = dict(
        **shared_config,
        transcription_method='dc',
        max_admm_iterations=100,
        admm_primal_residual_tol=0.5,
        admm_train_mse_tol=0.15
    )
    irrdc_config = dict(
        **shared_config,
        transcription_method='irrdc',
        max_admm_iterations=100,
        admm_primal_residual_tol=0.5,
        admm_train_mse_tol=0.15,
        residual_reg_param=0.1,
        num_res_eval_nodes=200
    )

    # colour each method (prediction lines + its observed markers share the shade)
    dc_color = (160/255, 190/255, 200/255)
    irrdc_color = (245/255, 150/255, 150/255)
    seq_color = (150/255, 200/255, 150/255)
    method_colors = {'dc': dc_color, 'irrdc': irrdc_color, 'seq': seq_color}
    method_labels = {
        'dc': 'collocation (w/o residual penalty)',
        'irrdc': 'collocation (w/ residual penalty)',
        'seq': 'sequential (torchdiffeq)',
    }
    # methods to include in the comparison plots
    plot_methods = ('dc', 'irrdc') + (('seq',) if run_sequential_baseline else ())

    def predict_trajectory(admm, y_0, grid, max_step):
        '''Predict the consensus-model trajectory on `grid` via the selected inference engine.'''
        if inference_method == 'sequential':
            return admm.get_predicted_trajectory(y_0, grid, rtol=1e-7, atol=1e-9, max_step=max_step)
        elif inference_method == 'collocation':
            engine = CollocationInference(
                Ws=admm.consensus_Ws,
                bs=admm.consensus_bs,
                layer_sizes=admm.layer_sizes,
                state_dim=admm.state_dim,
                end_time=grid[-1],
                num_colloc_nodes=len(grid),
                state_lower_bound=admm.state_lower_bound,
                state_upper_bound=admm.state_upper_bound,
                time_invariant=True,
            )
            # feasibility solve (no test observations), RK45 warm start, interpolated to `grid`
            return engine.predict(y_0, ipopt_options, warm_start=True, output_grid=grid)
        else:
            raise ValueError(f"Unknown inference_method: {inference_method!r}")

    # run DC then IRR-DC one after another; each ADMM opens/closes its own mp pool
    predicted_train = {} # {batch_id: predicted train trajectory}
    predicted_test = {} # {batch_id: predicted test trajectory}
    mse_histories = {}
    train_runtimes = {} # {method: training wall-clock seconds}
    infer_runtimes = {} # {method: total inference wall-clock seconds (all train+test trajectories)}

    for config in (dc_config, irrdc_config):
        admm = ADMM(**config)
        t0 = time.perf_counter()
        admm.run_admm_training(ipopt_options, mse_eval_options=mse_eval_options)
        method = admm.transcription_method
        train_runtimes[method] = time.perf_counter() - t0
        print(f"{method} ADMM training time: {train_runtimes[method]:.1f}s")

        mse_histories[method] = admm.mse_history
        t_inf = time.perf_counter()
        predicted_train[method] = {
            batch_id: predict_trajectory(
                admm, Y_obs[train_batch_ids.index(batch_id), 0, :], train_grid, dt
            )
            for batch_id in train_batch_ids
        }
        predicted_test[method] = {
            batch_id: predict_trajectory(
                admm, Y_test_obs[test_batch_ids.index(batch_id), 0, :], test_grid, dt_test
            )
            for batch_id in test_batch_ids
        }
        infer_runtimes[method] = time.perf_counter() - t_inf

    # ---- sequential (solver-in-the-loop) Neural ODE baseline on the same batches ----
    # One model trained jointly on all training batches (analogous to the ADMM consensus),
    # evaluated with its native sequential IVP solver. MSE history matches the ADMM format.
    if run_sequential_baseline:
        print("Training sequential Neural ODE baseline ...")
        t0 = time.perf_counter()
        seq_predict, mse_histories['seq'] = train_sequential_multitraj(
            Y_obs_train=Y_obs,
            train_grid=train_grid,
            layer_sizes=layer_sizes,
            num_iterations=seq_num_iterations,
            Y_obs_test=Y_test_obs,
            test_grid=test_grid,
            lambda_reg=shared_config['l2_reg_param'],
            learning_rate=1e-4,
            rtol=1e-5, atol=1e-6,
            record_every=10, verbose=True, log_every=50
        )
        train_runtimes['seq'] = time.perf_counter() - t0
        print(f"sequential training time: {train_runtimes['seq']:.1f}s")
        t_inf = time.perf_counter()
        predicted_train['seq'] = {
            batch_id: seq_predict(Y_obs[train_batch_ids.index(batch_id), 0, :], train_grid)
            for batch_id in train_batch_ids
        }
        predicted_test['seq'] = {
            batch_id: seq_predict(Y_test_obs[test_batch_ids.index(batch_id), 0, :], test_grid)
            for batch_id in test_batch_ids
        }
        infer_runtimes['seq'] = time.perf_counter() - t_inf

    # ---- runtime summary table (training + inference) ----
    # Note: DC/IRR-DC inference uses the selected `inference_method` engine for the
    # consensus model; the sequential baseline is evaluated with its own torchdiffeq solver.
    n_pred = len(train_batch_ids) + len(test_batch_ids)
    print("\n" + "=" * 80)
    print(f"Runtime summary  (DC/IRR-DC inference: '{inference_method}'; "
          f"sequential inference: torchdiffeq;  {n_pred} trajectories/method)")
    print("=" * 80)
    print(f"{'Method':<36}{'Train (s)':>12}{'Infer total (s)':>18}{'Infer/traj (ms)':>16}")
    print("-" * 80)
    for method in plot_methods:
        if method not in train_runtimes:
            continue
        tr = train_runtimes[method]
        inf = infer_runtimes.get(method, float('nan'))
        per_traj_ms = (inf / n_pred * 1e3) if n_pred else float('nan')
        print(f"{method_labels[method]:<36}{tr:>12.2f}{inf:>18.3f}{per_traj_ms:>16.2f}")
    print("=" * 80 + "\n")

    # ---- overlaid trajectory figure: collocation (DC, IRR-DC) vs sequential over data ----
    def plot_overlaid_trajectories(grid, Y_data, batch_ids, predicted, domain_name, out_name):
        fig, axes = plt.subplots(4, 2, figsize=(13, 11))
        axes = axes.flatten()
        for i, col in enumerate(state_cols):
            ax = axes[i]
            # observed data (true trajectory), plotted once per batch
            for j, batch_id in enumerate(batch_ids):
                batch_idx = batch_ids.index(batch_id)
                ax.scatter(
                    grid, Y_data[batch_idx][:, i],
                    color='lightgrey', marker='o', s=22, alpha=0.6, zorder=5,
                    label='Observed' if j == 0 else None
                )
            # predicted trajectories, coloured by method
            for method in plot_methods:
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
        plt.suptitle(f'Predicted vs Observed Trajectories ({domain_name} Batches; DC, IRR-DC, Sequential)', fontsize=14)
        plt.tight_layout()
        fig.savefig(Path(__file__).with_name(out_name), dpi=200, bbox_inches="tight")
        plt.show()

    plot_overlaid_trajectories(train_grid, Y_obs, train_batch_ids, predicted_train,
                               'Training', 'dc_vs_irrdc_run1_train.png')
    plot_overlaid_trajectories(test_grid, Y_test_obs, test_batch_ids, predicted_test,
                               'Test', 'dc_vs_irrdc_run1_test.png')

    # ---- overlaid wall-clock convergence plots (DC, IRR-DC, Sequential), train and test ----
    for split in ('train', 'test'):
        fig, ax = plt.subplots(figsize=(7, 5))
        finite_means = []
        for method in plot_methods:
            history = mse_histories[method]
            times = np.asarray(history["time_min"])
            mse = np.asarray(history[f"{split}_mse"])
            std = np.asarray(history[f"{split}_mse_std"])
            c = method_colors[method]
            ax.plot(times, mse, color=c, marker='o', markersize=3, linewidth=1.4,
                    label=f'{method_labels[method]} mean MSE')
            # translucent ±1 std band (clipped at a small positive floor for the log axis)
            lower = np.clip(mse - std, 1e-12, None)
            ax.fill_between(times, lower, mse + std, color=c, alpha=0.2,
                            label=f'{method_labels[method]} ±1 std')
            finite_means.append(mse[np.isfinite(mse) & (mse > 0)])
        ax.set_yscale('log')
        # Pin y-limits to the mean curves (with multiplicative padding) so outlier
        # std excursions are clipped rather than stretching the axis and flattening
        # the detail of the mean MSE lines.
        finite = np.concatenate(finite_means) if finite_means else np.array([])
        if finite.size > 0:
            ax.set_ylim(finite.min() / 3.0, finite.max() * 3.0)
        ax.set_xlabel('Training time (min)')
        ax.set_ylabel(f'{split.capitalize()}-set MSE')
        ax.grid(False)
        ax.legend()
        plt.tight_layout()
        out = Path(__file__).with_name(f"comparison_run1_{split}_convergence.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
