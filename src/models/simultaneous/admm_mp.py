'''multiprocessing wrapper to organize NODEPyomo submodels for ADMM-based batched training.'''
import os
for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(thread_var, "1")

from src.models.simultaneous.node_pyomo_admm import NODEPyomo
import numpy as np
import time
import threading
import multiprocessing as mp
from collections import deque
from scipy.integrate import solve_ivp

# ---- MODULE LEVEL FUNCTIONS (required for mp.Pool pickling) ----
WORKER = None # child processes (independent memory) each have own WORKER instance
BARRIER = None # shared barrier enforcing a strict 1:1 task->worker mapping 

def init_worker(id_queue: mp.Queue, submodel_config: dict, barrier):
    global WORKER, BARRIER
    worker_id = id_queue.get() # = submodel index, per order of worker process creation
    WORKER = MPWorker(worker_id, submodel_config)
    BARRIER = barrier

def solve_worker(payload: dict):
    try:
        WORKER.broadcast_consensus_and_duals(
            payload["consensus_Ws"],
            payload["consensus_bs"],
            payload["dual_var_Ws"][WORKER.worker_id],
            payload["dual_var_bs"][WORKER.worker_id],
            payload["iteration_count"],
            payload["admm_penalty_param"]
        )
        WORKER.solve_submodel(payload["solver_options"])
        return (WORKER.worker_id, WORKER.submodel_Ws, WORKER.submodel_bs,
                WORKER.submodel_obj, WORKER.submodel_local_dc_obj)
    finally:
        # Barrier blocks each worker until all N have finished, forcing strict 1:1 batch->worker mapping 
        # (no worker starts a second task early). In finally so a failed solve still releases the barrier instead of
        # deadlocking; the exception still propagates via pool.map
        if BARRIER is not None:
            try:
                BARRIER.wait()
            except threading.BrokenBarrierError:
                pass


class MPWorker:
    '''
    Wraps a NODEPyomo submodel and handles all Pyomo subproblem <-> numpy ADMM communication.
    Used directly in the main process for sequential mode, or inside worker processes for
    parallel mode.
    '''
    def __init__(self, worker_id: int, submodel_config: dict):
        self.worker_id = worker_id
        self.layer_sizes = submodel_config['layer_sizes']

        self.submodel = NODEPyomo(
            Y_obs=submodel_config['Y_obs'][worker_id],
            layer_sizes=self.layer_sizes,
            end_time=submodel_config['end_time'],
            state_lower_bound=submodel_config['state_lower_bound'],
            state_upper_bound=submodel_config['state_upper_bound'],
            param_lower_bound=submodel_config['param_lower_bound'],
            param_upper_bound=submodel_config['param_upper_bound'],
            l2_reg_param=submodel_config['l2_reg_param'],
            admm_submodel=True,
            admm_penalty_param=submodel_config['admm_penalty_param'],
            transcription_method=submodel_config['transcription_method'],
            irrdc_param=submodel_config['irrdc_param'],
            num_res_eval_nodes=submodel_config['num_res_eval_nodes'],
            obs_times=submodel_config['obs_times']
        )

        self.submodel_obj = float("nan")  # final subproblem objective, refreshed each solve
        self.submodel_local_dc_obj = float("nan")  # final subproblem objective without ADMM penalty, refreshed each solve

        if submodel_config['warm_init']:
            self.submodel_Ws = submodel_config['warm_init_submodel_Ws'][worker_id] 
            self.submodel_bs = submodel_config['warm_init_submodel_bs'][worker_id] 
            # broadcast warm init submodel params, consensus params and dual vars from the b=0 warmup solve 
            self.broadcast_consensus_and_duals(
                submodel_config['warm_init_consensus_Ws'],
                submodel_config['warm_init_consensus_bs'],
                submodel_config['init_dual_var_Ws'][worker_id],
                submodel_config['init_dual_var_bs'][worker_id],
                iteration_count=0,
                admm_penalty_param=submodel_config['admm_penalty_param'],
                warm_init=submodel_config['warm_init'],
                warm_init_submodel_Ws=submodel_config['warm_init_submodel_Ws'][worker_id],
                warm_init_submodel_bs=submodel_config['warm_init_submodel_bs'][worker_id]
            )
        else:
            # set to Xavier init ws/zero init bs
            self.submodel_Ws = [None for _ in range(len(self.layer_sizes) - 1)]
            self.submodel_bs = [None for _ in range(len(self.layer_sizes) - 1)]
            for l in range(1, len(self.layer_sizes)):
                dict_Ws = self.submodel.model.Ws[l].W.extract_values()
                dict_bs = self.submodel.model.bs[l].b.extract_values()
                self.submodel_Ws[l-1] = self.dict_to_ndarray(dict_Ws, (self.layer_sizes[l-1], self.layer_sizes[l]))
                self.submodel_bs[l-1] = self.dict_to_ndarray(dict_bs, (1, self.layer_sizes[l]))
       
    def dict_to_ndarray(self, d: dict, shape: tuple):
        arr = np.zeros(shape, dtype=float)
        for (i, j), val in d.items():
            arr[i-1, j-1] = float(val)
        return arr

    def ndarray_to_dict(self, arr: np.ndarray):
        return {(i+1, j+1): float(val) for (i, j), val in np.ndenumerate(arr)}

    def solve_submodel(self, solver_options: dict):
        self.submodel.solve_model(solver_options)

        # extract submodel parameters after solve 
        for l in range(1, len(self.layer_sizes)):
            dict_Ws = self.submodel.model.Ws[l].W.extract_values()
            dict_bs = self.submodel.model.bs[l].b.extract_values()
            self.submodel_Ws[l-1] = self.dict_to_ndarray(dict_Ws, (self.layer_sizes[l-1], self.layer_sizes[l]))
            self.submodel_bs[l-1] = self.dict_to_ndarray(dict_bs, (1, self.layer_sizes[l]))

        # final objective from this ipopt solve (NaN if no usable solution was loaded)
        self.submodel_obj = self.submodel.last_solve_info["objective"]
        self.submodel_local_dc_obj = self.submodel.last_solve_info["local_dc_objective"]

    def broadcast_consensus_and_duals(self,
            consensus_Ws: list, consensus_bs: list,
            dual_var_Ws: list, dual_var_bs: list,
            iteration_count: int,
            admm_penalty_param: float,
            warm_init: bool = False,
            warm_init_submodel_Ws: list = None, warm_init_submodel_bs: list = None
    ):
        # broadcast the current penalty parameter rho into the submodel objective (mutable Param)
        self.submodel.model.admm_penalty_param.set_value(admm_penalty_param)

        for l in range(1, len(self.layer_sizes)):
            self.submodel.model.consensus_Ws[l].W.store_values(self.ndarray_to_dict(consensus_Ws[l-1]))
            self.submodel.model.consensus_bs[l].b.store_values(self.ndarray_to_dict(consensus_bs[l-1]))
            self.submodel.model.dual_var_Ws[l].W.store_values(self.ndarray_to_dict(dual_var_Ws[l-1]))
            self.submodel.model.dual_var_bs[l].b.store_values(self.ndarray_to_dict(dual_var_bs[l-1]))

            # warm start submodel from consensus params (to help convergence)
            if iteration_count > 50: 
                self.submodel.model.Ws[l].W.set_values(self.ndarray_to_dict(consensus_Ws[l-1]))
                self.submodel.model.bs[l].b.set_values(self.ndarray_to_dict(consensus_bs[l-1]))

            # warm init submodel from b=0 submodel params
            if warm_init:
                self.submodel.model.Ws[l].W.set_values(self.ndarray_to_dict(warm_init_submodel_Ws[l-1]))
                self.submodel.model.bs[l].b.set_values(self.ndarray_to_dict(warm_init_submodel_bs[l-1]))


class PoolManager:
    """
    Manages the multiprocessing pool lifecycle and parallel submodel execution.
    MPWorker instances live entirely inside the child processes; this class only
    opens/closes the pool and dispatches work via Pool.map.
    """
    def __init__(self, num_batches: int, submodel_config: dict):
        self.num_batches = num_batches
        self.pool = self.open_pool(submodel_config)

    def open_pool(self, submodel_config: dict):
        ctx = mp.get_context("spawn")
        id_queue = ctx.Queue()
        for i in range(self.num_batches):
            id_queue.put(i)
        # Barrier spanning all workers; used in solve_worker to force a 1:1
        # task->worker mapping each round (see solve_worker for rationale).
        barrier = ctx.Barrier(self.num_batches)
        pool = ctx.Pool(
            processes=self.num_batches,
            initializer=init_worker,
            initargs=(id_queue, submodel_config, barrier)
        )
        print(f"Opened multiprocessing pool with {self.num_batches} workers.")
        return pool

    def map(self, payload: dict) -> list[tuple]:
        '''Dispatch payload to all workers; return list of
        (worker_id, submodel_Ws, submodel_bs, submodel_obj, submodel_local_dc_obj).'''
        return self.pool.map(solve_worker, [payload] * self.num_batches, chunksize=1)

    def close(self):
        if self.pool is not None:
            try:
                self.pool.close()
                self.pool.join()
            except Exception:
                self.pool.terminate()
                self.pool.join()
            finally:
                self.pool = None
            print("Closed multiprocessing pool.")


class ADMM:
    """
    Main interface between ADMM consensus problem and Pyomo subproblems.
    Owns the consensus parameters, dual variables, and all ADMM algorithm logic.
    Set use_pool=False (default) to run sequentially for testing.
    """
    def __init__(self,
            Y_obs: np.ndarray, 
            layer_sizes: list, 
            end_time: float,
            state_lower_bound: float, state_upper_bound: float,
            param_lower_bound: float, param_upper_bound: float,
            l2_reg_param: float,
            adaptive_penalty_update: bool = True,
            fixed_admm_penalty_param: float = None, # if no adaptive update, use this fixed value for ρ
            use_pool: bool = False,
            warm_init: bool = True,
            transcription_method: str = 'dc',
            irrdc_param: float = None,
            num_res_eval_nodes: int = None,
            obs_times: np.ndarray = None,
            adaptive_tol_update: bool = True
    ):
        self.Y_obs = Y_obs # ndarray (num_batches, batch_size, state_dim)
        self.num_batches, self.batch_size, self.state_dim = self.Y_obs.shape
        self.layer_sizes = layer_sizes
        self.adaptive_tol_update = adaptive_tol_update
        self.num_params = sum(
            self.layer_sizes[l] * self.layer_sizes[l+1] + self.layer_sizes[l+1]
            for l in range(len(self.layer_sizes) - 1)
        )
        self.end_time = end_time
        self.state_lower_bound = state_lower_bound
        self.state_upper_bound = state_upper_bound
        self.param_lower_bound = param_lower_bound
        self.param_upper_bound = param_upper_bound
        self.l2_reg_param = l2_reg_param
        self.adaptive_penalty_update = adaptive_penalty_update
        if self.adaptive_penalty_update:
            self.admm_penalty_param = np.sqrt(self.num_batches/2)  # initial rho, adapted each ADMM iteration
        else:
            if fixed_admm_penalty_param is None:
                raise ValueError("Must set fixed_admm_penalty_param when adaptive_penalty_update=False.")
            self.admm_penalty_param = fixed_admm_penalty_param
        self.use_pool = use_pool
        self.transcription_method = transcription_method
        self.irrdc_param = irrdc_param
        self.num_res_eval_nodes = num_res_eval_nodes
        self.obs_times = obs_times  # submodels interpolate Y_obs onto the collocation grid if not already

        self.dual_var_Ws, self.dual_var_bs = self.zero_init_dual_vars()
        if warm_init:
            self.consensus_Ws, self.consensus_bs, self.submodels_Ws, self.submodels_bs = self.warm_init_params()
        else:
            self.consensus_Ws = [np.zeros((self.layer_sizes[l], self.layer_sizes[l+1])) for l in range(len(self.layer_sizes) - 1)]
            self.consensus_bs = [np.zeros((1, self.layer_sizes[l+1])) for l in range(len(self.layer_sizes) - 1)]
            self.submodels_Ws = [[None for _ in range(len(self.layer_sizes) - 1)] for _ in range(self.num_batches)]
            self.submodels_bs = [[None for _ in range(len(self.layer_sizes) - 1)] for _ in range(self.num_batches)]

        self.submodels_obj = [float("nan") for _ in range(self.num_batches)] # final subproblem objective, overwritten each solve
        self.submodels_local_dc_obj = [float("nan") for _ in range(self.num_batches)]  # final subproblem objective without ADMM penalty, overwritten each solve
        submodel_config = {
            'Y_obs': Y_obs,
            'layer_sizes': layer_sizes,
            'end_time': end_time,
            'state_lower_bound': state_lower_bound,
            'state_upper_bound': state_upper_bound,
            'param_lower_bound': param_lower_bound,
            'param_upper_bound': param_upper_bound,
            'l2_reg_param': l2_reg_param,
            'admm_penalty_param': self.admm_penalty_param, # initial rho (1.0 if adaptive, else fixed)
            'transcription_method': transcription_method,
            'irrdc_param': irrdc_param,
            'num_res_eval_nodes': num_res_eval_nodes,
            'obs_times': obs_times, 
            'init_dual_var_Ws': self.dual_var_Ws, 
            'init_dual_var_bs': self.dual_var_bs,
            'warm_init': warm_init, 
            'warm_init_consensus_Ws': self.consensus_Ws, 
            'warm_init_consensus_bs': self.consensus_bs, 
            'warm_init_submodel_Ws': self.submodels_Ws, 
            'warm_init_submodel_bs': self.submodels_bs
        }

        if use_pool:
            self.workers = None # MPWorker instances live in child processes
            self.pool_manager = PoolManager(self.num_batches, submodel_config)
        else:
            self.workers = [MPWorker(i, submodel_config) for i in range(self.num_batches)] 
            self.pool_manager = None

        self.mse_history = {
            "time_min": [],
            "train_mse": [], "train_mse_std": [],
            "test_mse": [], "test_mse_std": [],
        }

        # per-iteration ADMM residual / penalty tracking
        self.residual_history = {
            "primal_avg": [],
            "primal_per_batch": [], # list[np.ndarray]: per-subproblem primal residual each iter
            "dual": [],
            "rho": [],
            "eps_pri": [], # adaptive Boyd primal tolerance each iter
            "eps_dual": [] # adaptive Boyd dual tolerance each iter
        }

        # per-iteration per-subproblem final objective tracking (same format as residual_history)
        self.objective_history = {
            "obj_per_batch": [],       # list[list]: per-subproblem final objective each iter
            "local_dc_obj_per_batch": [], # list[list]: per-subproblem final objective without ADMM penalty each iter
            "consensus_weights": [],   # list[list]: per-batch consensus-aggregation weight each iter
        }

        self.train_runtime = 0

    # ----------------- ADMM INITIALISATION METHODS -------------------

    def warm_init_params(self):
        '''
        Warm-initialise the consensus params and  
        all submodel params from a single (unpenalised) submodel solve.
        '''
        warmup_model = NODEPyomo(
            Y_obs=self.Y_obs[0], # choose first batch for warmup solve
            layer_sizes=self.layer_sizes,
            end_time=self.end_time,
            state_lower_bound=self.state_lower_bound, state_upper_bound=self.state_upper_bound,
            param_lower_bound=self.param_lower_bound, param_upper_bound=self.param_upper_bound,
            l2_reg_param=self.l2_reg_param,
            admm_submodel=False, # no consensus penalty term => params not forced towards zero  
            transcription_method=self.transcription_method,
            irrdc_param=self.irrdc_param,  
            num_res_eval_nodes=self.num_res_eval_nodes,
            obs_times=self.obs_times
        )
        solver_options = {
            "max_iter": 1000,
            "tol": 1e-7,
            "acceptable_tol": 1e-6,
            "acceptable_iter": 15,
        }
        warmup_model.solve_model(solver_options)
        init_consensus_Ws = [warmup_model.pyomo_var_to_numpy(
                                warmup_model.model.Ws[l].W, (self.layer_sizes[l-1], self.layer_sizes[l])
                            ) for l in warmup_model.model.l]
        init_consensus_bs = [warmup_model.pyomo_var_to_numpy(
                                warmup_model.model.bs[l].b, (1, self.layer_sizes[l])
                            ) for l in warmup_model.model.l]
        init_submodels_Ws = [[w.copy() for w in init_consensus_Ws] for _ in range(self.num_batches)]
        init_submodels_bs = [[b.copy() for b in init_consensus_bs] for _ in range(self.num_batches)]

        return init_consensus_Ws, init_consensus_bs, init_submodels_Ws, init_submodels_bs

    def zero_init_dual_vars(self):
        dual_var_Ws = [
            [np.zeros((self.layer_sizes[l], self.layer_sizes[l+1])) for l in range(len(self.layer_sizes) - 1)]
            for _ in range(self.num_batches)
        ]
        dual_var_bs = [
            [np.zeros((1, self.layer_sizes[l+1])) for l in range(len(self.layer_sizes) - 1)]
            for _ in range(self.num_batches)
        ]
        return dual_var_Ws, dual_var_bs


    # ------------ SUBMODEL SOLVE METHODS -----------------

    def solve_submodels_sequential(self, solver_options: dict, iteration_count: int):
        for i in range(self.num_batches):
            self.workers[i].broadcast_consensus_and_duals(
                self.consensus_Ws, self.consensus_bs,
                self.dual_var_Ws[i], self.dual_var_bs[i],
                iteration_count,
                self.admm_penalty_param
            )
            self.workers[i].solve_submodel(solver_options)
            self.submodels_Ws[i] = self.workers[i].submodel_Ws
            self.submodels_bs[i] = self.workers[i].submodel_bs
            self.submodels_obj[i] = self.workers[i].submodel_obj
            self.submodels_local_dc_obj[i] = self.workers[i].submodel_local_dc_obj

    def solve_submodels_parallel(self, solver_options: dict, iteration_count: int):
        payload = {
            "consensus_Ws": self.consensus_Ws,
            "consensus_bs": self.consensus_bs,
            "dual_var_Ws": self.dual_var_Ws,
            "dual_var_bs": self.dual_var_bs,
            "solver_options": solver_options,
            "iteration_count": iteration_count,
            "admm_penalty_param": self.admm_penalty_param
        }
        results = self.pool_manager.map(payload)
        for worker_id, submodel_Ws, submodel_bs, submodel_obj, submodel_local_dc_obj in results:
            self.submodels_Ws[worker_id] = submodel_Ws
            self.submodels_bs[worker_id] = submodel_bs
            self.submodels_obj[worker_id] = submodel_obj
            self.submodels_local_dc_obj[worker_id] = submodel_local_dc_obj


    # --------------- ADMM UPDATE METHODS -------------------

    def update_consensus_params_mean(self):
        '''Update consensus params by averaging submodel params.'''
        old_consensus_Ws = [w.copy() for w in self.consensus_Ws]
        old_consensus_bs = [b.copy() for b in self.consensus_bs]

        self.consensus_Ws = [w.copy() for w in self.submodels_Ws[0]]
        self.consensus_bs = [b.copy() for b in self.submodels_bs[0]]
        for b in range(1, self.num_batches):
            for l in range(len(self.layer_sizes) - 1):
                self.consensus_Ws[l] += self.submodels_Ws[b][l]
                self.consensus_bs[l] += self.submodels_bs[b][l]
        for l in range(len(self.layer_sizes) - 1):
            self.consensus_Ws[l] /= self.num_batches
            self.consensus_bs[l] /= self.num_batches

        return old_consensus_Ws, old_consensus_bs

    def update_consensus_params_lossweighted(self, beta: float = 0.5):
        '''
        Update consensus params by loss-weighted averaging submodel params, using a softmax
        over each subproblem's unpenalised (local DC) objective. Failed solves (NaN objective)
        get weight 0; if every submodel failed, falls back to mean aggregation.
        Returns (weights, old_consensus_Ws, old_consensus_bs).
        '''
        # convert submodel objectives (unpenalised) to weights (lower obj -> higher weight)
        obj_array = np.nan_to_num(np.array(self.submodels_local_dc_obj), nan=np.inf) # replace NaN with +inf so that failed solves get weight 0
        if np.all(np.isinf(obj_array)):
            # all submodels failed to solve => fallback to mean aggregation (uniform weights)
            print("All submodels failed to solve; falling back to mean consensus aggregation.")
            old_consensus_Ws, old_consensus_bs = self.update_consensus_params_mean()
            uniform_weights = np.full(self.num_batches, 1.0 / self.num_batches)
            return uniform_weights, old_consensus_Ws, old_consensus_bs
        weights = np.exp(-beta * obj_array) # exp(-β*obj) to ensure positivity
        weights /= np.sum(weights) # normalize to sum to 1

        old_consensus_Ws = [w.copy() for w in self.consensus_Ws]
        old_consensus_bs = [b.copy() for b in self.consensus_bs]

        self.consensus_Ws = [
            sum(weights[b] * self.submodels_Ws[b][l] 
            for b in range(self.num_batches)) 
            for l in range(len(self.layer_sizes) - 1)
        ]
        self.consensus_bs = [
            sum(weights[b] * self.submodels_bs[b][l] 
            for b in range(self.num_batches)) 
            for l in range(len(self.layer_sizes) - 1)   
        ]

        return weights, old_consensus_Ws, old_consensus_bs

    def update_consensus_params(self, consensus_aggregator: str, iteration_count: int,
            lossweighted_start_iter: int = 150):
        '''
        Dispatch to the chosen consensus aggregation. Loss-weighting is delayed until the run
        has settled (iteration_count > lossweighted_start_iter); before that point, and for the
        "mean" aggregator, plain mean aggregation is used. Always returns
        (weights, old_consensus_Ws, old_consensus_bs) so the caller never has to branch;
        weights are uniform (1/B) whenever mean aggregation is used.
        '''
        if consensus_aggregator == "loss-weighted" and iteration_count > lossweighted_start_iter:
            return self.update_consensus_params_lossweighted()
        old_consensus_Ws, old_consensus_bs = self.update_consensus_params_mean()
        weights = np.full(self.num_batches, 1.0 / self.num_batches)
        return weights, old_consensus_Ws, old_consensus_bs

    def update_dual_vars(self):
        '''Update dual variables for each subproblem. Call after update_consensus_params.'''
        for b in range(self.num_batches):
            for l in range(len(self.layer_sizes) - 1):
                self.dual_var_Ws[b][l] += (
                    self.admm_penalty_param * (self.submodels_Ws[b][l] - self.consensus_Ws[l])
                )
                self.dual_var_bs[b][l] += (
                    self.admm_penalty_param * (self.submodels_bs[b][l] - self.consensus_bs[l])
                )

    def compute_primal_residual(self):
        '''
        Returns vector of per-subproblem L2-norm primal residuals - ||θ_b^k - θ^k||^2_2 -  
        and the total primal residual - sqrt(Σ_b||θ_b^k - θ^k||^2_2) - as tuple
        (total_r_primal, r_primal).
        '''
        r_primal = np.zeros(self.num_batches) # store per-subproblem primal residuals
        for b in range(self.num_batches):
            for l in range(len(self.layer_sizes) - 1):
                r_primal[b] += (np.linalg.norm(self.submodels_Ws[b][l] - self.consensus_Ws[l]) ** 2)
                r_primal[b] += (np.linalg.norm(self.submodels_bs[b][l] - self.consensus_bs[l]) ** 2)

        return np.sqrt(np.sum(r_primal)), r_primal

    def compute_dual_residual(self, old_consensus_Ws, old_consensus_bs):
        '''Returns dual residual - sqrt(B)*rho^k*||(θ^k - θ^{k-1})||_2'''
        s_dual = 0
        for l in range(len(self.layer_sizes) - 1):
            s_dual += np.linalg.norm(self.consensus_Ws[l] - old_consensus_Ws[l])
            s_dual += np.linalg.norm(self.consensus_bs[l] - old_consensus_bs[l])

        return np.sqrt(self.num_batches) * self.admm_penalty_param * s_dual

    def update_admm_penalty_param(self, 
            current_admm_penalty_param: float, 
            current_primal_res: float, 
            current_dual_res: float
    ):
        '''
        Update the ADMM penalty parameter based on the current primal and dual residuals.
        This is a simple heuristic: if the primal residual is much larger than the dual residual,
        increase the penalty parameter; if the dual residual is much larger, decrease it.
        '''
        mu = 3.0  # factor to compare primal and dual residuals
        tau_incr = 2.0  # factor to increase penalty
        tau_decr = 2.0  # factor to decrease penalty
        rho_min, rho_max = 1e-2, 1e2  # clamp so residual balancing can't drift rho to extremes

        if current_primal_res > mu * current_dual_res:
            new_admm_penalty_param = current_admm_penalty_param * tau_incr
        elif current_dual_res > mu * current_primal_res:
            new_admm_penalty_param = current_admm_penalty_param / tau_decr
        else:
            new_admm_penalty_param = current_admm_penalty_param

        return float(np.clip(new_admm_penalty_param, rho_min, rho_max))

    def update_primal_stopping_condition(self,
            abs_tol: float, rel_tol: float,
            submodels_Ws, submodels_bs,
            consensus_Ws, consensus_bs
    ):
        '''
        Boyd's adaptive primal tolerance:
          ε_pri = sqrt(B*P) * abs_tol + rel_tol * max[(sum_b||theta_b||_2^2)^0.5, (B*||consensus||_2^2)^0.5]
        Recomputed each ADMM iteration from the current submodel and consensus params.
        '''
        eps_abs = np.sqrt(self.num_batches * self.num_params) * abs_tol
        if not self.adaptive_tol_update:
            return eps_abs

        Ax = np.zeros(self.num_batches)
        for b in range(self.num_batches):
            for l in range(len(self.layer_sizes) - 1):
                Ax[b] += (np.linalg.norm(submodels_Ws[b][l]) ** 2)
                Ax[b] += (np.linalg.norm(submodels_bs[b][l]) ** 2)
        norm_Ax = np.sqrt(np.sum(Ax))

        Bz = 0
        for l in range(len(self.layer_sizes) - 1):
            Bz += (np.linalg.norm(consensus_Ws[l]) ** 2)
            Bz += (np.linalg.norm(consensus_bs[l]) ** 2)
        norm_Bz = np.sqrt(self.num_batches * Bz)

        return eps_abs + rel_tol * max(norm_Ax, norm_Bz)

    def update_dual_stopping_condition(self,
            abs_tol: float, rel_tol: float,
            dual_var_Ws, dual_var_bs
    ):
        '''
        Boyd's adaptive dual tolerance:
          ε_dual = sqrt(B*P)*abs_tol + rel_tol * (sum_b||lambda_b||_2^2)^0.5
        Recomputed each ADMM iteration from the current dual variables.
        '''
        eps_abs = np.sqrt(self.num_batches * self.num_params) * abs_tol
        if not self.adaptive_tol_update:
            return eps_abs

        Ay = np.zeros(self.num_batches)
        for b in range(self.num_batches):
            for l in range(len(self.layer_sizes) - 1):
                Ay[b] += (np.linalg.norm(dual_var_Ws[b][l]) ** 2)
                Ay[b] += (np.linalg.norm(dual_var_bs[b][l]) ** 2)
        norm_lambda = np.sqrt(np.sum(Ay))

        return eps_abs + rel_tol * norm_lambda


    #----------------- WALL-CLOCK CONVERGENCE TRACKING -------------------

    def compute_prediction_mse(self,
            Y_obs, t_grid,
            method: str = 'RK45',
            rtol: float = 1e-3,
            atol: float = 1e-6,
            max_step: float = np.inf
    ):
        '''
        MSE between observed trajectories and trajectories predicted
        with the current consensus parameters. The MSE is computed per batch, then
        the mean and std across batches are returned.
        '''
        per_batch_mse = []
        for b in range(Y_obs.shape[0]):
            y0 = Y_obs[b, 0, :]
            try:
                pred = self.get_predicted_trajectory(
                    y0, t_grid, method=method, rtol=rtol, atol=atol, max_step=max_step
                )
            except RuntimeError:
                return np.nan, np.nan # unstable params early in training => skip point
            per_batch_mse.append(np.mean((pred - Y_obs[b]) ** 2))
        if not per_batch_mse:
            return np.nan, np.nan
        
        return float(np.mean(per_batch_mse)), float(np.std(per_batch_mse))
    

    #--------------------------- MAIN ADMM LOOP --------------------------------

    def set_solver_options(self,
            early_solver_options: dict, 
            late_solver_options: dict, 
            num_early_iterations: int, 
            iteration_count: int
    ):
        '''
        IPOPT options for the current ADMM iteration: the soft `early_solver_options' for the
        first `num_early_iterations` iterations, then the standard `late_solver_options'.
        '''
        if iteration_count < num_early_iterations:
            return early_solver_options
        return late_solver_options

    def run_admm_training(self,
            early_solver_options: dict,
            late_solver_options: dict,
            max_admm_iterations: int,
            admm_abs_tol: float, # ε_abs
            admm_rel_tol: float, # ε_rel
            num_early_iterations: int = 10,
            near_stopping_band: float = 1.5,
            num_convergence_confirmations: int = 3,
            consensus_aggregator: str = "loss-weighted", # "mean" | "loss-weighted"
            training_clock_budget: float = None,
            mse_eval_options: dict = None
    ):
        '''Run the ADMM consensus training loop.'''
        iteration_count = 0
        conv_count = 0

        t_start = time.perf_counter()
        try:
            for iteration in range(max_admm_iterations):

                if self.adaptive_penalty_update and iteration_count > 0:
                    self.admm_penalty_param = self.update_admm_penalty_param(
                        self.admm_penalty_param, r_primal, s_dual
                    )
                # adaptive Boyd tolerances from the CURRENT iterate (submodels / consensus / duals)
                eps_pri = self.update_primal_stopping_condition(
                    admm_abs_tol, admm_rel_tol,
                    self.submodels_Ws, self.submodels_bs, self.consensus_Ws, self.consensus_bs)
                eps_dual = self.update_dual_stopping_condition(
                    admm_abs_tol, admm_rel_tol, self.dual_var_Ws, self.dual_var_bs)
                
                print(f"ADMM iteration {iteration + 1}")
                solver_options = self.set_solver_options(
                    early_solver_options, late_solver_options, num_early_iterations, iteration_count
                )
                if self.use_pool:
                    self.solve_submodels_parallel(solver_options, iteration_count)
                else:
                    self.solve_submodels_sequential(solver_options, iteration_count)
                consensus_weights, old_consensus_Ws, old_consensus_bs = self.update_consensus_params(
                    consensus_aggregator, iteration_count
                )
                self.update_dual_vars()
                r_primal, r_primal_per_batch = self.compute_primal_residual()
                s_dual = self.compute_dual_residual(old_consensus_Ws, old_consensus_bs)

                # per-subproblem final objective (with and without ADMM penalty)
                obj_per_batch = np.array([self.submodels_obj[b] for b in range(self.num_batches)])
                local_dc_obj_per_batch = np.array([self.submodels_local_dc_obj[b] for b in range(self.num_batches)])

                iteration_count += 1

                # record per-iteration residuals and the penalty parameter used this iteration
                t_res0 = time.perf_counter()

                self.residual_history["primal_avg"].append(float(r_primal))
                self.residual_history["primal_per_batch"].append(r_primal_per_batch.tolist())
                self.residual_history["dual"].append(float(s_dual))
                self.residual_history["rho"].append(float(self.admm_penalty_param))
                self.residual_history["eps_pri"].append(float(eps_pri))
                self.residual_history["eps_dual"].append(float(eps_dual))
                self.objective_history["obj_per_batch"].append(obj_per_batch.tolist())
                self.objective_history["local_dc_obj_per_batch"].append(local_dc_obj_per_batch.tolist())
                self.objective_history["consensus_weights"].append(consensus_weights.tolist())

                print(f"Primal residual: {r_primal:.6f} (eps_pri {eps_pri:.6f})")
                print(f"Dual residual: {s_dual:.6f} (eps_dual {eps_dual:.6f})")
                print(f"Penalty param rho: {self.admm_penalty_param:.4g}")
                print("Per-batch primal residuals: "
                      + np.array2string(r_primal_per_batch, precision=4, max_line_width=np.inf))
                print("Per-batch subproblem objective: "
                      + np.array2string(obj_per_batch, precision=4, max_line_width=np.inf))
                print("Per-batch subproblem objective without ADMM penalty: "
                      + np.array2string(local_dc_obj_per_batch, precision=4, max_line_width=np.inf))
                print("Per-batch consensus weights: "
                      + np.array2string(consensus_weights, precision=4, max_line_width=np.inf))

                t_start += time.perf_counter() - t_res0  # exclude residual calculation from timer

                # Record wall-clock convergence (prediction MSE) using the current consensus model
                # Timer is paused for evaluation
                if mse_eval_options is not None:
                    t_mse0 = time.perf_counter()

                    pred_kw = {
                        "method": mse_eval_options.get("method", "RK45"),
                        "rtol": mse_eval_options.get("rtol", 1e-3),
                        "atol": mse_eval_options.get("atol", 1e-6),
                        "max_step": mse_eval_options.get("max_step", np.inf),
                    }
                    train_mse, train_mse_std = self.compute_prediction_mse(
                        mse_eval_options["Y_obs"], mse_eval_options["train_grid"], **pred_kw
                    )
                    test_mse, test_mse_std = self.compute_prediction_mse(
                        mse_eval_options["Y_test_obs"], mse_eval_options["test_grid"], **pred_kw
                    )
                    t_msefin = (time.perf_counter() - t_start) / 60.0  # training-only time at eval point
                    self.mse_history["time_min"].append(t_msefin)
                    self.mse_history["train_mse"].append(train_mse)
                    self.mse_history["train_mse_std"].append(train_mse_std)
                    self.mse_history["test_mse"].append(test_mse)
                    self.mse_history["test_mse_std"].append(test_mse_std)
                    print(f"Train MSE: {train_mse:.6e} | Test MSE: {test_mse:.6e}")

                    t_start += time.perf_counter() - t_mse0  # exclude eval cost from timer

                # stop if BOTH residuals sit within `near_stopping_band' for num_convergence_confirmations consecutive iterations.
                if r_primal <= near_stopping_band * eps_pri and s_dual <= near_stopping_band * eps_dual:
                    conv_count += 1
                else:
                    conv_count = 0
                if conv_count >= num_convergence_confirmations:
                    print(f"ADMM converged (r,s within {near_stopping_band}x tol for {conv_count} "
                          f"consecutive iters) after {iteration_count} iterations!")
                    self.train_runtime = time.perf_counter() - t_start
                    break

                # stop if the wall-clock training budget is reached (training-only time)
                if training_clock_budget is not None and (time.perf_counter() - t_start) / 60.0 >= training_clock_budget:
                    print(f"ADMM stopped: {training_clock_budget} min wall-clock budget reached at iter {iteration_count}.")
                    self.train_runtime = time.perf_counter() - t_start
                    break

                # max_admm_iterations is enforced by the enclosing for-loop range()

            # covers exit via max_admm_iterations (no break); break paths set it earlier
            self.train_runtime = time.perf_counter() - t_start

        finally:
            if self.use_pool:
                self.pool_manager.close()


    # ---------------- NODE PREDICTION METHODS --------------------

    def nn_prediction(self, t, y_0, Ws, bs):
        y_0 = y_0.reshape(1, -1)
        out = np.tanh(np.dot(y_0, Ws[0]) + bs[0])
        for W, b in zip(Ws[1:-1], bs[1:-1]):
            out = np.tanh(np.dot(out, W) + b)
        out = np.dot(out, Ws[-1]) + bs[-1]
        return out.flatten()

    def get_predicted_trajectory(self,
            y_0, t_grid,
            method: str = 'RK45',
            rtol: float = 1e-3,
            atol: float = 1e-6,
            max_step: float = np.inf
    ):
        '''Predict state trajectory over t_grid via forward integration of the consenus model.'''
        predicted_trajectory = solve_ivp(
            fun=self.nn_prediction,
            t_span=(t_grid[0], t_grid[-1]),
            y0=y_0,
            t_eval=t_grid,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            args=(self.consensus_Ws, self.consensus_bs)
        )
        if not predicted_trajectory.success:
            raise RuntimeError(
                f"solve_ivp failed: status={predicted_trajectory.status}, message={predicted_trajectory.message}"
            )
        return predicted_trajectory.y.T
    