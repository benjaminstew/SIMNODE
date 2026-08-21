"""
Monolithic collocation-based Neural ODE over B independent trajectories.

- Stitching = BLOCK DIAGONAL. Every trajectory keeps its own 
    collocation grid, its own initial-condition constraint, and its own collocation ODE
    constraints. The collocation derivative at a node depends ONLY on nodes within the
    same segment, so there is no spurious coupling across trajectory boundaries -- the
    trajectories are independent experiments each with their own initial condition. (This
    is the centralized fixed point that consensus ADMM targets, solved directly.)

- The objective is base's L summed over the B trajectories.

Practical additions beyond the base model (needed for this experiment; noted where they
appear): observation-time resampling onto the Chebyshev grid (``obs_times'', as in
node_pyomo_admm.py), and an optional ``seed'' for reproducible initialisation.
"""
import os
import time
import warnings
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.common.errors import ApplicationError

from src.models.simultaneous.direct_collocation import BarycentricInterpolation

class NODEPyomo:
    """
    Monolithic Pyomo model to train a single collocation-based Neural ODE jointly over B
    trajectories with the IPOPT solver, using block-diagonal stitching. 
    """
    def __init__(self,
            Y_obs: np.ndarray, 
            end_time: float,
            layer_sizes: list,
            state_lower_bound: float, state_upper_bound: float,
            param_lower_bound: float, param_upper_bound: float,
            l2_reg_param: float,
            obs_times: np.ndarray, 
            seed: int = None
    ):
        self.Y_obs = Y_obs
        if self.Y_obs.ndim != 3:
            raise ValueError(
                f"Y_obs must be 3D (num_segments, num_colloc_nodes, state_dim); got {self.Y_obs.shape}."
            )
        self.num_segments, self.num_colloc_nodes, self.state_dim = self.Y_obs.shape
        if self.state_dim > self.num_colloc_nodes:
            warnings.warn("Y_obs should be structured such that each row represents a collocation point.")
        self.obs_times = obs_times
        if self.obs_times.shape[0] != self.num_colloc_nodes:
            raise ValueError("obs_times length must match Y_obs.shape[1] (num_colloc_nodes).")

        self.end_time = end_time
        self.layer_sizes = layer_sizes
        if self.layer_sizes[0] != self.state_dim or self.layer_sizes[-1] != self.state_dim:
            raise ValueError(
                f"layer_sizes[0] and layer_sizes[-1] must equal state_dim ({self.state_dim}); got {self.layer_sizes}."
            )
        self.l2_reg_param = l2_reg_param  
        self.state_lower_bound = state_lower_bound
        self.state_upper_bound = state_upper_bound
        self.param_lower_bound = param_lower_bound
        self.param_upper_bound = param_upper_bound
        self.transcription_method = 'dc'
        if seed is not None:
            np.random.seed(seed)

        # Single collocation grid shared by every segment.
        # The full block-diagonal differentiation matrix is this N x N grid repeated once per
        # segment; we store only the block and offset by segment when indexing (efficient -- a
        # dense (B*N) x (B*N) matrix would be almost all zeros).
        bary = BarycentricInterpolation(0, self.end_time, self.num_colloc_nodes, transcription_method='dc')
        self.colloc_grid = bary.colloc_grid # (N,) ascending, colloc_grid[0] == 0
        self.bary_ws = bary.bary_ws
        self.D_colloc = bary.D_colloc # (N, N) local differentiation block

        # Interpolate each raw trajectory onto the collocation grid via obs_times. 
        self.Y_seg = np.empty((self.num_segments, self.num_colloc_nodes, self.state_dim), dtype=float)
        for s in range(self.num_segments):
            self.Y_seg[s] = CubicSpline(self.obs_times, self.Y_obs[s], axis=0)(self.colloc_grid)
        self.total_nodes = self.num_segments * self.num_colloc_nodes
        self.Y_obs = self.Y_seg.reshape(self.total_nodes, self.state_dim)  

        self.model = pyo.ConcreteModel()
        self.build_model()

    # ------------------------------ BASE MODEL BUILDING METHODS ------------------------------- #
    #  initialise_colloc_diff_matrix builds the LOCAL block; initialise_Y_star
    #  reads the stacked data.

    def initialise_weight(self, shape):
        '''Xavier initialisation of a weight matrix as pyomo Var'''
        W = np.random.randn(*shape) * np.sqrt(2 / (shape[0] + shape[1]))
        def init_rule(model, i, j):
            return W[i-1, j-1]  # W is a 0-indexed ndarray
        return init_rule

    def initialise_bias(self, shape):
        '''Normal random initialisation of a bias vector as pyomo Var'''
        b = np.random.randn(*shape) * 0.1
        def init_rule(model, i, j):
            return b[0, i-1]  # b is a 0-indexed ndarray
        return init_rule

    def initialise_colloc_diff_matrix(self):
        '''Initialisation of the (local, per-segment) collocation-barycentric differentiation
        matrix as a pyomo Param. This N x N block is the diagonal block repeated for every
        segment in the block-diagonal differentiation operator.'''
        def init_rule(model, n, n_prime):
            return self.D_colloc[n-1, n_prime-1]  # self.D_colloc is a 0-indexed ndarray
        return init_rule

    def initialise_Y_star(self):
        '''Initialisation of the approximate state trajectory Y_star as a pyomo Var using the
        (resampled, stacked) data plus small noise (matches node_pyomo_admm.py's Y_star init).'''
        def init_rule(model, g, d):
            return self.Y_obs[g-1, d-1] + np.random.randn() * 0.1  # self.Y_obs is a 0-indexed ndarray
        return init_rule

    def single_traj_objective_rule(self, model, s0):
        '''L_b(Y_b, theta) for segment s0 -- EXACTLY the single-trajectory DC objective from
        node_pyomo_base.py.dc_objective_rule, restricted to this segment's nodes:
            (1/N) ||Y_b - Y_obs,b||_F^2  +  lambda ||theta||_2^2.
        '''
        N = self.num_colloc_nodes
        offset = s0 * N  # global 1-index of this segment's first node, minus 1

        # MSE term: (1/N) * ||Y_star_b - Y_obs_b||_F^2 over this segment's N nodes
        mse = (1 / N) * sum(
            (model.Y_star[offset + n, d] - self.Y_obs[offset + n - 1, d-1])**2
            for n in model.nloc
            for d in model.d
        )

        # Regularised L2 norm term: lambda * ||theta||_2^2 (theta is shared across segments)
        l2_norm = sum(
            model.Ws[l].W[i, j]**2
            for l in model.l
            for i in range(1, self.layer_sizes[l-1] + 1)
            for j in range(1, self.layer_sizes[l] + 1)
        ) + sum(
            model.bs[l].b[1, j]**2
            for l in model.l
            for j in range(1, self.layer_sizes[l] + 1)
        )

        return mse + self.l2_reg_param * l2_norm

    def dc_objective_rule(self, model):
        return sum(self.single_traj_objective_rule(model, s0) for s0 in range(self.num_segments))

    def build_model(self):

        # ------------ SETS ------------
        self.model.l = pyo.RangeSet(len(self.layer_sizes) - 1)  # n layers -> n-1 W matrices / b vectors
        self.model.n = pyo.RangeSet(self.total_nodes)           # ALL collocation nodes across every segment
        self.model.d = pyo.RangeSet(self.state_dim)
        self.model.nloc = pyo.RangeSet(self.num_colloc_nodes)   # local node index within a segment

        # ----- DECISION VARIABLES -----
        # Approximate state trajectory over all stacked nodes
        self.model.Y_star = pyo.Var(
            self.model.n,
            self.model.d,
            domain=pyo.Reals,
            initialize=self.initialise_Y_star(),
            bounds=(self.state_lower_bound, self.state_upper_bound)
        )

        # Shared NN parameters (one f_theta for all segments)
        self.model.Ws = pyo.Block(self.model.l)
        self.model.bs = pyo.Block(self.model.l)
        for l in self.model.l:
            self.model.Ws[l].W = pyo.Var(
                {i for i in range(1, self.layer_sizes[l-1] + 1)},  # self.layer_sizes is 0-indexed list
                {j for j in range(1, self.layer_sizes[l] + 1)},
                domain=pyo.Reals,
                initialize=self.initialise_weight((self.layer_sizes[l-1], self.layer_sizes[l])),
                bounds=(self.param_lower_bound, self.param_upper_bound)
            )
            self.model.bs[l].b = pyo.Var(
                {1},
                {i for i in range(1, self.layer_sizes[l] + 1)},
                domain=pyo.Reals,
                initialize=self.initialise_bias((1, self.layer_sizes[l])),
                bounds=(self.param_lower_bound, self.param_upper_bound)
            )

        #------ INITIAL CONDITION CONSTRAINTS (one per segment) ------
        # Base pins Y_star[1, :] = Y_obs[0, :]. Here each segment's first node is pinned to that
        # trajectory's initial condition: node offset+1 for segment s0 (offset = s0*N).
        self.model.initial_condition = pyo.ConstraintList()
        for s0 in range(self.num_segments):
            offset = s0 * self.num_colloc_nodes
            for d in self.model.d:
                self.model.initial_condition.add(self.model.Y_star[offset + 1, d] == self.Y_obs[offset, d-1])

        # --------- COLLOCATION CONSTRAINT --------
        # Local (per-segment) differentiation block as a pyomo Param
        self.model.D_colloc = pyo.Param(
            self.model.nloc,
            self.model.nloc,
            domain=pyo.Reals,
            initialize=self.initialise_colloc_diff_matrix()
        )

        # LHS of ODE_colloc (dY_star/dt) as pyomo expressions -- BLOCK DIAGONAL: the derivative at a
        # global node uses ONLY its own segment's nodes. Base's version sums over all of model.n
        # (single trajectory); here we sum over the local block of the node's own segment.
        def ode_lhs_colloc_rule(model, g, d):
            s0 = (g - 1) // self.num_colloc_nodes     # segment (0-indexed)
            nloc = (g - 1) % self.num_colloc_nodes + 1  # local node within the segment (1-indexed)
            offset = s0 * self.num_colloc_nodes
            return sum(model.Y_star[offset + j, d] * model.D_colloc[nloc, j] for j in model.nloc)
        self.model.ode_lhs_colloc = pyo.Expression(self.model.n, self.model.d, rule=ode_lhs_colloc_rule)

        # Output of the shared NN at every (stacked) collocation node
        nn_output_colloc = self.nn_feedforward_pyo(self.model.Y_star, self.total_nodes)

        # Build Pyomo objective and constraints (DC)
        self.build_objective()
        self.build_constraints(nn_output_colloc)

    def build_objective(self):
        self.model.obj = pyo.Objective(rule=lambda m: self.dc_objective_rule(m), sense=pyo.minimize)

    def build_constraints(self, nn_output_colloc):
        # ODE equality constraint dY_star/dt = NN(Y_star) at every node EXCEPT each segment's
        # first node (whose value is already pinned by that segment's initial-condition constraint).
        self.model.ode = pyo.ConstraintList()
        for g in range(1, self.total_nodes + 1):
            if (g - 1) % self.num_colloc_nodes == 0:
                continue  # first node of a segment -> initial condition, skip (as base skips n=1)
            for d in self.model.d:
                self.model.ode.add(self.model.ode_lhs_colloc[g, d] == nn_output_colloc[g-1, d-1])


    # ----------------------------------- TRAINING METHODS ----------------------------------- #

    def nn_feedforward_pyo(self, Y, num_nodes):
        '''
        Compute a feedforward pass of the NN (act func = tanh, linear output layer)
        Y is a time invariant pyo Var or ndarray of pyo Expressions.
        Returns: output layer activations as an ndarray (i, j) of pyo.Expressions (one for each element of Y)
        (Copied verbatim from node_pyomo_base.py.)
        '''
        # Input activations = Y
        activations = [Y]
        # For each layer
        for l in self.model.l:
            # Create ndarray to store expressions for current layer
            current_activation = np.empty((num_nodes, self.layer_sizes[l]), dtype=object)
            # For each unit in layer
            for u in range(1, self.layer_sizes[l] + 1):
                # For each node (collocation or residual evaluation)
                for i in range(1, num_nodes + 1):
                    # Linear output: activations[l-1] . W + b
                    if l == 1:  # if at first hidden layer
                        linear_out = sum(activations[l-1][i, d] * self.model.Ws[l].W[d, u] for d in self.model.d) + self.model.bs[l].b[1, u]
                    else:
                        linear_out = sum(activations[l-1][i-1, k-1] * self.model.Ws[l].W[k, u] for k in range(1, self.layer_sizes[l-1] + 1)) + self.model.bs[l].b[1, u]
                    # Apply activation func: tanh for hidden layers, linear for output layer
                    if l < len(self.layer_sizes) - 1:
                        current_activation[i-1, u-1] = pyo.tanh(linear_out)
                    else:
                        current_activation[i-1, u-1] = linear_out
            activations.append(current_activation)

        return activations[-1]

    def freeze_vars(self):
        '''Fix values of vars in pyomo model'''
        self.model.Y_star.fix()
        for l in self.model.l:
            self.model.Ws[l].W.fix()
            self.model.bs[l].b.fix()

    def solve_model(self, solver_options: dict):
        '''
        Solve the monolithic Pyomo model using the IPOPT solver.
        
        Returns:
            self.last_solve_info dict.
        '''
        # set up IPOPT solver 
        #solver = pyo.SolverFactory("ipopt", executable="/rds/general/user/bms125/home/miniforge3/envs/cnode/bin/ipopt") # HPC
        solver = pyo.SolverFactory("ipopt", executable="/usr/local/bin/ipopt")
        #solver = pyo.SolverFactory("ipopt", executable="/opt/homebrew/bin/ipopt") 
        #print("Solver available?: {}".format(solver.available())) 

        solver.options['max_iter'] = solver_options["max_iter"]
        solver.options['tol'] = solver_options["tol"]
        solver.options['acceptable_tol'] = solver_options["acceptable_tol"]
        solver.options['acceptable_iter'] = solver_options["acceptable_iter"]
        repo_root = Path(__file__).resolve().parents[3]  # .../CNODE
        option_path = repo_root / "ipopt.opt"
        if not option_path.exists():
            raise FileNotFoundError(f"IPOPT option file not found at {option_path}")
        solver.options["option_file_name"] = str(option_path)

        t0 = time.perf_counter()
        # load_solutions=True lets Pyomo load the solution straight into the model on success.
        # Wrapped in try/except so a genuinely failed solve does not raise. NOTE: with
        # load_solutions=True Pyomo EMPTIES result.solution after loading, so success must be
        # judged from the solver status / termination condition, NOT len(result.solution).
        result = None
        try:
            result = solver.solve(self.model, tee=False, load_solutions=True)
            tc = result.solver.termination_condition
            solution_loaded = (
                result.solver.status in (SolverStatus.ok, SolverStatus.warning)
                and tc in (
                    TerminationCondition.optimal,
                    TerminationCondition.locallyOptimal,
                    TerminationCondition.feasible,
                    TerminationCondition.maxIterations,
                )
            )
        except (ApplicationError, ValueError) as exc:
            solution_loaded = False
            print(f"Monolithic solve raised {type(exc).__name__}: {exc}.")
        solve_wall_time = time.perf_counter() - t0

        status = str(result.solver.status) if result is not None else "error"
        termination = str(result.solver.termination_condition) if result is not None else "solve_raised"
        iterations = getattr(result.solver, "iterations", None) if result is not None else None
        self.last_solve_info = {
            "status": status,
            "termination_condition": termination,
            "iterations": int(iterations) if iterations is not None else None,
            "solve_wall_time_s": solve_wall_time,
            "solution_loaded": solution_loaded,
        }

        if solution_loaded:
            self.freeze_vars()
            print("Monolithic solve loaded (status={}, termination={}); NN parameters frozen.".format(
                status, termination))
        else:
            print("Monolithic solve returned no usable solution (status={}, termination={}).".format(
                status, termination))

        return self.last_solve_info


    # ------------------------------ DIAGNOSTIC METHODS ------------------------------- #

    def check_param_values(self):
        '''Print summary statistics for NN parameters.'''
        for l in self.model.l:
            W = self.pyomo_var_to_numpy(self.model.Ws[l].W, (self.layer_sizes[l-1], self.layer_sizes[l]))
            b = self.pyomo_var_to_numpy(self.model.bs[l].b, (1, self.layer_sizes[l]))
            print(f"Layer {l}: W range [{W.min():.4f}, {W.max():.4f}], |W| mean {np.abs(W).mean():.4f}")
            print(f"Layer {l}: b range [{b.min():.4f}, {b.max():.4f}], |b| mean {np.abs(b).mean():.4f}")


    # ---------------------------- PREDICTION METHODS -------------------------------- #

    def pyomo_var_to_numpy(self, pyo_obj, shape):
        '''Copy pyo_obj (a Pyomo Var/Param/Expression) into a numpy ndarray.'''
        return np.array([[pyo.value(pyo_obj[i, j]) for j in range(1, shape[1]+1)] for i in range(1, shape[0]+1)])

    def convert_weights(self):
        return [self.pyomo_var_to_numpy(self.model.Ws[l].W, (self.layer_sizes[l-1], self.layer_sizes[l])) for l in self.model.l]

    def convert_biases(self):
        return [self.pyomo_var_to_numpy(self.model.bs[l].b, (1, self.layer_sizes[l])) for l in self.model.l]

    def nn_prediction(self, t, y_0, Ws, bs):
        y_0 = y_0.reshape(1, -1)  # reshape to row vector
        out = np.tanh(np.dot(y_0, Ws[0]) + bs[0])  # first hidden layer
        for W, b in zip(Ws[1:-1], bs[1:-1]):
            out = np.tanh(np.dot(out, W) + b)  # calc activation for hidden layer, move forward
        out = np.dot(out, Ws[-1]) + bs[-1]  # linear output layer

        return out.flatten()  # return 1D array for solve_ivp

    def get_predicted_trajectory(self,
            y_0, t_grid,
            method: str = 'RK45',
            rtol: float = 1e-3,
            atol: float = 1e-6,
            max_step: float = np.inf
    ):
        '''
        Predict state trajectory over t_grid via forward integration of the trained NN using scipy.solve_ivp.
        (Copied verbatim from node_pyomo_base.py.)
        '''
        if t_grid.ndim != 1 or t_grid.size < 2:
            raise ValueError(f"t_grid must be a 1D array with >=2 points; got shape {t_grid.shape}.")
        if not np.all(np.isfinite(t_grid)):
            raise ValueError("t_grid must contain only finite values.")
        if np.any(np.diff(t_grid) <= 0):
            raise ValueError("t_grid must be strictly increasing (solve_ivp expects monotone time).")

        y_0 = np.asarray(y_0, dtype=float)
        if y_0.ndim != 1:
            raise ValueError(f"y_0 must be 1D; got shape {y_0.shape}.")
        if hasattr(self, 'state_dim') and y_0.shape[0] != self.state_dim:
            raise ValueError(f"y_0 has length {y_0.shape[0]} but state_dim is {self.state_dim}.")
        if not np.all(np.isfinite(y_0)):
            raise ValueError("y_0 must contain only finite values.")

        Ws = self.convert_weights()
        bs = self.convert_biases()

        predicted_trajectory = solve_ivp(
            fun=self.nn_prediction,
            t_span=(t_grid[0], t_grid[-1]),
            y0=y_0,
            t_eval=t_grid,
            method=method,
            rtol=rtol, 
            atol=atol,
            max_step=max_step,
            args=(Ws, bs)
        )

        if not predicted_trajectory.success:
            raise RuntimeError("solve_ivp failed: " f"status={predicted_trajectory.status}, message={predicted_trajectory.message}")

        return predicted_trajectory.y.T

    def compute_prediction_mse(self, 
            t_grid, 
            Y_obs, 
            method='RK45', 
            rtol=1e-3, 
            atol=1e-6, 
            max_step=np.inf
    ):
        '''
        Mean / std of per-trajectory MSE under the trained network, matching
        ADMM.compute_prediction_mse semantics (per-batch MSE over (time, state), then
        mean / std across batches). Returns (nan, nan) if any forward solve fails.

        Args:
            t_grid: (num_nodes,) time grid to integrate / evaluate on.
            Y_obs:  (num_batches, num_nodes, state_dim) reference trajectories (initial
                    conditions taken from Y_obs[:, 0, :]).
        '''
        Y_obs = np.asarray(Y_obs, dtype=float)
        per_batch = []
        for b in range(Y_obs.shape[0]):
            try:
                pred = self.get_predicted_trajectory(Y_obs[b, 0, :], t_grid, method, rtol, atol, max_step)
            except RuntimeError:
                return np.nan, np.nan
            per_batch.append(np.mean((pred - Y_obs[b]) ** 2))
        per_batch = np.asarray(per_batch, dtype=float)
        return float(per_batch.mean()), float(per_batch.std())
