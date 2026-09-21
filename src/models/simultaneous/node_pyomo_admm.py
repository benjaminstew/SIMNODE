import warnings
import numpy as np 
import time
from pathlib import Path
from scipy.interpolate import CubicSpline
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.common.errors import ApplicationError

from src.models.simultaneous.direct_collocation import BarycentricInterpolation

class NODEPyomo:
    """
    Implementation of a base Pyomo model to train collocation-based Neural ODEs using the IPOPT solver. 
    Additional methods for ADMM-based batched training are implemented.
    """
    def __init__(self, 
            Y_obs: np.ndarray, 
            layer_sizes: list, 
            end_time: float,
            state_lower_bound: float, state_upper_bound: float,
            param_lower_bound: float, param_upper_bound: float,
            l2_reg_param: float,
            admm_submodel: bool = True,
            admm_penalty_param: float = None,
            transcription_method: str = 'dc',
            irrdc_param: float = None,
            num_res_eval_nodes: int = None,
            obs_times: np.ndarray = None,
            seed: int = None
    ):
        self.rng = np.random.default_rng(seed) # if seed = None -> nondeterministic; int -> reproducible
        if Y_obs.ndim != 3:
            raise ValueError(
                f"Y_obs must be 3D (num_traj, num_colloc_nodes, state_dim); got {Y_obs.shape}. "
                f"A single trajectory should be passed with a leading axis of length 1."
            )
        self.num_traj, self.num_colloc_nodes, self.state_dim = Y_obs.shape
        if self.state_dim > self.num_colloc_nodes:
            warnings.warn(
                f"Y_obs should be structured such that each row represents a new collocation point."
            )
        self.end_time = end_time
        if transcription_method == 'irrdc' and self.num_traj > 1:
            raise ValueError(
                "Multi-trajectory subproblems (num_traj > 1) are only supported for "
                "transcription_method='dc'; got 'irrdc'."
            )
        bary = BarycentricInterpolation(
            0, self.end_time,
            self.num_colloc_nodes,
            transcription_method=transcription_method,
            num_res_eval_nodes=num_res_eval_nodes
        )
        self.colloc_grid = bary.colloc_grid # (num_colloc_nodes,)
        self.bary_ws = bary.bary_ws
        self.D_colloc = bary.D_colloc # (num_colloc_nodes, num_colloc_nodes) local block
        self.total_nodes = self.num_traj * self.num_colloc_nodes

        # Align each trajectory onto the collocation grid (if obs_times given), then stack the
        # trajectories (total_nodes, state_dim) array indexed by a global node index
        # (num_traj=1 recovers the single-trajectory 2D behaviour).
        if obs_times is not None:
            obs_times = np.asarray(obs_times)
            if obs_times.shape[0] != self.num_colloc_nodes:
                raise ValueError("obs_times length must match the number of Y_obs collocation nodes.")
            Y_seg = np.empty((self.num_traj, self.num_colloc_nodes, self.state_dim), dtype=float)
            for s in range(self.num_traj):
                Y_seg[s] = CubicSpline(obs_times, Y_obs[s], axis=0)(self.colloc_grid)
        else:
            Y_seg = np.asarray(Y_obs, dtype=float)
        self.Y_obs = Y_seg.reshape(self.total_nodes, self.state_dim)

        self.layer_sizes = layer_sizes
        self.model = pyo.ConcreteModel()
        self.state_lower_bound = state_lower_bound
        self.state_upper_bound = state_upper_bound
        self.param_lower_bound = param_lower_bound
        self.param_upper_bound = param_upper_bound
        self.l2_reg_param = l2_reg_param 
        
        self.transcription_method = transcription_method
        if self.transcription_method == 'irrdc': 
            if irrdc_param is None or num_res_eval_nodes is None:
                raise ValueError("Need to set hyperparameter irrdc_param and num_res_eval_nodes for irrdc method.")
            self.num_res_eval_nodes = num_res_eval_nodes
            self.res_eval_grid = bary.res_eval_grid
            self.L_res = bary.L_res
            self.D_res = bary.D_res
            self.irrdc_param = irrdc_param # regularisation hyperparameter for residual penalty term 
            self.quadrature_ws = bary.quadrature_ws # quadrature weights for residual penalty term

        self.admm_submodel = admm_submodel
        if self.admm_submodel:
            if admm_penalty_param is None:
                raise ValueError("Must set penalty hyperparameter admm_penalty_param for ADMM submodel.")  
            self.admm_penalty_param = admm_penalty_param # ADMM penalty parameter for consensus penalty term 
        
        self.build_model()


    # ------------------------------ BASE MODEL BUILDING METHODS ------------------------------- #

    def initialise_weight(self, shape):
        '''Xavier initialisation of a weight matrix as pyomo Var'''
        W = self.rng.standard_normal(shape) * np.sqrt(2 / (shape[0] + shape[1]))
        def init_rule(model, i, j):
            return W[i-1, j-1] # W is a 0-indexed ndarray 
        return init_rule
    
    def initialise_bias(self, shape):
        '''Zero initialisation of a bias vector as pyomo Var'''
        b = np.zeros(shape)
        def init_rule(model, i, j):
            return b[0, i-1] # b is a 0-indexed ndarray
        return init_rule
    
    def initialise_colloc_diff_matrix(self):
        '''Initialisation of collocation-barycentric differentiation matrix as a pyomo Param.'''
        def init_rule(model, n, n_prime):
            return self.D_colloc[n-1, n_prime-1] # self.D_colloc is a 0-indexed ndarray
        return init_rule
    
    def initialise_interpolant_eval_matrix(self):
        '''Initialisation of the interpolant evaluation matrix as a pyomo Param.'''
        def init_rule(model, m, n):
            return self.L_res[m-1, n-1] # self.L_res is a 0-indexed ndarray
        return init_rule
    
    def initialise_res_diff_matrix(self):
        '''Initialisation of residual-evaluation-barycentric differentiation matrix as a pyomo Param.'''
        def init_rule(model, m, n):
            return self.D_res[m-1, n-1] # self.D_res is a 0-indexed ndarray
        return init_rule
    
    def initialise_Y_star(self):
        '''Initialisation of the approximate state trajectory Y_star as a pyomo Var using smooth data.'''
        def init_rule(model, m, d):
            return self.Y_obs[m-1, d-1] # self.Y_obs is a 0-indexed ndarray
        return init_rule
    
    def local_dc_objective_rule(self, model):
        '''
        Underlying DC objective as a pyomo expression, without the ADMM consensus penalty -
        sum_s (1/N) * ||Y_star_s - Y_obs_s||_F^2 + Λ * ||θ||_2^2.

        The misfit is the per-trajectory DC misfit (1/N)||.||_F^2 summed over the num_traj
        trajectories stitched block-diagonally into this subproblem (num_traj=1 recovers the
        single-trajectory objective). The L2 term is added ONCE: θ is shared across all
        trajectories in the subproblem.
        '''
        N = self.num_colloc_nodes
        # MSE term: sum over trajectories of (1/N) * ||Y_star_s - Y_obs_s||_F^2
        mse = sum(
            (1 / N) * sum(
                (model.Y_star[s0 * N + n, d] - self.Y_obs[s0 * N + n - 1, d-1])**2
                for n in model.nloc
                for d in model.d
            )
            for s0 in range(self.num_traj)
        )
        # Reg L2 norm term: Λ * ||Θ||_2^2 (θ shared across trajectories -> counted once)
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
        '''Returns the DC objective function as a pyomo expression.'''
        if self.admm_submodel:
            # ADMM consensus penalty term: (ρ/2) * ||θ - θ_consensus + (1/ρ) * λ||_2^2
            consensus_penalty = (model.admm_penalty_param / 2) * (sum(
                (model.Ws[l].W[i, j] - model.consensus_Ws[l].W[i, j] + (1 / model.admm_penalty_param) * model.dual_var_Ws[l].W[i, j])**2
                for l in model.l
                for i in range(1, self.layer_sizes[l-1] + 1)
                for j in range(1, self.layer_sizes[l] + 1)
            ) + sum(
                (model.bs[l].b[1, j] - model.consensus_bs[l].b[1, j] + (1 / model.admm_penalty_param) * model.dual_var_bs[l].b[1, j])**2
                for l in model.l
                for j in range(1, self.layer_sizes[l] + 1)
            ))
            return model.local_dc_objective + consensus_penalty
        
        return model.local_dc_objective
    
    def irrdc_objective_rule(self, model, nn_output_res):
        '''Returns the objective function for IRR-DC'''
        reg_mse = self.dc_objective_rule(model) 

        # Residual penalty term: (1 / (2 * μ)) * sum_n(w_m * ||interpolant_m||_2^2) 
        residual_penalty = (1 / (2 * self.irrdc_param)) * sum(
            self.quadrature_ws[m-1] * (model.ode_lhs_res[m, d] - nn_output_res[m-1, d-1])**2
            for m in model.m
            for d in model.d
        )

        return reg_mse + residual_penalty

    def build_model(self):

        # ------------ SETS ------------
        self.model.l = pyo.RangeSet(len(self.layer_sizes) - 1) # [1:len(self.layer_sizes)-1] as for n layers, have n-1 W matrices/b vectors
        self.model.n = pyo.RangeSet(self.total_nodes)          # ALL collocation nodes across every trajectory
        self.model.nloc = pyo.RangeSet(self.num_colloc_nodes)  # local node index within a trajectory
        self.model.d = pyo.RangeSet(self.state_dim)
        self.model.m = pyo.RangeSet(self.num_res_eval_nodes) if self.transcription_method == 'irrdc' else None

        # ----- DECISION VARIABLES -----
        # Initialise approximate state trajectory
        self.model.Y_star = pyo.Var(
            self.model.n, 
            self.model.d, 
            domain=pyo.Reals, 
            initialize=self.initialise_Y_star(),  
            bounds=(self.state_lower_bound, self.state_upper_bound)
        )
        
        # Initialise NN parameters
        self.model.Ws = pyo.Block(self.model.l) 
        self.model.bs = pyo.Block(self.model.l) 
        for l in self.model.l: 
            self.model.Ws[l].W = pyo.Var(
                {i for i in range(1, self.layer_sizes[l-1] + 1)}, # self.layer_sizes is 0-indexed list 
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

        #------ INITIAL CONDITION CONSTRAINTS (one per trajectory) ------
        # Each trajectory's first node (global index offset+1, offset = s0*N) is pinned to that
        # trajectory's initial condition.
        self.model.initial_condition = pyo.ConstraintList()
        for s0 in range(self.num_traj):
            offset = s0 * self.num_colloc_nodes
            for d in self.model.d:
                self.model.initial_condition.add(self.model.Y_star[offset + 1, d] == self.Y_obs[offset, d-1])

        # --------- COLLOCATION CONSTRAINT --------
        # Local (per-trajectory) collocation differentiation block as a pyomo Param (N x N); it is
        # the diagonal block repeated for every trajectory in the block-diagonal operator.
        self.model.D_colloc = pyo.Param(
            self.model.nloc,
            self.model.nloc,
            domain=pyo.Reals,
            initialize=self.initialise_colloc_diff_matrix()
        )

        # Define LHS of ODE_colloc (dY_star/dt) as matrix of pyomo expressions -- BLOCK DIAGONAL:
        # the derivative at a global node uses ONLY its own trajectory's nodes.
        def ode_lhs_colloc_rule(model, g, d):
            s0 = (g - 1) // self.num_colloc_nodes      # trajectory (0-indexed)
            nloc = (g - 1) % self.num_colloc_nodes + 1 # local node within the trajectory (1-indexed)
            offset = s0 * self.num_colloc_nodes
            return sum(self.model.Y_star[offset + j, d] * self.model.D_colloc[nloc, j] for j in model.nloc)
        self.model.ode_lhs_colloc = pyo.Expression(
            self.model.n,
            self.model.d,
            rule=ode_lhs_colloc_rule
        )

        # Compute output of NN for collocation points as a ndarray of Pyomo expressions
        nn_output_colloc = self.nn_feedforward_pyo(self.model.Y_star, self.total_nodes) # (total_nodes, state_dim) ndarray of pyomo expressions

        #----- INTEGRATED RESIDUAL OBJECTIVE -------
        if self.transcription_method == 'irrdc':
            # Initialise the interpolant matrix for residual eval as a pyomo Param
            self.model.L_res = pyo.Param(
                self.model.m, 
                self.model.n, 
                domain=pyo.Reals, 
                initialize=self.initialise_interpolant_eval_matrix()
            )

            # Initialise the differentiation matrix for residual eval as a pyomo Param
            self.model.D_res = pyo.Param(
                self.model.m,  
                self.model.n, 
                domain=pyo.Reals, 
                initialize=self.initialise_res_diff_matrix()
            )

            # Define LHS of ODE_res (dinterpolant/dt) as matrix of pyomo expressions
            self.model.ode_lhs_res = pyo.Expression(
                self.model.m, 
                self.model.d, 
                rule=lambda model, m, d: sum(self.model.D_res[m, j] * self.model.Y_star[j, d] for j in model.n)
            )

            # Define interpolant of Y_star at residual evaluation points as matrix of pyomo expressions 
            self.model.Y_res = pyo.Expression(
                self.model.m, 
                self.model.d, 
                rule=lambda model, m, d: sum(self.model.L_res[m, j] * self.model.Y_star[j, d] for j in model.n)
            )

            # Compute output of NN at residual evaluation points as a ndarray of Pyomo expressions
            nn_output_res = self.nn_feedforward_pyo(self.model.Y_res, self.num_res_eval_nodes) # (num_res_eval_nodes, state_dim) ndarray of pyomo expressions
        
        else:
            nn_output_res = None

        #---- ADMM CONSENUS PARAMETERS AND DUAL VARIABLES -----
        if self.admm_submodel:
            # ADMM penalty parameter as a mutable scalar Param so it can be adapted
            # each ADMM iteration (residual balancing) and broadcast into the submodel objective.
            self.model.admm_penalty_param = pyo.Param(initialize=self.admm_penalty_param, mutable=True)

            # Make ADMM consensus params and dual vars mutable Pyomo Params
            self.model.consensus_Ws = pyo.Block(self.model.l)
            self.model.consensus_bs = pyo.Block(self.model.l) 
            self.model.dual_var_Ws = pyo.Block(self.model.l)
            self.model.dual_var_bs = pyo.Block(self.model.l)

            for l in self.model.l: 
                self.model.consensus_Ws[l].W = pyo.Param(
                    {i for i in range(1, self.layer_sizes[l-1] + 1)},  
                    {j for j in range(1, self.layer_sizes[l] + 1)},
                    domain=pyo.Reals,
                    initialize=0.0,
                    mutable=True
                )
                self.model.consensus_bs[l].b = pyo.Param(
                    {1},
                    {i for i in range(1, self.layer_sizes[l] + 1)},
                    domain=pyo.Reals,
                    initialize=0.0,
                    mutable=True
                )
                self.model.dual_var_Ws[l].W = pyo.Param(
                    {i for i in range(1, self.layer_sizes[l-1] + 1)}, 
                    {j for j in range(1, self.layer_sizes[l] + 1)},
                    domain=pyo.Reals,
                    initialize=0.0,
                    mutable=True
                )
                self.model.dual_var_bs[l].b = pyo.Param(
                    {1},
                    {i for i in range(1, self.layer_sizes[l] + 1)}, 
                    domain=pyo.Reals,
                    initialize=0.0,
                    mutable=True
                )

        # Named Expression for the underlying DC objective (mse + l2 reg, no ADMM consensus
        # penalty), so it can be extracted from model after each solve.
        self.model.local_dc_objective = pyo.Expression(rule=lambda m: self.local_dc_objective_rule(m))

        # Build Pyomo objective function based on transcription method
        self.build_objective(nn_output_res)

        # Build Pyomo DC constraints 
        self.build_constraints(nn_output_colloc)

    def build_objective(self, nn_output_res):
        if self.transcription_method == 'dc':
            self.model.obj = pyo.Objective(rule=lambda m: self.dc_objective_rule(m), sense=pyo.minimize)

        elif self.transcription_method == 'irrdc' and nn_output_res is not None:
            self.model.obj = pyo.Objective(rule=lambda m: self.irrdc_objective_rule(m, nn_output_res), sense=pyo.minimize)

    def build_constraints(self, nn_output_colloc):
        # Define ODE equality constraint: dY_star/dt = NN(Y_star), block-diagonal over trajectories.
        # Skip each trajectory's first node (its initial condition is already enforced).
        self.model.ode = pyo.ConstraintList()
        for g in range(1, self.total_nodes + 1):
            if (g - 1) % self.num_colloc_nodes == 0:
                continue # first node of a trajectory -> initial condition, skip
            for d in self.model.d:
                self.model.ode.add(self.model.ode_lhs_colloc[g, d] == nn_output_colloc[g-1, d-1])


    # ----------------------------------- TRAINING METHODS ----------------------------------- #
    
    def nn_feedforward_pyo(self, Y, num_nodes): 
        '''
        Compute a feedforward pass of the NN (act func = tanh, linear output layer) 
        Y is a time invariant pyo Var or ndarray of pyo Expressions.
        Returns: output layer activations as an ndarray (i, j) of pyo.Expressions (one for each element of Y)
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
                    if l == 1: # if at first hidden layer 
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
        '''Freeze values of vars in pyomo model'''
        self.model.Y_star.fix() 
        for l in self.model.l:
            self.model.Ws[l].W.fix()
            self.model.bs[l].b.fix()

    def solve_model(self, solver_options: dict):
        '''Solve the Pyomo model using the IPOPT solver.'''
        # set up IPOPT solver
        solver = pyo.SolverFactory("ipopt", executable="/usr/local/bin/ipopt")
        
        # Numeric run limits (not in the option file) come from solver_options; all other IPOPT
        # settings (linear_solver, nlp scaling, mu_strategy, print_level, HSL lib, ...) are read
        # from the option file in the repo root (CNODE/ipopt.opt).
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
        # Wrapped in try/except so a genuinely failed solve (no solution returned) keeps the
        # submodel's warm-start params instead of raising and crashing the multiprocessing pool.
        # note: with load_solutions=True Pyomo EMPTIES result.solution after loading, so success
        # must be judged from the solver status / termination condition, NOT len(result.solution).
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
            print(f"Submodel solve raised {type(exc).__name__}: {exc}; keeping warm-start params.")
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
            # Final objective IPOPT minimised this solve. Gated on solution_loaded: an
            # unloaded solve leaves warm-start params in the model, so an ungated read would
            # report the warm-start objective, not a real solve result.
            "objective": float(pyo.value(self.model.obj)) if solution_loaded else float("nan"),
            # Underlying DC objective (mse + l2 reg), without the ADMM consensus penalty.
            "local_dc_objective": float(pyo.value(self.model.local_dc_objective)) if solution_loaded else float("nan"),
        }

        if solution_loaded:
            if not self.admm_submodel:
                self.freeze_vars()
                print("\nSolve time {:.1f}s for {} iterations (termination={})\n".format(
                    solve_wall_time, iterations, termination))
        else:
            # No usable solution: the submodel keeps its warm-start (consensus) params
            # for this ADMM iteration rather than aborting the run.
            print("Submodel solve returned no usable solution "
                  "(status={}, termination={}); keeping warm-start params for this batch.".format(
                      status, termination))

    
    # ------------------------------ DIAGNOSTIC & MISC METHODS ------------------------------- #

    def pyomo_var_to_numpy(self, pyo_obj, shape):
        '''Copy pyo_obj (a Pyomo Var/Param/Expression) into a numpy ndarray.'''
        return np.array([[pyo.value(pyo_obj[i, j]) for j in range(1, shape[1]+1)] for i in range(1, shape[0]+1)])

    def convert_weights(self):
        return [self.pyomo_var_to_numpy(self.model.Ws[l].W, (self.layer_sizes[l-1], self.layer_sizes[l])) for l in self.model.l]

    def convert_biases(self):
        return [self.pyomo_var_to_numpy(self.model.bs[l].b, (1, self.layer_sizes[l])) for l in self.model.l]
    
    def check_param_values(self):
        '''Print summary statistics for NN parameters.'''
        for l in self.model.l:
            W = self.pyomo_var_to_numpy(self.model.Ws[l].W, (self.layer_sizes[l-1], self.layer_sizes[l]))
            b = self.pyomo_var_to_numpy(self.model.bs[l].b, (1, self.layer_sizes[l]))
            print(f"Layer {l}: W range [{W.min():.4f}, {W.max():.4f}], |W| mean {np.abs(W).mean():.4f}")
            print(f"Layer {l}: b range [{b.min():.4f}, {b.max():.4f}], |b| mean {np.abs(b).mean():.4f}")
