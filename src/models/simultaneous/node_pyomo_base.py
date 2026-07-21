# currently lacks customisation on the numerical analysis methods used, interpolation scheme used, param initialisation methods, etc...
from pyexpat import model
import warnings
import numpy as np 
import time
from pathlib import Path
from scipy.integrate import solve_ivp
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from src.models.simultaneous.direct_collocation import BarycentricInterpolation

class NODEPyomo:
    """
    Implementation of a base Pyomo model to train collocation-based Neural ODEs using the IPOPT solver. 
    I have tried to make the code as scalable as possible with NN size etc., but there are still may improvements to be made.
    """
    def __init__(self, 
            Y_obs, layer_sizes, end_time, 
            state_lower_bound, state_upper_bound, 
            param_lower_bound, param_upper_bound,
            lambda_reg, 
            Y_smooth=None, 
            transcription_method='dc',
            rho_reg=None,
            num_res_eval_nodes=None, 
    ):
        self.Y_obs = Y_obs # observed state trajectory from measurements. ndarray (num_nodes, state_dim)
        self.num_colloc_nodes, self.state_dim = self.Y_obs.shape
        if self.state_dim > self.num_colloc_nodes:
            warnings.warn("Y_obs should be structured such that each row represents a new collocation point.")
        self.end_time = end_time

        bary = BarycentricInterpolation(
            0, self.end_time,
            self.num_colloc_nodes,
            transcription_method=transcription_method,
            num_res_eval_nodes=num_res_eval_nodes
        )
        self.colloc_grid = bary.colloc_grid
        self.bary_ws = bary.bary_ws
        self.D_colloc = bary.D_colloc # ndarray (num_colloc_nodes, num_colloc_nodes)

        self.layer_sizes = layer_sizes
        self.model = pyo.ConcreteModel()
        self.state_lower_bound = state_lower_bound
        self.state_upper_bound = state_upper_bound
        self.param_lower_bound = param_lower_bound
        self.param_upper_bound = param_upper_bound
        self.lambda_reg = lambda_reg # regularisation hyperparameter for loss function 
        self.Y_smooth = Y_smooth if Y_smooth is not None else Y_obs # noise-free trajectory for Y_star init
        
        self.transcription_method = transcription_method

        if self.transcription_method == 'irrdc': 
            if rho_reg is None or num_res_eval_nodes is None:
                raise ValueError("Need to set hyperparameter rho_reg and num_res_eval_nodes for IRRDC method.")
            self.num_res_eval_nodes = num_res_eval_nodes
            self.res_eval_grid = bary.res_eval_grid
            self.L_res = bary.L_res
            self.D_res = bary.D_res
            self.rho_reg = rho_reg # regularisation hyperparameter for residual penalty term 
            self.quadrature_ws = bary.quadrature_ws # quadrature weights for residual penalty term
        
        self.build_model()

    # ------------------------------ BASE MODEL BUILDING METHODS ------------------------------- #

    def initialise_weight(self, shape):
        '''Xavier initialisation of a weight matrix as pyomo Var'''
        W = np.random.randn(*shape) * np.sqrt(2 / (shape[0] + shape[1]))
        def init_rule(model, i, j):
            return W[i-1, j-1] # W is a 0-indexed ndarray 
        return init_rule

    def initialise_bias(self, shape):
        '''Normal random initialisation of a bias vector as pyomo Var'''
        b = np.random.randn(*shape) * 0.1
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
        if self.Y_smooth is None:
            def init_rule(model, m, d):
                return self.Y_obs[m-1, d-1] * np.random.randn() * 0.1 # self.Y_obs is a 0-indexed ndarray
        else:
            def init_rule(model, m, d):
                if self.Y_smooth is not None:
                    return self.Y_smooth[m-1, d-1] # self.Y_smooth is a 0-indexed ndarray
        return init_rule
    
    def dc_objective_rule(self, model):
        '''Returns the objective function for DC as a pyomo expression.'''
        # MSE term: (1/N) * ||Y_star - Y_obs||_F^2
        mse = (1 / self.num_colloc_nodes) * sum(
            (model.Y_star[n, d] - self.Y_obs[n-1, d-1])**2
            for n in model.n 
            for d in model.d
        ) 

        # Regularised L2 norm term: lambda * ||theta||_2^2
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

        reg_l2_norm = self.lambda_reg * l2_norm

        return mse + reg_l2_norm
    
    def irrdc_objective_rule(self, model, nn_output_res):
        '''Returns the objective function for IRR-DC'''
        # Regularised MSE term: (1/N) * ||Y_star - Y_obs||_F^2 + lambda * ||theta||_2^2
        reg_mse = self.dc_objective_rule(model) 

        # Residual penalty term: (1 / (2 * rho)) * sum_n(w_m * ||interpolant_m||_2^2) 
        residual_penalty = (1 / (2 * self.rho_reg)) * sum(
            self.quadrature_ws[m-1] * (model.ode_lhs_res[m, d] - nn_output_res[m-1, d-1])**2
            for m in model.m
            for d in model.d
        )
        
        return reg_mse + residual_penalty

    def build_model(self):
  
        # ------------ SETS ------------
        self.model.l = pyo.RangeSet(len(self.layer_sizes) - 1) # [1:len(self.layer_sizes)-1] as for n layers, have n-1 W matrices/b vectors
        self.model.n = pyo.RangeSet(self.num_colloc_nodes) 
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

        #------ INITIAL CONDITION CONSTRAINT ------
        self.model.initial_condition = pyo.ConstraintList()
        for d in self.model.d:
            self.model.initial_condition.add(self.model.Y_star[1, d] == self.Y_obs[0, d-1])
        
        # --------- COLLOCATION CONSTRAINT --------
        # Initialise the collocation-estimated differentiation matrix as a pyomo Param
        self.model.D_colloc = pyo.Param(
            self.model.n,  
            self.model.n, 
            domain=pyo.Reals, 
            initialize=self.initialise_colloc_diff_matrix()
        )

        # Define LHS of ODE_colloc (dY_star/dt) as matrix of pyomo expressions 
        self.model.ode_lhs_colloc = pyo.Expression( 
            self.model.n, 
            self.model.d, 
            rule=lambda model, n, d: sum(self.model.Y_star[j, d] * self.model.D_colloc[n, j] for j in model.n) 
        )

        # Compute output of NN for collocation points as a ndarray of Pyomo expressions 
        nn_output_colloc = self.nn_feedforward_pyo(self.model.Y_star, self.num_colloc_nodes) # (num_colloc_nodes, state_dim) ndarray of pyomo expressions

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

        # Build Pyomo objective function based on transcription method used 
        self.build_objective(nn_output_res)

        # Build Pyomo constraints based on transcription method used
        self.build_constraints(nn_output_colloc)

    def build_objective(self, nn_output_res):
        if self.transcription_method == 'dc':
            self.model.obj = pyo.Objective(rule=lambda m: self.dc_objective_rule(m), sense=pyo.minimize)

        elif self.transcription_method == 'irrdc' and nn_output_res is not None:
            self.model.obj = pyo.Objective(rule=lambda m: self.irrdc_objective_rule(m, nn_output_res), sense=pyo.minimize)

    def build_constraints(self, nn_output_colloc):
        # Define ODE equality constraint: dY_star/dt = NN(Y_star)
        self.model.ode = pyo.ConstraintList()
        for n in range(2, self.num_colloc_nodes + 1): # skip n=1 (initial condition already enforced)
            for d in self.model.d:
                self.model.ode.add(self.model.ode_lhs_colloc[n, d] == nn_output_colloc[n-1, d-1])


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
        '''Fix values of vars in pyomo model'''
        self.model.Y_star.fix() 
        for l in self.model.l:
            self.model.Ws[l].W.fix()
            self.model.bs[l].b.fix()

    def solve_model(self, solver_options):
        '''
        Solve the Pyomo model using the IPOPT solver.
        
        Args:
            solver_options: dictionary of solver options specific to the transcription method used.
        '''
        # set up IPOPT solver
        #solver = pyo.SolverFactory("ipopt", executable="/usr/local/bin/ipopt")
        solver = pyo.SolverFactory("ipopt", executable="/opt/homebrew/bin/ipopt")
        #print("Solver available?: {}".format(solver.available())) 
        #print("\nInitial NN parameter summary stats (pre-IPOPT): ")
        #self.check_param_values()
        
        # set global solver options
        solver.options['max_iter'] = solver_options["max_iter"]
        solver.options['nlp_scaling_method'] = solver_options["nlp_scaling_method"]
        solver.options['mu_strategy'] = solver_options["mu_strategy"]
        solver.options['tol'] = solver_options["tol"] 
        solver.options['acceptable_tol'] = solver_options["acceptable_tol"]
        solver.options['acceptable_iter'] = solver_options["acceptable_iter"]
        repo_root = Path(__file__).resolve().parents[2]
        option_path = repo_root / "ipopt.opt"
        if option_path.exists():
            solver.options["option_file_name"] = str(option_path)

        t0 = time.perf_counter()
        result = solver.solve(self.model, tee=True)
        solve_wall_time = time.perf_counter() - t0

        iterations = getattr(result.solver, "iterations", None)
        self.last_solve_info = {
            "status": str(result.solver.status),
            "termination_condition": str(result.solver.termination_condition),
            "iterations": int(iterations) if iterations is not None else None,
            "solve_wall_time_s": solve_wall_time,
        }
            
        if result.solver.status == SolverStatus.ok and result.solver.termination_condition == TerminationCondition.optimal:
            print("Solution is feasible and optimal")
            self.freeze_vars()
            print("NN parameters have been frozen")
            #print("\nFinal NN parameter summary stats (post-IPOPT): ")
            #self.check_param_values()
        else:
            print("Solution is not feasible and/or not optimal: {}".format(str(result.solver)))

    
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
        y_0 = y_0.reshape(1, -1) # reshape to row vector
        out = np.tanh(np.dot(y_0, Ws[0]) + bs[0]) # first hidden layer
        for W, b in zip(Ws[1:-1], bs[1:-1]): 
            out = np.tanh(np.dot(out, W) + b) # calc activation for hidden layer, move forward
        out = np.dot(out, Ws[-1]) + bs[-1] # linear output layer
        
        return out.flatten() # return 1D array for solve_ivp

    def get_predicted_trajectory(self, 
            y_0, t_grid,
            method: str = 'RK45',
            rtol: float = 1e-3,
            atol: float = 1e-6,
            max_step: float = np.inf
    ):
        '''
        Predict state trajectory over t_grid via forward integration of the trained NN using scipy.solve_ivp.

        Args:
            y_0: Initial state, ndarray (state_dim,). Must be 1D for scipy.solve_ivp.
            t_grid: Strictly increasing time grid, ndarray (num_nodes, ).
            method: solve_ivp method (e.g. 'RK45', 'DOP853', 'Radau', 'BDF').
            rtol, atol, max_step: solve_ivp tolerances/step control.

        Returns:
            Predicted state trajectory as ndarray (num_nodes, state_dim).
        '''
        if t_grid.ndim != 1 or t_grid.size < 2:
            raise ValueError(f"t_grid must be a 1D array with >=2 points; got shape {t_grid.shape}.")
        if not np.all(np.isfinite(t_grid)):
            raise ValueError("t_grid must contain only finite values.")
        if np.any(np.diff(t_grid) <= 0):
            raise ValueError("t_grid must be strictly increasing (solve_ivp expects monotone time).")

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



