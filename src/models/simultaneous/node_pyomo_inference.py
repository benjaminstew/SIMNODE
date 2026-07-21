"""
Collocation-based inference engine for trained Neural ODEs.

Given a trained network theta* (raw numpy weights, e.g. a solved NODEPyomo instance or
the ADMM consensus parameters), this formulates an NLP for the test-horizon trajectory
Y_test by enforcing the collocation dynamics as HARD equality constraints

    D_test @ Y_test = F_theta*(Y_test, xi_test)

together with the initial condition. The objective is zero (pure feasibility) and the
square collocation system determines Y_test; optionally, observations Y_obs_test can be
fitted in the objective (the retrospective evaluation protocol of Shapovalova, Stewart &
Tsay 2026, JPC, Sec. 4.3.3).

Design notes
------------
* Decoupled: theta* enters as plain numpy arrays, so the trained weights are baked into
  the constraints as numeric constants (no Pyomo Params needed for fixed data).
* The network is evaluated with numpy `@` on the weight arrays applied to lists of Pyomo
  variables -- object-array broadcasting yields Pyomo expressions. Scales to any depth.
* Supports non-autonomous systems (time_invariant=False) and exogenous inputs, in the
  training input order [state, (time), (exogenous...)].
* The collocation grid is Chebyshev-Lobatto (non-uniform). predict() can interpolate the
  solved trajectory back onto an arbitrary output grid via the barycentric formula.
* 0-based indexing throughout to align directly with the numpy differentiation matrix.
"""
import time
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from scipy.integrate import solve_ivp

from src.models.simultaneous.direct_collocation import BarycentricInterpolation


def make_ipopt_solver(solver_options):
    '''Build an IPOPT solver configured like NODEPyomo.solve_model.'''
    solver = pyo.SolverFactory("ipopt", executable="/opt/homebrew/bin/ipopt")
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
    return solver


class CollocationInference:
    """
    Collocation-based inference NLP for a trained Neural ODE.

    The trained parameters theta* are held fixed (numeric constants); the discretised
    test-horizon trajectory Y_test is the single decision variable, subject to the
    collocation dynamics (hard equality constraints) and the initial condition.
    """

    def __init__(self,
            Ws, bs, layer_sizes, state_dim,
            end_time, num_colloc_nodes,
            state_lower_bound, state_upper_bound,
            time_invariant=True):
        '''
        Args:
            Ws, bs: trained network parameters as lists of numpy arrays.
                    Ws[l] has shape (layer_sizes[l], layer_sizes[l+1]); bs[l] has shape
                    (1, layer_sizes[l+1]) or (layer_sizes[l+1],).
            layer_sizes: list of layer widths, e.g. [input_dim, w, state_dim].
            state_dim: state dimension d.
            end_time: end of the test horizon T (interval [0, T]).
            num_colloc_nodes: number of collocation nodes on the test horizon.
            state_lower_bound, state_upper_bound: bounds on Y_test.
            time_invariant: if False, node time is appended to the network input.
        '''
        self.Ws = [np.asarray(W, dtype=float) for W in Ws]
        self.bs = [np.asarray(b, dtype=float).reshape(-1) for b in bs]  # flatten to (out,)
        self.layer_sizes = layer_sizes
        self.state_dim = state_dim
        self.end_time = end_time
        self.num_colloc_nodes = num_colloc_nodes
        self.state_lower_bound = state_lower_bound
        self.state_upper_bound = state_upper_bound
        self.time_invariant = time_invariant

        # Chebyshev-Lobatto grid + differentiation matrix for the test horizon.
        bary = BarycentricInterpolation(0, self.end_time, self.num_colloc_nodes)
        self.colloc_grid = bary.colloc_grid          # xi_test, ascending, (N,)
        self.bary_ws = bary.bary_ws                   # barycentric weights, (N,)
        self.D = bary.D_colloc                        # D_test, (N, N)

        # Expected exogenous-input width, inferred from the first weight matrix.
        base_dim = self.state_dim + (0 if self.time_invariant else 1)
        self.num_exog = self.Ws[0].shape[0] - base_dim
        if self.num_exog < 0:
            raise ValueError(
                f"First weight matrix expects input dim {self.Ws[0].shape[0]}, smaller "
                f"than state_dim (+time) = {base_dim}."
            )

        self.model = None
        self.last_solve_info = None

    @classmethod
    def from_trained(cls, trained, end_time=None, num_colloc_nodes=None, time_invariant=True):
        '''Construct an inference engine from a solved NODEPyomo instance.'''
        return cls(
            Ws=trained.convert_weights(),
            bs=trained.convert_biases(),
            layer_sizes=trained.layer_sizes,
            state_dim=trained.state_dim,
            end_time=end_time if end_time is not None else trained.end_time,
            num_colloc_nodes=num_colloc_nodes if num_colloc_nodes is not None else trained.num_colloc_nodes,
            state_lower_bound=trained.state_lower_bound,
            state_upper_bound=trained.state_upper_bound,
            time_invariant=time_invariant,
        )

    # ------------------------------ NN EVALUATION ------------------------------- #

    def _nn_output(self, nn_input):
        '''
        Feedforward pass (tanh hidden layers, linear output) over a single input.
        nn_input entries may be Pyomo variables/expressions or floats; numpy `@` yields
        Pyomo expressions. Returns: object ndarray of length state_dim.
        '''
        activation = np.asarray(nn_input, dtype=object)
        n_layers = len(self.Ws)
        for l, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = np.dot(activation, W) + b                      # (out,)
            if l < n_layers - 1:                               # hidden -> tanh
                activation = np.array([pyo.tanh(zi) for zi in z], dtype=object)
            else:                                              # output -> linear
                activation = z
        return activation

    def _nn_forward_numeric(self, x):
        '''Pure-numpy forward pass, for the sequential warm-start rollout.'''
        a = np.asarray(x, dtype=float)
        n_layers = len(self.Ws)
        for l, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = a @ W + b
            a = np.tanh(z) if l < n_layers - 1 else z
        return a

    def _build_nn_input(self, y_components, t_value, exog_row):
        '''Assemble the network input in training order: [state, (time), (exogenous...)].'''
        nn_input = list(y_components)
        if not self.time_invariant:
            nn_input.append(t_value)
        if self.num_exog > 0:
            if exog_row is None:
                raise ValueError(f"Network expects {self.num_exog} exogenous inputs but none supplied.")
            nn_input.extend(exog_row)
        return nn_input

    # ------------------------------ MODEL BUILDING ------------------------------- #

    def _init_Y_test(self, Y_guess):
        '''Var init rule for Y_test (0-based) from a numpy warm-start guess.'''
        def init_rule(model, i, d):
            return float(Y_guess[i, d])
        return init_rule

    def _build_model(self, y_0, Y_guess, Y_obs_test, exog):
        '''Build the inference NLP for a given initial condition and warm start.'''
        self.model = pyo.ConcreteModel()
        N, d_dim = self.num_colloc_nodes, self.state_dim

        # ------------ SETS (0-based, aligned with numpy D) ------------
        self.model.t = pyo.RangeSet(0, N - 1)
        self.model.d = pyo.RangeSet(0, d_dim - 1)

        # ----- DECISION VARIABLE: Y_test -----
        self.model.Y = pyo.Var(
            self.model.t,
            self.model.d,
            domain=pyo.Reals,
            initialize=self._init_Y_test(Y_guess),
            bounds=(self.state_lower_bound, self.state_upper_bound),
        )

        # ----- INITIAL-CONDITION CONSTRAINT (hard) -----
        self.model.initial_condition = pyo.Constraint(
            self.model.d, rule=lambda m, d: m.Y[0, d] == float(y_0[d])
        )

        # ----- COLLOCATION CONSTRAINTS (hard; skip node 0, replaced by IC) -----
        self.model.ode = pyo.ConstraintList()
        for i in range(1, N):
            y_i = [self.model.Y[i, d] for d in self.model.d]
            exog_row = None if self.num_exog == 0 else exog[i, :]
            nn_input = self._build_nn_input(y_i, self.colloc_grid[i], exog_row)
            nn_out = self._nn_output(nn_input)                 # length state_dim
            for d in self.model.d:
                dy_dt = sum(self.D[i, j] * self.model.Y[j, d] for j in self.model.t)
                self.model.ode.add(dy_dt == nn_out[d])

        # ----- OBJECTIVE: feasibility (0) or data fit -----
        if Y_obs_test is None:
            self.model.obj = pyo.Objective(expr=0.0, sense=pyo.minimize)
        else:
            self.model.obj = pyo.Objective(
                rule=lambda m: (1.0 / N) * sum(
                    (m.Y[i, d] - float(Y_obs_test[i, d]))**2
                    for i in m.t for d in m.d
                ),
                sense=pyo.minimize,
            )

    # ------------------------------ WARM START ------------------------------- #

    def _warm_start_guess(self, y_0, exog):
        '''Cheap sequential RK45 rollout on the test grid to seed Y_test (solver-based init).'''
        if self.num_exog > 0:
            exog = np.asarray(exog, dtype=float)

            def exog_interp(tt):
                return [np.interp(tt, self.colloc_grid, exog[:, k]) for k in range(self.num_exog)]
        else:
            exog_interp = None

        def rhs(t, y):
            nn_input = list(y)
            if not self.time_invariant:
                nn_input.append(t)
            if exog_interp is not None:
                nn_input.extend(exog_interp(t))
            return self._nn_forward_numeric(nn_input)

        try:
            sol = solve_ivp(
                fun=rhs,
                t_span=(self.colloc_grid[0], self.colloc_grid[-1]),
                y0=np.asarray(y_0, dtype=float),
                t_eval=self.colloc_grid,
                method='RK45', rtol=1e-3, atol=1e-6,
            )
            if sol.success and sol.y.shape[1] == self.num_colloc_nodes:
                return sol.y.T
        except Exception:
            pass
        # Fall back to a constant y_0 guess (e.g. if theta* gives an unstable field).
        return np.tile(np.asarray(y_0, dtype=float).reshape(1, -1), (self.num_colloc_nodes, 1))

    # ------------------------------ INTERPOLATION ------------------------------- #

    def _bary_interp(self, Y_nodes, t_query):
        '''
        Barycentric interpolation of the solved nodal trajectory onto t_query.

        Y_nodes: (num_colloc_nodes, state_dim) on self.colloc_grid.
        t_query: 1D array of output times within [0, end_time].
        Returns: (len(t_query), state_dim).
        '''
        x, w = self.colloc_grid, self.bary_ws
        t_query = np.asarray(t_query, dtype=float)
        out = np.empty((t_query.shape[0], Y_nodes.shape[1]), dtype=float)
        for k, tq in enumerate(t_query):
            match = np.isclose(tq, x, atol=1e-12)
            if match.any():                      # query coincides with a node
                out[k] = Y_nodes[np.argmax(match)]
                continue
            coeff = w / (tq - x)                 # barycentric weights / (tq - x_j)
            out[k] = (coeff @ Y_nodes) / coeff.sum()
        return out

    # ------------------------------ PREDICTION ------------------------------- #

    def predict(self, y_0, solver_options, Y_obs_test=None, exog=None,
                warm_start=True, output_grid=None):
        '''
        Solve the collocation inference NLP and return the predicted trajectory.

        Args:
            y_0: initial state, shape (state_dim,).
            solver_options: IPOPT options dict (same keys as NODEPyomo.solve_model).
            Y_obs_test: optional observations on the collocation grid, shape
                        (num_colloc_nodes, state_dim). If given, fitted in the objective;
                        otherwise pure feasibility.
            exog: exogenous inputs aligned to the collocation grid, shape
                  (num_colloc_nodes, num_exog). Required iff the network expects exog.
            warm_start: if True, initialise Y_test from a sequential RK45 rollout.
            output_grid: optional 1D array of output times. If given, the solved nodal
                         trajectory is barycentric-interpolated onto this grid; otherwise
                         the result is returned on the (Chebyshev) collocation grid.

        Returns:
            Predicted trajectory as ndarray; shape (num_colloc_nodes, state_dim) or
            (len(output_grid), state_dim) if output_grid is given.
        '''
        y_0 = np.asarray(y_0, dtype=float).reshape(-1)
        if y_0.shape[0] != self.state_dim:
            raise ValueError(f"y_0 has length {y_0.shape[0]} but state_dim is {self.state_dim}.")
        if Y_obs_test is not None:
            Y_obs_test = np.asarray(Y_obs_test, dtype=float)
            if Y_obs_test.shape != (self.num_colloc_nodes, self.state_dim):
                raise ValueError(
                    f"Y_obs_test must have shape {(self.num_colloc_nodes, self.state_dim)}; "
                    f"got {Y_obs_test.shape}."
                )
        if self.num_exog > 0:
            if exog is None:
                raise ValueError(f"Network expects exog of shape (num_colloc_nodes, {self.num_exog}).")
            exog = np.asarray(exog, dtype=float)
            if exog.shape != (self.num_colloc_nodes, self.num_exog):
                raise ValueError(
                    f"exog must have shape {(self.num_colloc_nodes, self.num_exog)}; got {exog.shape}."
                )

        if warm_start:
            Y_guess = self._warm_start_guess(y_0, exog)
        else:
            Y_guess = np.tile(y_0.reshape(1, -1), (self.num_colloc_nodes, 1))

        self._build_model(y_0, Y_guess, Y_obs_test, exog)

        solver = make_ipopt_solver(solver_options)
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

        if result.solver.status == SolverStatus.ok and (
            result.solver.termination_condition == TerminationCondition.optimal
        ):
            print("Inference solve feasible and optimal")
        else:
            print("Inference solve not feasible and/or not optimal: {}".format(str(result.solver)))

        Y_nodes = np.array([
            [pyo.value(self.model.Y[i, d]) for d in range(self.state_dim)]
            for i in range(self.num_colloc_nodes)
        ])

        if output_grid is None:
            return Y_nodes
        return self._bary_interp(Y_nodes, output_grid)
