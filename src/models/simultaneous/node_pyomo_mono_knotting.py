"""
Monolithic training of collocation-based Neural ODE over B independent trajectories, with pseudospectral
(PS) knotting at known discontinuities (Ross & Fahroo, JGCD 27(3), 2004).

Each trajectory is split at its feed times tau_1 < ... < tau_K (known a priori)into K+1 phases. Every phase gets
its own collocation grid. The feed events are 'hard knots': the last node of phase p-1 (x^-) and the first node of phase p (x^+) sit at the same
time tau_p but are distinct variables, linked by the event (knotting) condition - x^+ = x^- + a_p e_G.        
"""
import time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.common.errors import ApplicationError
from src.models.simultaneous.barycentric_lagrange_collocation import BarycentricInterpolation
from data.bioreactor_simulation.raw_data.fedbatch_loader import build_phase_blocks

class KnottedNODEPyomo:
    """
    Monolithic Pyomo model to train a single collocation-based Neural ODE jointly over B
    trajectories with the IPOPT solver, with a hard knot at every feed event.
    """
    def __init__(self,
            trajs: list,
            num_colloc_nodes: int,
            layer_sizes: list,
            state_lower_bound: float, state_upper_bound: float,
            param_lower_bound: float, param_upper_bound: float,
            l2_reg_param: float,
            jump_state_idx: int = 0,
            end_time: float = None,
            seed: int = None,
            transcription_method: str = 'dc',
            irrdc_param: float = None,
            num_res_eval_nodes: int = None
    ):
        '''
        Args:
            trajs: list of trajectory dicts from feed_data.load_feed_trajectories(), each with
                'obs_times' (n_b,), 'Y_obs' (n_b, D), 'y0' (D,), 'feed_times' (K_b,), 'feed_amounts' (K_b,)
            num_colloc_nodes: Chebyshev-Lobatto nodes PER PHASE (N)
            jump_state_idx: 0-based column of Y_obs that the bolus is added to (G)
            end_time: horizon end T (default: each trajectory's last observation time)
            transcription_method: 'dc' or 'irrdc'; irrdc needs irrdc_param and num_res_eval_nodes
                (residual-evaluation nodes PER PHASE)
        '''
        self.rng = np.random.default_rng(seed)
        self.trajs = trajs
        self.num_trajs = len(trajs)
        if self.num_trajs == 0:
            raise ValueError("trajs is empty.")
        self.state_dim = trajs[0]['Y_obs'].shape[1]
        for tr in trajs:
            if tr['Y_obs'].ndim != 2 or tr['Y_obs'].shape[1] != self.state_dim:
                raise ValueError(f"{tr.get('batch')}: Y_obs must be (n_obs, {self.state_dim}); got {tr['Y_obs'].shape}.")
            if tr['obs_times'].shape[0] != tr['Y_obs'].shape[0]:
                raise ValueError(f"{tr.get('batch')}: obs_times and Y_obs disagree on the number of observations.")
        self.num_colloc_nodes = num_colloc_nodes
        if num_colloc_nodes < 3:
            raise ValueError("num_colloc_nodes (per phase) must be at least 3.")
        if not 0 <= jump_state_idx < self.state_dim:
            raise ValueError(f"jump_state_idx must be in [0, {self.state_dim}); got {jump_state_idx}.")
        self.jump_state_idx = jump_state_idx
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
        self.transcription_method = transcription_method
        if self.transcription_method not in ('dc', 'irrdc'):
            raise ValueError(f"transcription_method must be 'dc' or 'irrdc'; got {transcription_method!r}.")
        if self.transcription_method == 'irrdc':
            if irrdc_param is None or num_res_eval_nodes is None:
                raise ValueError("Need to set hyperparameter irrdc_param and num_res_eval_nodes for irrdc method.")
            self.irrdc_param = irrdc_param
            self.num_res_eval_nodes = num_res_eval_nodes

        # ---- Phase blocks: one per (trajectory, phase), stacked trajectory by trajectory ----
        self.blocks = build_phase_blocks(trajs, T=end_time)
        self.num_blocks = len(self.blocks)
        self.total_nodes = self.num_blocks * self.num_colloc_nodes
        if self.transcription_method == 'irrdc':
            self.total_res_nodes = self.num_blocks * self.num_res_eval_nodes

        # ---- Per-block collocation operators on [t0, t1] (D scales with 2/(t1-t0), so one per block) ----
        self.colloc_grids = []   # (N,) per block
        self.bary_ws = None      # (N,) same for every block (depends only on the node pattern)
        self.D_colloc = []       # (N, N) per block
        if self.transcription_method == 'irrdc':
            self.res_eval_grids, self.L_res, self.D_res, self.quadrature_ws = [], [], [], []
        for blk in self.blocks:
            bary = BarycentricInterpolation(
                blk['t0'], blk['t1'], self.num_colloc_nodes,
                transcription_method=self.transcription_method,
                num_res_eval_nodes=num_res_eval_nodes if self.transcription_method == 'irrdc' else None
            )
            self.colloc_grids.append(bary.colloc_grid)
            self.bary_ws = bary.bary_ws
            self.D_colloc.append(bary.D_colloc)
            if self.transcription_method == 'irrdc':
                self.res_eval_grids.append(bary.res_eval_grid)
                self.L_res.append(bary.L_res)
                self.D_res.append(bary.D_res)
                self.quadrature_ws.append(bary.quadrature_ws)  # sums to t1 - t0 for this phase

        # ---- Observation terms: barycentric evaluation of each phase's interpolant at its observation times ----
        # self.obs_terms[b] = list of (node_offset, L_row (N,), y_obs (D,)) for trajectory b
        self.obs_terms = [[] for _ in range(self.num_trajs)]
        for k, blk in enumerate(self.blocks):
            tr = trajs[blk['traj']]
            for i in blk['obs_idx']:
                L_row = self._bary_eval_row(self.colloc_grids[k], tr['obs_times'][i])
                self.obs_terms[blk['traj']].append((k * self.num_colloc_nodes, L_row, tr['Y_obs'][i]))
        self.num_obs = [len(terms) for terms in self.obs_terms]
        if any(n == 0 for n in self.num_obs):
            raise ValueError("Every trajectory needs at least one observation.")

        # ---- Initial guess for Y_star: phase-wise linear interpolation of the data (never across a feed) ----
        self.Y_init = self._initial_trajectory_guess()

        self.model = pyo.ConcreteModel()
        self.build_model()

    # ------------------------------ GRID / DATA HELPERS ------------------------------- #

    def _bary_eval_row(self, grid, t):
        '''Row of the barycentric interpolant-evaluation operator at time t for a phase grid:
        y(t) = L_row . Y_phase. One-hot if t coincides with a node (covers observations at knots).'''
        hit = np.flatnonzero(np.isclose(t, grid, rtol=0, atol=1e-12))
        row = np.zeros(len(grid))
        if hit.size:
            row[hit[0]] = 1.0
        else:
            c = self.bary_ws / (t - grid)
            row = c / c.sum()
        return row

    def _initial_trajectory_guess(self):
        '''(total_nodes, D) initial values for Y_star. Each phase is initialised by linear interpolation
        of that phase's own observations, anchored at the phase's start value: y0 for phase 0, and the
        previous phase's end guess plus the bolus for later phases. So the guess already satisfies the
        initial-condition and knot constraints and never interpolates across a jump.'''
        N, D = self.num_colloc_nodes, self.state_dim
        Y_init = np.empty((self.total_nodes, D))
        for k, blk in enumerate(self.blocks):
            tr = self.trajs[blk['traj']]
            if blk['prev'] is None:
                start = tr['y0'].astype(float).copy()
            else:
                start = Y_init[blk['prev'] * N + N - 1].copy()
                start[self.jump_state_idx] += blk['jump']
            idx = [i for i in blk['obs_idx'] if tr['obs_times'][i] > blk['t0']]
            t_pts = np.concatenate([[blk['t0']], tr['obs_times'][idx]])
            y_pts = np.vstack([start, tr['Y_obs'][idx]]) if idx else start[None, :]
            grid = self.colloc_grids[k]
            for d in range(D):
                Y_init[k * N:(k + 1) * N, d] = np.interp(grid, t_pts, y_pts[:, d])  # constant beyond last obs
        return np.clip(Y_init, self.state_lower_bound, self.state_upper_bound)

    # ------------------------------ BASE MODEL BUILDING METHODS ------------------------------- #

    def initialise_weight(self, shape):
        '''Xavier initialisation of a weight matrix as pyomo Var'''
        W = self.rng.standard_normal(shape) * np.sqrt(2 / (shape[0] + shape[1]))
        def init_rule(model, i, j):
            return W[i-1, j-1]
        return init_rule

    def initialise_bias(self, shape):
        '''Zero initialisation of a bias vector as pyomo Var'''
        b = np.zeros(shape)
        def init_rule(model, i, j):
            return b[0, i-1]
        return init_rule

    def initialise_colloc_diff_matrix(self):
        '''Per-block (N x N) barycentric differentiation matrices as one pyomo Param indexed (block, n, n').'''
        def init_rule(model, k, n, n_prime):
            return self.D_colloc[k-1][n-1, n_prime-1]
        return init_rule

    def initialise_interpolant_eval_matrix(self):
        def init_rule(model, k, m, n):
            return self.L_res[k-1][m-1, n-1]
        return init_rule

    def initialise_res_diff_matrix(self):
        def init_rule(model, k, m, n):
            return self.D_res[k-1][m-1, n-1]
        return init_rule

    def initialise_quadrature_weights(self):
        def init_rule(model, k, m):
            return self.quadrature_ws[k-1][m-1]
        return init_rule

    def initialise_Y_star(self):
        def init_rule(model, g, d):
            return self.Y_init[g-1, d-1]
        return init_rule

    def single_traj_objective_rule(self, model, b):
        '''L_b(Y_b, theta) for trajectory b:
            (1/n_obs_b) sum_obs ||y~(t_obs) - y_obs||^2  +  lambda ||theta||_2^2
        where y~(t_obs) is the phase interpolant evaluated at the observation time through its
        barycentric row L. Reduces to the monolithic DC objective when observations sit on nodes.
        The L2 term is added once per trajectory, matching node_pyomo_monolithic.py.'''
        N = self.num_colloc_nodes
        mse = (1 / self.num_obs[b]) * sum(
            (sum(L_row[j-1] * model.Y_star[offset + j, d] for j in model.nloc if L_row[j-1] != 0.0)
             - y_obs[d-1])**2
            for (offset, L_row, y_obs) in self.obs_terms[b]
            for d in model.d
        )
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
        return sum(self.single_traj_objective_rule(model, b) for b in range(self.num_trajs))

    def irrdc_objective_rule(self, model, nn_output_res):
        '''IRR-DC objective: DC objective + (1/(2 mu)) sum_blocks sum_m w_m ||r_m||^2, with the residual
        integral taken phase by phase (the weights of block k sum to that phase's length).'''
        reg_mse = self.dc_objective_rule(model)
        M = self.num_res_eval_nodes
        residual_penalty = (1 / (2 * self.irrdc_param)) * sum(
            model.quad_ws[(r - 1) // M + 1, (r - 1) % M + 1] * (model.ode_lhs_res[r, d] - nn_output_res[r-1, d-1])**2
            for r in model.m
            for d in model.d
        )
        return reg_mse + residual_penalty

    def build_model(self):

        # ------------ SETS ------------
        self.model.l = pyo.RangeSet(len(self.layer_sizes) - 1)
        self.model.k = pyo.RangeSet(self.num_blocks)            # phase blocks (all trajectories)
        self.model.n = pyo.RangeSet(self.total_nodes)           # ALL collocation nodes across every block
        self.model.d = pyo.RangeSet(self.state_dim)
        self.model.nloc = pyo.RangeSet(self.num_colloc_nodes)   # local node index within a block
        if self.transcription_method == 'irrdc':
            self.model.m = pyo.RangeSet(self.total_res_nodes)
            self.model.mloc = pyo.RangeSet(self.num_res_eval_nodes)

        # ----- DECISION VARIABLES -----
        self.model.Y_star = pyo.Var(
            self.model.n,
            self.model.d,
            domain=pyo.Reals,
            initialize=self.initialise_Y_star(),
            bounds=(self.state_lower_bound, self.state_upper_bound)
        )

        # Shared NN parameters (one f_theta for all trajectories and phases)
        self.model.Ws = pyo.Block(self.model.l)
        self.model.bs = pyo.Block(self.model.l)
        for l in self.model.l:
            self.model.Ws[l].W = pyo.Var(
                {i for i in range(1, self.layer_sizes[l-1] + 1)},
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

        # ------ INITIAL CONDITION + KNOT CONSTRAINTS (one per block, on the block's first node) ------
        # phase 0:  Y[first] = y0                      (hard knot at t0, one-sided)
        # phase p:  Y[first] = Y[prev block, last] + a_p e_G   (fixed hard knot at the feed time)
        N = self.num_colloc_nodes
        self.model.initial_condition = pyo.ConstraintList()
        self.model.knot = pyo.ConstraintList()
        for k, blk in enumerate(self.blocks):
            first = k * N + 1
            if blk['prev'] is None:
                y0 = self.trajs[blk['traj']]['y0']
                for d in self.model.d:
                    self.model.initial_condition.add(self.model.Y_star[first, d] == float(y0[d-1]))
            else:
                last_prev = blk['prev'] * N + N
                for d in self.model.d:
                    jump = blk['jump'] if (d - 1) == self.jump_state_idx else 0.0
                    self.model.knot.add(self.model.Y_star[first, d] == self.model.Y_star[last_prev, d] + jump)

        # --------- COLLOCATION CONSTRAINT --------
        self.model.D_colloc = pyo.Param(
            self.model.k, self.model.nloc, self.model.nloc,
            domain=pyo.Reals,
            initialize=self.initialise_colloc_diff_matrix()
        )

        # dY_star/dt at a global node uses ONLY its own block's nodes and that block's D (block diagonal)
        def ode_lhs_colloc_rule(model, g, d):
            k = (g - 1) // N + 1
            nloc = (g - 1) % N + 1
            offset = (k - 1) * N
            return sum(model.Y_star[offset + j, d] * model.D_colloc[k, nloc, j] for j in model.nloc)
        self.model.ode_lhs_colloc = pyo.Expression(self.model.n, self.model.d, rule=ode_lhs_colloc_rule)

        nn_output_colloc = self.nn_feedforward_pyo(self.model.Y_star, self.total_nodes)

        # ----- INTEGRATED RESIDUAL (per phase) -------
        if self.transcription_method == 'irrdc':
            M = self.num_res_eval_nodes
            self.model.L_res = pyo.Param(self.model.k, self.model.mloc, self.model.nloc, domain=pyo.Reals,
                                         initialize=self.initialise_interpolant_eval_matrix())
            self.model.D_res = pyo.Param(self.model.k, self.model.mloc, self.model.nloc, domain=pyo.Reals,
                                         initialize=self.initialise_res_diff_matrix())
            self.model.quad_ws = pyo.Param(self.model.k, self.model.mloc, domain=pyo.Reals,
                                           initialize=self.initialise_quadrature_weights())

            def ode_lhs_res_rule(model, r, d):
                k = (r - 1) // M + 1
                mloc = (r - 1) % M + 1
                offset = (k - 1) * N
                return sum(model.D_res[k, mloc, j] * model.Y_star[offset + j, d] for j in model.nloc)
            self.model.ode_lhs_res = pyo.Expression(self.model.m, self.model.d, rule=ode_lhs_res_rule)

            def Y_res_rule(model, r, d):
                k = (r - 1) // M + 1
                mloc = (r - 1) % M + 1
                offset = (k - 1) * N
                return sum(model.L_res[k, mloc, j] * model.Y_star[offset + j, d] for j in model.nloc)
            self.model.Y_res = pyo.Expression(self.model.m, self.model.d, rule=Y_res_rule)

            nn_output_res = self.nn_feedforward_pyo(self.model.Y_res, self.total_res_nodes)
        else:
            nn_output_res = None

        self.build_objective(nn_output_res)
        self.build_constraints(nn_output_colloc)

    def build_objective(self, nn_output_res):
        if self.transcription_method == 'dc':
            self.model.obj = pyo.Objective(rule=lambda m: self.dc_objective_rule(m), sense=pyo.minimize)
        elif self.transcription_method == 'irrdc' and nn_output_res is not None:
            self.model.obj = pyo.Objective(rule=lambda m: self.irrdc_objective_rule(m, nn_output_res), sense=pyo.minimize)

    def build_constraints(self, nn_output_colloc):
        '''ODE equality dY_star/dt = NN(Y_star) at every node EXCEPT each block's first node, which is
        fixed by the initial-condition or knot constraint. This keeps every phase square (N unknowns,
        1 + (N-1) equations per state); collocating at both nodes of a knot would be overdetermined.'''
        self.model.ode = pyo.ConstraintList()
        for g in range(1, self.total_nodes + 1):
            if (g - 1) % self.num_colloc_nodes == 0:
                continue
            for d in self.model.d:
                self.model.ode.add(self.model.ode_lhs_colloc[g, d] == nn_output_colloc[g-1, d-1])

    # ----------------------------------- TRAINING METHODS ----------------------------------- #

    def nn_feedforward_pyo(self, Y, num_nodes):
        '''
        Feedforward pass of the NN (tanh hidden layers, linear output) over a stack of nodes.
        Y is a 1-indexed pyo Var/Expression (node, state). Returns a (num_nodes, out) ndarray of
        pyo.Expressions. (Copied verbatim from node_pyomo_monolithic.py.)
        '''
        activations = [Y]
        for l in self.model.l:
            current_activation = np.empty((num_nodes, self.layer_sizes[l]), dtype=object)
            for u in range(1, self.layer_sizes[l] + 1):
                for i in range(1, num_nodes + 1):
                    if l == 1:
                        linear_out = sum(activations[l-1][i, d] * self.model.Ws[l].W[d, u] for d in self.model.d) + self.model.bs[l].b[1, u]
                    else:
                        linear_out = sum(activations[l-1][i-1, k-1] * self.model.Ws[l].W[k, u] for k in range(1, self.layer_sizes[l-1] + 1)) + self.model.bs[l].b[1, u]
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
        Solve the knotted monolithic Pyomo model using the IPOPT solver.
        (Copied verbatim from node_pyomo_monolithic.py.)

        Returns:
            self.last_solve_info dict.
        '''
        # set up IPOPT solver 
        solver = pyo.SolverFactory("ipopt", executable="/usr/local/bin/ipopt")

        # Every key in solver_options is forwarded to IPOPT (max_iter, tol, acceptable_*, print_level, ...);
        # everything else is read from the option file below - don't also set a solver_options key there.
        for key, value in solver_options.items():
            solver.options[key] = value
        repo_root = Path(__file__).resolve().parents[3]  # .../CNODE
        option_path = repo_root / "ipopt.opt"
        if not option_path.exists():
            raise FileNotFoundError(f"IPOPT option file not found at {option_path}")
        solver.options["option_file_name"] = str(option_path)

        t0 = time.perf_counter()
        result = None
        try:
            result = solver.solve(self.model, tee=True, load_solutions=True)
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
            print(f"Knotted monolithic solve raised {type(exc).__name__}: {exc}.")
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
            print("Knotted monolithic solve loaded (status={}, termination={}); NN parameters frozen.".format(
                status, termination))
        else:
            print("Knotted monolithic solve returned no usable solution (status={}, termination={}).".format(
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

    def get_phase_solution(self, b: int):
        '''Current Y_star values of trajectory b, phase by phase: list of (grid (N,), Y (N, D)).
        Consecutive phases share their boundary time; the two values there are x^- and x^+.'''
        N = self.num_colloc_nodes
        out = []
        for k, blk in enumerate(self.blocks):
            if blk['traj'] != b:
                continue
            Y = np.array([[pyo.value(self.model.Y_star[k * N + n, d]) for d in self.model.d] for n in self.model.nloc])
            out.append((self.colloc_grids[k], Y))
        return out

    def evaluate_interpolant(self, b: int, t_eval: np.ndarray):
        '''Evaluate trajectory b's knotted interpolant at times t_eval (pre-feed value at a feed time).'''
        phases = self.get_phase_solution(b)
        t_eval = np.atleast_1d(np.asarray(t_eval, dtype=float))
        out = np.full((t_eval.size, self.state_dim), np.nan)
        for p, (grid, Y) in enumerate(phases):
            lo, hi = grid[0], grid[-1]
            mask = (t_eval > lo) & (t_eval <= hi) if p > 0 else (t_eval >= lo) & (t_eval <= hi)
            for i in np.flatnonzero(mask):
                out[i] = self._bary_eval_row(grid, t_eval[i]) @ Y
        return out

    # ---------------------------- PREDICTION METHODS -------------------------------- #

    def pyomo_var_to_numpy(self, pyo_obj, shape):
        return np.array([[pyo.value(pyo_obj[i, j]) for j in range(1, shape[1]+1)] for i in range(1, shape[0]+1)])

    def convert_weights(self):
        return [self.pyomo_var_to_numpy(self.model.Ws[l].W, (self.layer_sizes[l-1], self.layer_sizes[l])) for l in self.model.l]

    def convert_biases(self):
        return [self.pyomo_var_to_numpy(self.model.bs[l].b, (1, self.layer_sizes[l])) for l in self.model.l]

    def nn_prediction(self, t, y_0, Ws, bs):
        y_0 = y_0.reshape(1, -1)
        out = np.tanh(np.dot(y_0, Ws[0]) + bs[0])
        for W, b in zip(Ws[1:-1], bs[1:-1]):
            out = np.tanh(np.dot(out, W) + b)
        out = np.dot(out, Ws[-1]) + bs[-1]
        return out.flatten()

    def get_predicted_trajectory(self,
            y_0, t_grid,
            feed_times=(), feed_amounts=(),
            method: str = 'RK45',
            rtol: float = 1e-3,
            atol: float = 1e-6,
            max_step: float = np.inf
    ):
        '''
        Predict the state trajectory over t_grid by PIECEWISE forward integration of the trained NN:
        integrate up to each feed time, add the bolus to the jump state, restart (exactly as
        FedBatchSim._integrate_batch). A grid point that coincides with a feed time returns the
        PRE-feed state, matching how the data are recorded. With no feeds this is a single solve_ivp.

        Returns: (len(t_grid), state_dim) array.
        '''
        t_grid = np.asarray(t_grid, dtype=float)
        if t_grid.ndim != 1 or t_grid.size < 2:
            raise ValueError(f"t_grid must be a 1D array with >=2 points; got shape {t_grid.shape}.")
        if not np.all(np.isfinite(t_grid)) or np.any(np.diff(t_grid) <= 0):
            raise ValueError("t_grid must be finite and strictly increasing.")
        y_0 = np.asarray(y_0, dtype=float)
        if y_0.shape != (self.state_dim,) or not np.all(np.isfinite(y_0)):
            raise ValueError(f"y_0 must be a finite 1D array of length {self.state_dim}; got shape {y_0.shape}.")
        feed_times = np.asarray(feed_times, dtype=float).ravel()
        feed_amounts = np.asarray(feed_amounts, dtype=float).ravel()
        if feed_times.shape != feed_amounts.shape:
            raise ValueError("feed_times and feed_amounts must have the same length.")
        order = np.argsort(feed_times)
        feed_times, feed_amounts = feed_times[order], feed_amounts[order]
        inside = (feed_times > t_grid[0]) & (feed_times < t_grid[-1])
        feed_times, feed_amounts = feed_times[inside], feed_amounts[inside]

        Ws, bs = self.convert_weights(), self.convert_biases()
        edges = np.concatenate([[t_grid[0]], feed_times, [t_grid[-1]]])
        pred = np.empty((t_grid.size, self.state_dim))
        state = y_0.copy()
        for p, (t0, t1) in enumerate(zip(edges[:-1], edges[1:])):
            if p > 0:
                state = state.copy()
                state[self.jump_state_idx] += feed_amounts[p - 1]
            mask = (t_grid > t0) & (t_grid <= t1)
            if p == 0:
                mask |= np.isclose(t_grid, t0)
            t_eval = np.unique(np.concatenate([t_grid[mask], [t1]]))  # always carry the end state
            sol = solve_ivp(
                fun=self.nn_prediction, t_span=(t0, t1), y0=state, t_eval=t_eval,
                method=method, rtol=rtol, atol=atol, max_step=max_step, args=(Ws, bs)
            )
            if not sol.success:
                raise RuntimeError(f"solve_ivp failed on [{t0}, {t1}]: status={sol.status}, message={sol.message}")
            for i in np.flatnonzero(mask):
                pred[i] = sol.y[:, np.flatnonzero(np.isclose(sol.t, t_grid[i]))[0]]
            state = sol.y[:, -1]
        return pred

    def compute_prediction_mse(self,
            trajs: list,
            method='RK45',
            rtol=1e-3,
            atol=1e-6,
            max_step=np.inf,
            return_range=False
    ):
        '''
        Mean / std over trajectories of the per-trajectory MSE (over observation times x states)
        of the piecewise forward prediction, each started from that trajectory's y0 and using its
        own feed schedule. trajs is a list of dicts as returned by feed_data.load_feed_trajectories()
        (train or test set). Returns (mean, std) or, with return_range, (mean, std, min, max);
        nan(s) if any forward solve fails.
        '''
        nan_out = (np.nan, np.nan, np.nan, np.nan) if return_range else (np.nan, np.nan)
        per_traj = []
        for tr in trajs:
            try:
                pred = self.get_predicted_trajectory(
                    tr['y0'], tr['obs_times'], tr['feed_times'], tr['feed_amounts'],
                    method, rtol, atol, max_step
                )
            except RuntimeError:
                return nan_out
            per_traj.append(np.mean((pred - tr['Y_obs']) ** 2))
        per_traj = np.asarray(per_traj, dtype=float)
        mean, std = float(per_traj.mean()), float(per_traj.std())
        if return_range:
            return mean, std, float(per_traj.min()), float(per_traj.max())
        return mean, std
