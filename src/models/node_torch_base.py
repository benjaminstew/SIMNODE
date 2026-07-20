import math
import time
from typing import Iterable, Optional
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torchdiffeq import odeint
from scipy.interpolate import CubicSpline


class FeedForwardNN(nn.Module):
    """Feedforward MLP with tanh hidden layers and linear output layer."""
    def __init__(self, layer_sizes: Iterable[int]):
        super().__init__()
        sizes = list(layer_sizes)
        if len(sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output dimensions.")
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        # Xavier/Glorot normal for weights; uniform bias as per nn.Linear default
        # (Appendix A.1: W_ij ~ N(0, 2/(n_in + n_out)), b_i ~ U(-1/sqrt(n_in), 1/sqrt(n_in)))
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            fan_in = layer.weight.shape[1]
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


class ExogenousSpline:
    """Cubic spline interpolant for exogenous inputs on a time grid (Appendix A.3)."""

    def __init__(self, t_grid: np.ndarray, x_samples: np.ndarray):
        if t_grid.ndim != 1:
            raise ValueError("t_grid must be a 1D array.")
        if x_samples.ndim != 2:
            raise ValueError("x_samples must be a 2D array of shape (N, p).")
        if t_grid.shape[0] != x_samples.shape[0]:
            raise ValueError("t_grid and x_samples must have the same length.")
        if np.any(np.diff(t_grid) <= 0):
            raise ValueError("t_grid must be strictly increasing.")
        self._spline = CubicSpline(t_grid, x_samples, axis=0)

    def __call__(self, t: float) -> np.ndarray:
        return np.asarray(self._spline(float(t)), dtype=np.float32)


class NODESequential:
    """
    Sequential Neural ODE training via solver-in-the-loop backpropagation.

    Implements Algorithm 1 / Section 2.2 of Sharapolova et al. (2025, JPC):
      1. Forward solve: Yhat = ODESolve(f_theta, y0, t)
      2. Loss: MSE (1/N)||Yhat - Yobs||_F^2 + optional L2 regularisation
      3. Gradients: backprop through torchdiffeq (discrete adjoint)
      4. Update: Adam

    Defaults follow Appendix A.2: dopri5, rtol=1e-3, atol=1e-6, Adam lr=1e-3.

    Optional supervised pre-training on estimated derivatives (Section 5.2) can
    substantially improve convergence when starting from random initialisation.
    """

    def __init__(
        self,
        Y_obs: np.ndarray,
        t_grid: np.ndarray,
        layer_sizes: Iterable[int],
        exogenous_inputs: Optional[np.ndarray] = None,
        lambda_reg: float = 0.0,
        solver_method: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_step: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        if Y_obs.ndim != 2:
            raise ValueError("Y_obs must be a 2D array of shape (N, state_dim).")
        if t_grid.ndim != 1:
            raise ValueError("t_grid must be a 1D array.")
        if t_grid.shape[0] != Y_obs.shape[0]:
            raise ValueError("t_grid length must match Y_obs rows.")
        if np.any(np.diff(t_grid) <= 0):
            raise ValueError("t_grid must be strictly increasing.")

        self.num_nodes, self.state_dim = Y_obs.shape
        self.lambda_reg = float(lambda_reg)
        self.solver_method = solver_method
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_step = max_step

        self.device = torch.device(device or "cpu")
        self.t_grid = torch.tensor(t_grid, dtype=torch.float32, device=self.device)
        self.Y_obs = torch.tensor(Y_obs, dtype=torch.float32, device=self.device)

        self._t_grid_np = t_grid
        self._Y_obs_np = Y_obs

        exog_dim = 0
        self.exogenous_spline: Optional[ExogenousSpline] = None
        if exogenous_inputs is not None:
            if exogenous_inputs.ndim != 2:
                raise ValueError("exogenous_inputs must be a 2D array of shape (N, p).")
            if exogenous_inputs.shape[0] != self.num_nodes:
                raise ValueError("exogenous_inputs must have the same length as t_grid.")
            exog_dim = exogenous_inputs.shape[1]
            self.exogenous_spline = ExogenousSpline(t_grid, exogenous_inputs)

        sizes = list(layer_sizes)
        if sizes[0] != self.state_dim + exog_dim:
            raise ValueError(
                f"layer_sizes[0] must equal state_dim + exogenous_dim "
                f"({self.state_dim + exog_dim})."
            )
        if sizes[-1] != self.state_dim:
            raise ValueError("layer_sizes[-1] must equal state_dim.")

        self.model = FeedForwardNN(sizes).to(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_nn_input(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Assemble the network input [y, x(t)] at solver time t.

        Handles both batched (2-D) and unbatched (1-D) state tensors.
        Exogenous inputs are evaluated via the cubic spline at t (Appendix A.3).
        """
        y_in = y.unsqueeze(0) if y.dim() == 1 else y  # ensure 2-D: (batch, state_dim)
        if self.exogenous_spline is not None:
            x_np = self.exogenous_spline(float(t))
            x = torch.tensor(x_np, dtype=y_in.dtype, device=y_in.device)
            x = x.expand(y_in.shape[0], -1)
            y_in = torch.cat([y_in, x], dim=-1)
        return y_in  # (batch, input_dim)

    def _ode_rhs(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Right-hand side f_theta(y, t) called by the IVP solver."""
        y_in = self._build_nn_input(t, y)
        dy = self.model(y_in)
        return dy.squeeze(0) if y.dim() == 1 else dy

    # ------------------------------------------------------------------
    # Forward solve
    # ------------------------------------------------------------------

    def forward_solve(
        self,
        y0: Optional[np.ndarray] = None,
        t_grid: Optional[np.ndarray] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_step: Optional[float] = None,
    ) -> torch.Tensor:
        """Integrate the Neural ODE from y0 over t_grid and return Yhat (N, state_dim).

        rtol/atol/max_step override the construction-time tolerances for this call when
        provided (e.g. tighter tolerances at inference); otherwise the instance defaults
        are used.
        """
        if y0 is None:
            y0_t = self.Y_obs[0]
        else:
            y0_t = torch.tensor(y0, dtype=torch.float32, device=self.device)
        if y0_t.dim() != 1:
            raise ValueError("y0 must be a 1D array.")

        if t_grid is None:
            t_t = self.t_grid
        else:
            t_t = torch.tensor(t_grid, dtype=torch.float32, device=self.device)

        rtol = self.rtol if rtol is None else float(rtol)
        atol = self.atol if atol is None else float(atol)
        max_step = self.max_step if max_step is None else float(max_step)
        options = None if max_step is None else {"max_step": float(max_step)}

        return odeint(
            self._ode_rhs,
            y0_t,
            t_t,
            method=self.solver_method,
            rtol=rtol,
            atol=atol,
            options=options,
        )  # shape: (N, state_dim)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, Y_hat: torch.Tensor) -> torch.Tensor:
        """(1/N) * ||Yhat - Yobs||_F^2 plus optional L2 regularisation (Eq. 4)."""
        if Y_hat.shape != self.Y_obs.shape:
            raise ValueError("Predicted trajectory shape must match Y_obs.")
        mse = (Y_hat - self.Y_obs).pow(2).sum() / float(self.num_nodes)
        if self.lambda_reg <= 0.0:
            return mse
        l2 = sum(p.pow(2).sum() for p in self.model.parameters())
        return mse + self.lambda_reg * l2

    # ------------------------------------------------------------------
    # Supervised pre-training (Section 5.2)
    # ------------------------------------------------------------------

    def pretrain(
        self,
        num_iterations: int,
        learning_rate: float = 1e-3,
        fraction: float = 0.2,
        verbose: bool = False,
        log_every: int = 100,
    ) -> list[float]:
        """
        Supervised pre-training on numerically estimated state derivatives.

        Uses a cubic spline fitted to the first `fraction` of training observations
        to estimate dY/dt and trains f_theta as a regression directly on those
        targets. This places the network in a reasonable region before the more
        expensive solver-in-the-loop phase (Section 5.2).
        """
        n_pre = max(2, int(self.num_nodes * fraction))
        t_sub = self._t_grid_np[:n_pre]
        Y_sub = self._Y_obs_np[:n_pre]

        spline = CubicSpline(t_sub, Y_sub, axis=0)
        dY_sub = spline(t_sub, 1).astype(np.float32)  # first derivative

        # Build network inputs for each pre-training point
        inputs = []
        for i in range(n_pre):
            y_i = torch.tensor(Y_sub[i], dtype=torch.float32, device=self.device)
            t_i = torch.tensor(t_sub[i], dtype=torch.float32, device=self.device)
            inp = self._build_nn_input(t_i, y_i).squeeze(0)  # (input_dim,)
            inputs.append(inp)

        X_pre = torch.stack(inputs)  # (n_pre, input_dim)
        dY_target = torch.tensor(dY_sub, device=self.device)  # (n_pre, state_dim)

        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        history: list[float] = []

        self.model.train()
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            dY_hat = self.model(X_pre)
            loss_val = (dY_hat - dY_target).pow(2).mean()
            loss_val.backward()
            optimizer.step()

            loss_scalar = float(loss_val.detach().cpu())
            history.append(loss_scalar)
            if verbose and (iteration + 1) % max(1, log_every) == 0:
                print(f"  [pretrain] iter {iteration + 1}/{num_iterations}: loss={loss_scalar:.6f}")

        return history

    # ------------------------------------------------------------------
    # End-to-end training (Algorithm 1 / Section 2.2)
    # ------------------------------------------------------------------

    def train(
        self,
        num_iterations: int,
        learning_rate: float = 1e-3,
        pretrain_iterations: int = 0,
        pretrain_fraction: float = 0.2,
        verbose: bool = False,
        log_every: int = 100,
    ) -> dict[str, list[float]]:
        """
        Train via solver-in-the-loop backpropagation.

        Parameters
        ----------
        num_iterations : int
            Number of end-to-end Adam steps.
        learning_rate : float
            Adam learning rate (default 1e-3, per Appendix A.2).
        pretrain_iterations : int
            Adam steps for supervised derivative pre-training (0 = disabled).
            Setting this to ~1000 iterations typically resolves convergence
            difficulties when starting from random initialisation (Section 5.2).
        pretrain_fraction : float
            Fraction of training data used to estimate derivatives for pre-training.
        verbose : bool
            Print loss at regular intervals.
        log_every : int
            Print frequency (in iterations).

        Returns
        -------
        dict with keys 'pretrain' and 'train', each containing a loss history list.
        """
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")

        pretrain_history: list[float] = []
        if pretrain_iterations > 0:
            if verbose:
                print(
                    f"Pre-training for {pretrain_iterations} iterations "
                    f"on first {pretrain_fraction * 100:.0f}% of observations ..."
                )
            pretrain_history = self.pretrain(
                num_iterations=pretrain_iterations,
                learning_rate=learning_rate,
                fraction=pretrain_fraction,
                verbose=verbose,
                log_every=log_every,
            )

        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        train_history: list[float] = []

        self.model.train()
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            Y_hat = self.forward_solve()
            loss_val = self.loss(Y_hat)
            loss_val.backward()
            optimizer.step()

            loss_scalar = float(loss_val.detach().cpu())
            train_history.append(loss_scalar)
            if verbose and (iteration + 1) % max(1, log_every) == 0:
                print(f"Iter {iteration + 1}/{num_iterations}: loss={loss_scalar:.6f}")

        return {"pretrain": pretrain_history, "train": train_history}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_predicted_trajectory(
        self,
        y0: np.ndarray,
        t_grid: np.ndarray,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_step: Optional[float] = None,
    ) -> np.ndarray:
        """Integrate the trained model from y0 over t_grid (eval mode, no grad).
s
        Optional rtol/atol/max_step override the construction-time tolerances, mirroring
        the NODEPyomo.get_predicted_trajectory signature so callers can use both uniformly.
        """
        self.model.eval()
        with torch.no_grad():
            Y_hat = self.forward_solve(y0=y0, t_grid=t_grid, rtol=rtol, atol=atol, max_step=max_step)
        return Y_hat.cpu().numpy()


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def train_sequential_node(
    Y_obs: np.ndarray,
    t_grid: np.ndarray,
    layer_sizes: Iterable[int],
    num_iterations: int,
    exogenous_inputs: Optional[np.ndarray] = None,
    lambda_reg: float = 0.0,
    learning_rate: float = 1e-3,
    pretrain_iterations: int = 0,
    pretrain_fraction: float = 0.2,
    solver_method: str = "dopri5",
    rtol: float = 1e-3,
    atol: float = 1e-6,
    max_step: Optional[float] = None,
    device: Optional[str] = None,
    verbose: bool = False,
    log_every: int = 100,
) -> tuple["NODESequential", dict[str, list[float]]]:
    """
    Build and train a sequential Neural ODE.

    Default solver/optimiser settings match Appendix A.2 (dopri5, Adam lr=1e-3).
    Pass pretrain_iterations > 0 to run supervised derivative pre-training first.
    """
    model = NODESequential(
        Y_obs=Y_obs,
        t_grid=t_grid,
        layer_sizes=layer_sizes,
        exogenous_inputs=exogenous_inputs,
        lambda_reg=lambda_reg,
        solver_method=solver_method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        device=device,
    )
    history = model.train(
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        pretrain_iterations=pretrain_iterations,
        pretrain_fraction=pretrain_fraction,
        verbose=verbose,
        log_every=log_every,
    )
    return model, history


# ---------------------------------------------------------------------------
# Multi-trajectory sequential training (for consensus-style comparisons)
# ---------------------------------------------------------------------------

def train_sequential_multitraj(
    Y_obs_train: np.ndarray,
    train_grid: np.ndarray,
    layer_sizes: Iterable[int],
    num_iterations: int,
    Y_obs_test: Optional[np.ndarray] = None,
    test_grid: Optional[np.ndarray] = None,
    lambda_reg: float = 0.0,
    learning_rate: float = 1e-3,
    solver_method: str = "dopri5",
    rtol: float = 1e-3,
    atol: float = 1e-6,
    max_step: Optional[float] = None,
    record_every: int = 10,
    device: Optional[str] = None,
    verbose: bool = False,
    log_every: int = 50,
):
    """
    Train ONE sequential Neural ODE jointly on multiple trajectories that share the
    same dynamics (e.g. bioreactor batches with different initial conditions). This is
    the solver-in-the-loop analogue of the ADMM consensus model, which couples all
    batches into a single network -- so the two are directly comparable.

    Each trajectory is integrated from its own initial condition over a shared time grid
    (batched odeint); the loss is the summed per-trajectory MSE (Eq. 4) plus optional L2
    regularisation. Train/test MSE are recorded versus wall-clock time using the same
    per-batch-mean convention as ADMM.compute_prediction_mse, so the returned history
    drops directly into the convergence plots.

    Autonomous networks only (input = state); raise if layer_sizes is not state->state.

    Parameters
    ----------
    Y_obs_train : ndarray (num_batches, num_nodes, state_dim)
    train_grid  : ndarray (num_nodes,)
    Y_obs_test, test_grid : optional held-out trajectories for the test-MSE history.
    record_every : record train/test MSE every this many iterations.

    Returns
    -------
    predict : callable(y0 (state_dim,), grid (M,)) -> ndarray (M, state_dim)
        Forward-integrates the trained model in eval mode (for plotting).
    mse_history : dict with keys time_min, train_mse, train_mse_std, test_mse,
        test_mse_std (lists), matching ADMM.mse_history.
    """
    if Y_obs_train.ndim != 3:
        raise ValueError("Y_obs_train must be 3D (num_batches, num_nodes, state_dim).")
    state_dim = Y_obs_train.shape[2]
    sizes = list(layer_sizes)
    if sizes[0] != state_dim or sizes[-1] != state_dim:
        raise ValueError(
            "train_sequential_multitraj supports autonomous networks only "
            f"(layer_sizes must map {state_dim} -> {state_dim})."
        )

    dev = torch.device(device or "cpu")
    model = FeedForwardNN(sizes).to(dev)

    t_train = torch.tensor(train_grid, dtype=torch.float32, device=dev)
    Y_train = torch.tensor(Y_obs_train, dtype=torch.float32, device=dev)   # (B, N, D)
    y0_train = Y_train[:, 0, :]                                            # (B, D)
    Y_train_NBD = Y_train.permute(1, 0, 2)                                 # (N, B, D)

    has_test = Y_obs_test is not None and test_grid is not None
    if has_test:
        t_test = torch.tensor(test_grid, dtype=torch.float32, device=dev)
        Y_test = torch.tensor(Y_obs_test, dtype=torch.float32, device=dev)  # (B, N, D)
        y0_test = Y_test[:, 0, :]

    options = None if max_step is None else {"max_step": float(max_step)}

    def rhs(t, y):                       # autonomous: f_theta(y)
        return model(y)

    def rollout(y0_batch, t_eval):
        return odeint(rhs, y0_batch, t_eval, method=solver_method,
                      rtol=rtol, atol=atol, options=options)               # (M, B, D)

    def per_batch_mse(y0_batch, t_eval, Y_ref_BND):
        # mean/std across batches of mean((pred - obs)^2), matching ADMM convention
        with torch.no_grad():
            pred = rollout(y0_batch, t_eval).permute(1, 0, 2)              # (B, M, D)
        mse_b = (pred - Y_ref_BND).pow(2).mean(dim=(1, 2)).cpu().numpy()   # (B,)
        return float(np.mean(mse_b)), float(np.std(mse_b))

    mse_history = {"time_min": [], "train_mse": [], "train_mse_std": [],
                   "test_mse": [], "test_mse_std": []}
    optimizer = Adam(model.parameters(), lr=learning_rate)
    t_start = time.perf_counter()

    def record():
        elapsed_min = (time.perf_counter() - t_start) / 60.0
        model.eval()
        try:
            tr_m, tr_s = per_batch_mse(y0_train, t_train, Y_train)
        except (RuntimeError, AssertionError):
            tr_m, tr_s = np.nan, np.nan   # unstable field early in training => skip point
        if has_test:
            try:
                te_m, te_s = per_batch_mse(y0_test, t_test, Y_test)
            except (RuntimeError, AssertionError):
                te_m, te_s = np.nan, np.nan
        else:
            te_m, te_s = np.nan, np.nan
        model.train()
        mse_history["time_min"].append(elapsed_min)
        mse_history["train_mse"].append(tr_m)
        mse_history["train_mse_std"].append(tr_s)
        mse_history["test_mse"].append(te_m)
        mse_history["test_mse_std"].append(te_s)

    N = t_train.shape[0]
    model.train()
    record()  # initial (pre-training) point
    for it in range(num_iterations):
        optimizer.zero_grad()
        Y_hat = rollout(y0_train, t_train)                                 # (N, B, D)
        loss_val = (Y_hat - Y_train_NBD).pow(2).sum() / float(N)
        if lambda_reg > 0.0:
            loss_val = loss_val + lambda_reg * sum(p.pow(2).sum() for p in model.parameters())
        loss_val.backward()
        optimizer.step()

        if (it + 1) % max(1, record_every) == 0:
            record()
        if verbose and (it + 1) % max(1, log_every) == 0:
            print(f"[seq] iter {it + 1}/{num_iterations}: loss={float(loss_val.detach().cpu()):.6e}")

    if num_iterations % max(1, record_every) != 0:
        record()  # ensure the final state is recorded

    def predict(y0, grid):
        model.eval()
        y0_t = torch.tensor(np.asarray(y0, dtype=np.float32).reshape(-1), device=dev)
        grid_t = torch.tensor(np.asarray(grid, dtype=np.float32), device=dev)
        with torch.no_grad():
            pred = odeint(rhs, y0_t, grid_t, method=solver_method,
                          rtol=rtol, atol=atol, options=options)
        return pred.cpu().numpy()        # (M, state_dim)

    return predict, mse_history
