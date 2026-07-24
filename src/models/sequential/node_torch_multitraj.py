'''
Multi-trajectory (batched) sequential Neural ODE training.

Extends the single-trajectory 'NODESequential' baseline to train ONE shared
vector field f_theta jointly across multiple independent trajectories.

All trajectories in a mini-batch are integrated together
with a single batched `odeint` call over one shared time grid (one shared
adaptive step schedule). This is the canonical approach on GPUs
and a simple baseline on a CPU. 
'''
import time
import numpy as np
import torch
from torch.optim import Adam
from torchdiffeq import odeint
from scipy.interpolate import CubicSpline
from .node_torch_base import FeedForwardNN

class NODESequentialMultiTraj():
    """
    Sequential Neural ODE training jointly on B trajectories sharing one f_theta.

    Y_obs is a batch of trajectories of shape (B, N, state_dim); all trajectories share the
    time grid. Autonomous networks only (no exogeneous inputs).

    Loss is the summed per-trajectory MSE plus optional L2 regularisation, so
    'mini_batch_size = B' reproduces the all-trajectory objective.
    """
    def __init__(self,
        Y_obs: torch.Tensor,
        t_grid: torch.Tensor,
        layer_sizes: list[int],
        lambda_reg: float = 0.0,
        solver_method: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_step: float = None,
        device: str = "cpu"
    ) -> None:
        if Y_obs.ndim != 3:
            raise ValueError("Y_obs must be a 3D tensor of shape (B, N, state_dim).")
        if t_grid.ndim != 1:
            raise ValueError("t_grid must be a 1D tensor.")
        if t_grid.shape[0] != Y_obs.shape[1]:
            raise ValueError("t_grid length must match Y_obs time dimension (axis 1).")
        if torch.any(torch.diff(t_grid) <= 0):
            raise ValueError("t_grid must be strictly increasing.")

        self.device = torch.device(device)
        self.B, self.N, self.state_dim = Y_obs.shape
        self.Y_obs = Y_obs.to(device=self.device, dtype=torch.float32)
        self.y0_batch = self.Y_obs[:, 0, :].contiguous() # per-trajectory initial values - (B, D)

        self.lambda_reg = lambda_reg
        self.solver_method = solver_method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.t_grid = t_grid.to(device=self.device, dtype=torch.float32) # (N,)
        self.layer_sizes = layer_sizes

        if self.layer_sizes[0] != self.state_dim:
            raise ValueError(f"layer_sizes[0] must equal state_dim ({self.state_dim}).")
        if self.layer_sizes[-1] != self.state_dim:
            raise ValueError("layer_sizes[-1] must equal state_dim.")

        self.model = FeedForwardNN(self.layer_sizes).to(self.device)

    def ode_rhs(self, t: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        """RHS f_theta(y, t) called by the IVP solver."""
        return self.model(y0)

    def forward_solve(self,
        y0: torch.Tensor = None,
        t_grid: torch.Tensor = None,
        rtol: float = None,
        atol: float = None,
        max_step: float = None
    ) -> torch.Tensor:
        """
        Integrate the Neural ODE from a batched y0 over t_grid.

        Defaults to the full batch of initial conditions (self.y0_batch, shape
        (B, D)) over self.t_grid, returning (B, N, D). Passing a subset (m, D)
        integrates that mini-batch -> (m, N, D). y0 is always 2D (m, D); a single
        trajectory is m = 1, so the output is uniformly (m, N, D).
        """
        y0_eval = self.y0_batch if y0 is None else y0.to(device=self.device, dtype=torch.float32)
        if y0_eval.ndim != 2:
            raise ValueError("forward_solve expects a batched y0 of shape (m, D).")
        t_eval = self.t_grid if t_grid is None else t_grid.to(device=self.device, dtype=torch.float32)
        rtol = self.rtol if rtol is None else float(rtol)
        atol = self.atol if atol is None else float(atol)
        max_step = self.max_step if max_step is None else float(max_step)
        options = None if max_step is None else {"max_step": float(max_step)}

        return odeint(
            self.ode_rhs,
            y0_eval,
            t_eval,
            method=self.solver_method,
            rtol=rtol,
            atol=atol,
            options=options
        ).permute(1, 0, 2).contiguous() # changed to (m, N, D) 

    def compute_loss(self, Y: torch.Tensor, Y_obs: torch.Tensor = None) -> torch.Tensor:
        """
        Summed per-trajectory MSE + L2 reg.

        Y and Y_obs are (m, N, D). Defaults to the full batch (m = B).
        """
        target = self.Y_obs if Y_obs is None else Y_obs
        mse = (Y - target).pow(2).sum() / self.N
        if self.lambda_reg <= 0.0:
            return mse
        l2 = sum(p.pow(2).sum() for p in self.model.parameters())
        return mse + self.lambda_reg * l2

    def pretrain(self,
        num_iterations: int,
        Y_obs: torch.Tensor = None,
        learning_rate: float = 1e-3,
        fraction: float = 0.2,
        verbose: bool = False,
        log_every: int = 100
    ) -> list[float]:
        """
        Supervised pre-training on numerically estimated state derivatives.
        
        Uses a cubic spline fitted to the first 'fraction' of training observations
        to estimate dY/dt and trains f_theta as a regression model directly on those
        targets. This places the network in a reasonable region before the more
        expensive solver-in-the-loop phase.
        """
        n_pre = max(2, int(self.N * fraction))
        t_pre = self.t_grid[:n_pre] # (n_pre,)
        Y_pre = self.Y_obs[:, :n_pre, :] if Y_obs is None else Y_obs[:, :n_pre, :] 

        t_pre_np = t_pre.detach().cpu().numpy()
        Y_pre_np = Y_pre.detach().cpu().numpy()
        spline = CubicSpline(t_pre_np, Y_pre_np, axis=1)
        dY_target_np = spline(t_pre_np, 1).astype(np.float32) # first derivative 

        dY_target = torch.tensor(dY_target_np, device=self.device, dtype=torch.float32)  # (m, n_pre, state_dim)

        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        loss_history: list[float] = []

        self.model.train() # set NN to training mode 
        for iteration in range(num_iterations):
            optimizer.zero_grad() # clear grads from prev step
            dY_pred = self.model(Y_pre) # (m, n_pre, state_dim)
            loss_val = (dY_pred - dY_target).pow(2).mean()
            loss_val.backward() # backprop
            optimizer.step() # update params

            loss_scalar = float(loss_val.detach().cpu())
            loss_history.append(loss_scalar)
            if verbose and (iteration + 1) % max(1, log_every) == 0:
                print(f"  [pretrain] iter {iteration + 1}/{num_iterations}: loss={loss_scalar:.6f}")

        return loss_history

    def compute_prediction_mse(self,
            Y_obs: torch.Tensor,
            t_grid: torch.Tensor,
            rtol: float = None,
            atol: float = None,
            max_step: float = None
    ) -> tuple[float, float]:
        """
        MSE between observed and predicted trajectories with the current parameters. 
        Per-trajectory mean, then the mean and (population) std across trajectories.

        Returns (nan, nan) if the field is unstable (solver raises), so unstable
        early-training points are skipped rather than crashing the run.
        """
        Y_ref = self.Y_obs if Y_obs is None else Y_obs.to(device=self.device, dtype=torch.float32) # (m, N, D)

        self.model.eval()
        try:
            with torch.no_grad():
                Y_pred = self.forward_solve(y0=Y_ref[:, 0, :], t_grid=t_grid, rtol=rtol, atol=atol, max_step=max_step)  # (m, N, D)
        except (RuntimeError, AssertionError):
            return float("nan"), float("nan")
        finally:
            self.model.train()
        mse = (Y_pred - Y_ref).pow(2).mean(dim=(1, 2)) # (B,)

        return float(mse.mean()), float(mse.std(unbiased=False)) # unbiased=False to match numpy's np.std (population) used by ADMM.

    def train(self,
        num_epochs: int,
        learning_rate: float = 1e-3,
        mini_batch_size: int = None,
        pretrain_iterations: int = 0,
        pretrain_fraction: float = 0.2,
        Y_obs_test: torch.Tensor = None,
        test_grid: torch.Tensor = None,
        record_mse_every: int = 3,
        time_budget_min: float = None,
        eval_rtol: float = 1e-3,
        eval_atol: float = 1e-6,
        seed: int = None,
        verbose: bool = False,
        log_every: int = 50
    ) -> dict[str, list[float]]:
        """
        Train via mini-batch Adam solver-in-the-loop backpropagation.

        Args:
        - num_epochs: passes over all B trajectories.
        - learning_rate: Adam learning rate (default 1e-3, per JPC Appendix A.2).
        - mini_batch_size: trajectories per Adam step; None or >= B => full batch.
        - pretrain_iterations / pretrain_fraction: optional supervised
          derivative pre-training across trajectories (0 = disabled).
        - Y_obs_test (B_test, N_test, D), test_grid (N_test,): optional held-out
          trajectories for the test-MSE history.
        - record_mse_every: record train/test MSE every this many OPTIMISER UPDATES
          (mini-batch steps), not epochs. 1 = record after every update.
        - time_budget_min: if set, stop once training wall-clock exceeds this.
        - eval_rtol / eval_atol: tolerances for the MSE-evaluation forward solves;
          set these to match the tolerances used to evaluate the ADMM model.
        - seed: seeds mini-batch shuffling for reproducibility.

        Returns 'mse_history' with keys time_min, train_mse, train_mse_std,
        test_mse, test_mse_std. 'time_min' measures wall-clock training time, with 
        the timer paused during MSE evaluation.
        """
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))

        if pretrain_iterations > 0:
            if verbose:
                print(
                    f"Pre-training for {pretrain_iterations} iterations on first "
                    f"{pretrain_fraction * 100:.0f}% of each trajectory ..."
                )
            self.pretrain(
                num_iterations=pretrain_iterations,
                learning_rate=learning_rate,
                fraction=pretrain_fraction,
                verbose=verbose,
                log_every=log_every
            )

        m = self.B if mini_batch_size is None else max(1, mini_batch_size)
        if Y_obs_test is not None and test_grid is not None:
            Y_test = Y_obs_test.to(device=self.device, dtype=torch.float32)
            t_test = test_grid.to(device=self.device, dtype=torch.float32)
        mse_history = {
            "time_min": [], "train_mse": [], "train_mse_std": [],
            "test_mse": [], "test_mse_std": [],
        }

        optimizer = Adam(self.model.parameters(), lr=learning_rate) 
        self.model.train()
        t_start = time.perf_counter()

        def mse_record():
            # Record wall-clock convergence (prediction MSE) using the current model
            # Timer is paused for evaluation
            nonlocal t_start
            t_eval0 = time.perf_counter()
            train_mse, train_mse_std = self.compute_prediction_mse(
                self.Y_obs, self.t_grid, rtol=eval_rtol, atol=eval_atol
            )
            if Y_obs_test is not None and test_grid is not None:
                test_mse, test_mse_std = self.compute_prediction_mse(
                    Y_test, t_test, rtol=eval_rtol, atol=eval_atol
                )
            else:
                test_mse, test_mse_std = float("nan"), float("nan")
            elapsed_min = (time.perf_counter() - t_start) / 60.0
            mse_history["time_min"].append(elapsed_min)
            mse_history["train_mse"].append(train_mse)
            mse_history["train_mse_std"].append(train_mse_std)
            mse_history["test_mse"].append(test_mse)
            mse_history["test_mse_std"].append(test_mse_std)
            t_start += time.perf_counter() - t_eval0  # exclude eval cost from timer

        # training loop
        update = 0
        mse_record() # initial (post-pretrain) point
        stop = False
        for epoch in range(num_epochs):
            perm = torch.randperm(self.B, generator=generator).to(self.device)
            for start in range(0, self.B, m):
                batch_idxs = perm[start:start + m]
                optimizer.zero_grad()
                Y = self.forward_solve(y0=self.y0_batch[batch_idxs]) # (m, N, D)
                loss_val = self.compute_loss(Y, self.Y_obs[batch_idxs, :, :]) 
                loss_val.backward()
                optimizer.step()
                update += 1
                if update % max(1, record_mse_every) == 0:
                    mse_record()
                if verbose and update % max(1, log_every) == 0:
                    print(f"[seq-multitraj] epoch {epoch + 1} update {update}: "
                          f"loss={float(loss_val.detach().cpu()):.6e}")
                if time_budget_min is not None and (time.perf_counter() - t_start) / 60.0 >= time_budget_min:
                    stop = True
                    break
            if stop:
                break

        if not mse_history["time_min"] or update % max(1, record_mse_every) != 0:
            mse_record()  # ensure the final state is recorded

        return mse_history

    def get_predicted_trajectory(self,
            y0: torch.Tensor,
            t_grid: torch.Tensor = None,
            rtol: float = None,
            atol: float = None,
            max_step: float = None
    ) -> torch.Tensor:
        """
        Integrate the trained model from a batch of initial conditions y0 (m, D)
        over t_grid, returning (m, N, D). A single (D,) state is promoted to
        (1, D), so a single trajectory comes back as (1, N, D).

        Optional rtol/atol/max_step args override the construction-time tolerances.
        """
        y0 = y0.to(device=self.device, dtype=torch.float32)
        if y0.ndim == 1:
            y0 = y0.unsqueeze(0)  # (D,) -> (1, D)
        self.model.eval()
        with torch.no_grad():
            Y = self.forward_solve(
                y0=y0,
                t_grid=t_grid if t_grid is not None else self.t_grid,
                rtol=rtol, atol=atol, max_step=max_step
            )
        return Y


def train_sequential_multitraj(
    Y_obs: torch.Tensor,
    t_grid: torch.Tensor,
    layer_sizes: list[int],
    num_epochs: int,
    mini_batch_size: int = None,
    lambda_reg: float = 0.0,
    learning_rate: float = 1e-3,
    pretrain_iterations: int = 0,
    pretrain_fraction: float = 0.2,
    Y_obs_test: torch.Tensor = None,
    test_grid: torch.Tensor = None,
    record_mse_every: int = 3,
    time_budget_min: float = None,
    solver_method: str = "dopri5",
    rtol: float = 1e-3,
    atol: float = 1e-6,
    max_step: float = None,
    eval_rtol: float = 1e-3,
    eval_atol: float = 1e-6,
    device: str = "cpu",
    seed: int = None,
    verbose: bool = False,
    log_every: int = 100
) -> tuple["NODESequentialMultiTraj", dict[str, list[float]]]:
    """
    Build and train a sequential Neural ODE jointly on multiple trajectories.

    Y_obs has shape (B, N, state_dim). Set mini_batch_size to B (or None) for
    full-batch Adam optimization. Returns (model, mse_history).
    """
    if seed is not None:
        # seed before constructing the network so weight init is reproducible too
        torch.manual_seed(seed)
        np.random.seed(seed)

    model = NODESequentialMultiTraj(
        Y_obs=Y_obs,
        t_grid=t_grid,
        layer_sizes=layer_sizes,
        lambda_reg=lambda_reg,
        solver_method=solver_method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        device=device
    )
    mse_history = model.train(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        mini_batch_size=mini_batch_size,
        pretrain_iterations=pretrain_iterations,
        pretrain_fraction=pretrain_fraction,
        Y_obs_test=Y_obs_test,
        test_grid=test_grid,
        record_mse_every=record_mse_every,
        time_budget_min=time_budget_min,
        eval_rtol=eval_rtol,
        eval_atol=eval_atol,
        seed=seed,
        verbose=verbose,
        log_every=log_every
    )

    return model, mse_history
