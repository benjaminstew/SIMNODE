'''
PyTorch implementation of a sequential Neural ODE 
training (solver-in-the-loop backpropagation) baseline.
'''
import math
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torchdiffeq import odeint
from scipy.interpolate import CubicSpline


class FeedForwardNN(nn.Module):
    """Feedforward MLP with tanh hidden layers and linear output layer."""
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )
        self.init_parameters()

    def init_parameters(self) -> None:
        # Xavier/Glorot normal for weights; uniform bias 
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            input_dim = layer.weight.shape[1]
            bound = 1.0 / math.sqrt(input_dim) 
            nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


class NODESequential:
    """
    Sequential Neural ODE training via solver-in-the-loop backpropagation.
    Y is a single trajectory of shape (N, state_dim).
    
    Implements the algorithm described in Section 2.2 of the JPC paper:
      1. Forward solve: Y = ODESolve(f_theta, y0, t)
      2. Loss: MSE (1/N)||Y - Yobs||_F^2 + l2 reg
      3. Gradients: backprop through torchdiffeq (discrete adjoint)
      4. Update: Adam

    Optional supervised pre-training on estimated derivatives can
    substantially improve convergence over random initialization.
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
        if Y_obs.ndim != 2:
            raise ValueError("Y_obs must be a 2D tensor of shape (N, state_dim).")
        if t_grid.ndim != 1:
            raise ValueError("t_grid must be a 1D tensor.")
        if t_grid.shape[0] != Y_obs.shape[0]:
            raise ValueError("t_grid length must match Y_obs rows.")
        if torch.any(torch.diff(t_grid) <= 0):
            raise ValueError("t_grid must be strictly increasing.")

        self.N, self.state_dim = Y_obs.shape
        self.lambda_reg = lambda_reg
        self.solver_method = solver_method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.device = torch.device(device)
        self.t_grid = t_grid.to(device=self.device, dtype=torch.float32)
        self.Y_obs = Y_obs.to(device=self.device, dtype=torch.float32)

        sizes = list(layer_sizes)
        if sizes[0] != self.state_dim:
            raise ValueError(
                f"layer_sizes[0] must equal state_dim "
                f"({self.state_dim})."
            )
        if sizes[-1] != self.state_dim:
            raise ValueError("layer_sizes[-1] must equal state_dim.")

        self.model = FeedForwardNN(sizes).to(self.device)

    def ode_rhs(self, t: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        """RHS f_theta(y, t) called by the IVP solver."""
        return self.model(y0)

    def forward_solve(self,
        t_grid: torch.Tensor = None,
        rtol: float = None,
        atol: float = None,
        max_step: float = None
    ) -> torch.Tensor:
        """
        Integrate the Neural ODE from y0 over t_grid and return Y (N, state_dim).

        rtol/atol/max_step override the construction-time tolerances for this call when
        provided (e.g. tighter tolerances at inference); otherwise the instance defaults
        are used.
        """
        y0 = self.Y_obs[0]
        if y0.dim() != 1:
            raise ValueError("y0 must be a 1D tensor.")

        t_eval = self.t_grid if t_grid is None else t_grid.to(device=self.device, dtype=torch.float32)
        rtol = self.rtol if rtol is None else float(rtol)
        atol = self.atol if atol is None else float(atol)
        max_step = self.max_step if max_step is None else float(max_step)
        options = None if max_step is None else {"max_step": float(max_step)}

        return odeint(
            self.ode_rhs,
            y0,
            t_eval,
            method=self.solver_method,
            rtol=rtol,
            atol=atol,
            options=options
        ) # shape (N, state_dim)

    def compute_loss(self, Y: torch.Tensor) -> torch.Tensor:
        """(1/N) * ||Y - Yobs||_F^2 + l2 reg"""
        if Y.shape != self.Y_obs.shape:
            raise ValueError("Predicted trajectory shape must match Y_obs.")
        mse = (Y - self.Y_obs).pow(2).sum() / float(self.N)
        if self.lambda_reg <= 0.0:
            return mse
        l2 = sum(p.pow(2).sum() for p in self.model.parameters())
        return mse + self.lambda_reg * l2

    def pretrain(self,
        num_iterations: int,
        learning_rate: float = 1e-3,
        fraction: float = 0.2,
        verbose: bool = False,
        log_every: int = 100,
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
        Y_pre = self.Y_obs[:n_pre] # (n_pre, state_dim)

        t_pre_np = t_pre.detach().cpu().numpy()
        Y_pre_np = Y_pre.detach().cpu().numpy()
        spline = CubicSpline(t_pre_np, Y_pre_np, axis=0)
        dY_target_np = spline(t_pre_np, 1).astype(np.float32) # first derivative 

        dY_target = torch.tensor(dY_target_np, device=self.device, dtype=torch.float32)  # (n_pre, state_dim)

        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        loss_history: list[float] = []

        self.model.train() # set NN to training mode 
        for iteration in range(num_iterations):
            optimizer.zero_grad() # clear grads from prev step
            dY_pred = self.model(Y_pre) # (n_pre, state_dim)
            loss_val = (dY_pred - dY_target).pow(2).mean()
            loss_val.backward() # backprop
            optimizer.step() # update params

            loss_scalar = float(loss_val.detach().cpu())
            loss_history.append(loss_scalar)
            if verbose and (iteration + 1) % max(1, log_every) == 0:
                print(f"  [pretrain] iter {iteration + 1}/{num_iterations}: loss={loss_scalar:.6f}")

        return loss_history

    def train(self,
        num_iterations: int,
        learning_rate: float = 1e-3,
        pretrain_iterations: int = 0,
        pretrain_fraction: float = 0.2,
        verbose: bool = False,
        log_every: int = 100,
    ) -> dict[str, list[float]]:
        """
        Train via solver-in-the-loop backpropagation.

        Args:
        - num_iterations : int
            Number of end-to-end Adam steps.
        - learning_rate : float
            Adam learning rate (default 1e-3, per Appendix A.2).
        - pretrain_iterations : int
            Adam steps for supervised derivative pre-training (0 = disabled).
            Setting this to ~1000 iterations typically resolves convergence
            difficulties when starting from random initialisation.
        - pretrain_fraction : float
            Fraction of training data used to estimate derivatives for pre-training.
        - verbose : bool
            Print loss at regular intervals.
        - log_every : int
            Print frequency (in iterations).

        Returns:
        dict with keys 'pretrain' and 'train', each containing a loss history list.
        """
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")

        if pretrain_iterations > 0:
            if verbose:
                print(
                    f"Pre-training for {pretrain_iterations} iterations "
                    f"on first {pretrain_fraction * 100:.0f}% of observations ..."
                )
            pretrain_loss_history = self.pretrain(
                num_iterations=pretrain_iterations,
                learning_rate=learning_rate,
                fraction=pretrain_fraction,
                verbose=verbose,
                log_every=log_every
            )

        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        train_loss_history: list[float] = []

        y0 = self.Y_obs[0] 
        self.model.train()
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            Y = self.forward_solve(y0)
            loss_val = self.compute_loss(Y)
            loss_val.backward()
            optimizer.step()

            loss_scalar = float(loss_val.detach().cpu())
            train_loss_history.append(loss_scalar)
            if verbose and (iteration + 1) % max(1, log_every) == 0:
                print(f"Iter {iteration + 1}/{num_iterations}: loss={loss_scalar:.6f}")

        return {"pretrain": pretrain_loss_history, "train": train_loss_history}

    def get_predicted_trajectory(self,
        y0: torch.Tensor,
        t_grid: torch.Tensor = None,
        rtol: float = None,
        atol: float = None,
        max_step: float = None
    ) -> torch.Tensor:
        """
        Integrate the trained model from y0 over t_grid (eval mode, no grad).

        Optional rtol/atol/max_step args override the construction-time tolerances.
        """
        self.model.eval() # switch to eval mode 
        with torch.no_grad():
            Y = self.forward_solve(
                y0=y0, 
                t_grid=t_grid if t_grid is not None else self.t_grid, 
                rtol=rtol, atol=atol, max_step=max_step
            )
        return Y

def train_sequential_node(
    Y_obs: torch.Tensor,
    t_grid: torch.Tensor,
    layer_sizes: list[int],
    num_iterations: int,
    lambda_reg: float = 0.0,
    learning_rate: float = 1e-3,
    pretrain_iterations: int = 0,
    pretrain_fraction: float = 0.2,
    solver_method: str = "dopri5",
    rtol: float = 1e-3,
    atol: float = 1e-6,
    max_step: float = None,
    device: str = "cpu",
    verbose: bool = False,
    log_every: int = 100
) -> tuple["NODESequential", dict[str, list[float]]]:
    """
    Build and train a sequential Neural ODE on a single trajectory.

    Pass pretrain_iterations > 0 to run supervised derivative pre-training first.
    """
    model = NODESequential(
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
    train_loss_history = model.train(
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        pretrain_iterations=pretrain_iterations,
        pretrain_fraction=pretrain_fraction,
        verbose=verbose,
        log_every=log_every
    )

    return model, train_loss_history

