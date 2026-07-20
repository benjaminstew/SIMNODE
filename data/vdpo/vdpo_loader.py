import numpy as np
from scipy.integrate import solve_ivp
from src.models.direct_collocation import BarycentricInterpolation

class VanDerPolOscillator:
    """
    The forced Van der Pol oscillator - a 2D system represented by a
    second-order ODE, which can be split into two first-order ODEs. The ODE describes 
    a system with nonlinear damping of degree μ. 
    If μ = 0 the system is linear and undamped, and we have a simple harmonic oscillator.
    TIME INVARIANT STATE REPRESENTATION ONLY.
    """
    def __init__(self, mu=1.0, external_force=1.0, angular_freq=1.0, u_0=0.0, v_0=1.0):
        self.mu = mu # nonlinearity/damping parameter
        self.external_force = external_force # external periodic force parameter
        self.angular_freq = angular_freq
        self.u_0 = u_0 # initial displacement
        self.v_0 = v_0 # initial velocity

    def get_initial_state(self):
        '''
        Returns the initial state vector of shape (2, ) containing the
        initial displacement (u_0) and initial velocity (v_0).
        '''
        return np.array([self.u_0, self.v_0], dtype=np.float32)

    def get_dynamics(self, t, y):
        '''The sequential solver calls this method at each time step t to update y.'''
        u = y[0]
        v = y[1]
        dudt = v
        dvdt = self.mu * (1 - u**2) * v - u + self.external_force * np.cos(self.angular_freq * t)

        return np.array([dudt, dvdt], dtype=np.float32)
    
def create_vdpo_dataset(num_nodes, end_time, add_noise=True, noise_sd=None): 
    '''
    Computes single trajectory of VDPO state over [0, end_time] and splits into 50/50 train set and test set.
    
    Train grid: chebyshev 2nd kind nodes on [0, end_time/2] (for collocation).
    Test grid: chebyshev 2nd kind nodes on [end_time/2, end_time] (for evaluation).
    
    The IVP is solved on the joint (sorted) grid so that train and test trajectories are consistent 
    (same continuous solution). 
    '''
    if num_nodes % 2 != 0 or end_time % 2 != 0:
        raise ValueError("num_nodes and end_time must be even so train/test splits are the same size.")
    half_nodes = num_nodes // 2
    half_time = end_time // 2

    #1. Create the system 
    system = VanDerPolOscillator()

    #2. Build train and test grids 
    train_interpol = BarycentricInterpolation(0, half_time, half_nodes)
    test_interpol  = BarycentricInterpolation(half_time, end_time, half_nodes)
    train_grid = train_interpol.colloc_grid # (half_nodes, )
    test_grid  = test_interpol.colloc_grid # (half_nodes, )

    #3. Merge into a single strictly-increasing grid and solve the IVP once
    full_grid = np.unique(np.concatenate([train_grid, test_grid]))  # sorted + deduplicated
    y_0 = system.get_initial_state()
    solution = solve_ivp(
        fun=system.get_dynamics,
        t_span=(full_grid[0], full_grid[-1]),
        y0=y_0,
        t_eval=full_grid,
        method='RK45'
    )
    full_trajectory = solution.y.T # (num_nodes, state_dim)
    full_smooth_trajectory = full_trajectory.copy()

    #4. Extract train/test rows by matching grid values
    train_idx = np.isin(full_grid, train_grid)
    test_idx  = np.isin(full_grid, test_grid)
    train_trajectory = full_trajectory[train_idx, :] # (half_nodes, state_dim)
    train_smooth_trajectory = full_smooth_trajectory[train_idx, :]
    test_trajectory = full_trajectory[test_idx, :] # (half_nodes, state_dim)

    #5. Add Gaussian noise to training data only
    if add_noise and noise_sd is not None and noise_sd > 0:
        noise = np.random.randn(*train_trajectory.shape).astype(np.float32) * float(noise_sd)
        train_trajectory += noise

    return train_interpol, train_grid, train_trajectory, test_interpol, test_grid, test_trajectory, full_grid, full_smooth_trajectory, train_smooth_trajectory