"""Standalone fed-batch bioreactor simulation with piecewise bolus integration."""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.stats.qmc import LatinHypercube


class FedBatchSim:
    """
    Fed-batch bioreactor simulation using piecewise integration for bolus feeds.

    Uses the same kinetics as BioreactorSim but handles glucose feeds as true
    discrete bolus additions between integration segments (no sigmoid smoothing).

    Each batch gets a random number of feeds (variable timing and amounts),
    making this a challenging benchmark for neural ODE approaches.
    """

    X_names: list = ["Scale", "G", "O", "X", "Xd", "P", "L", "CO2"] # ignore scale as a parameter; generally it would be a representation of how process will change across scales
    X_min: np.ndarray = np.array([0.5, 2, 10, 2, 0, 0, 0, 5]) # these are the minimum values for initial conditions
    X_max: np.ndarray = np.array([0.5, 6, 20, 10, 0, 0, 0, 5]) # these are max for initial conditions - to be adjusted based on the kinetics

    def __init__(self, seed: int | None = None):
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)

    def ode(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Rate laws for the cell metabolism model (same as BioreactorSim).

        Args:
            X: state vector [Scale, G, O, X, Xd, P, L, CO2]
            t: current time

        Returns:
            Derivative vector dX/dt
        """
        scale, G, O, X_v, Xd, P, L, CO2 = X

        p = {
            "q_G_max": 0.08,
            "q_GL_max": 100,
            "q_GL1": 5,
            "K_iG": 5,
            "K_s": 1.0,
            "q_O_max": 1,
            "K_O": 1,
            "K_iX": 5,
            "mu_max": 200,
            "K_iCO2": 0.5,
            "K_iO": 5,
            "k_l": 0.01,
            "k_Gd": 0.05,
            "q_P_max": 100,
            "Y_L_G": 1,
            "q_L_max": 10,
            "K_L": 5,
            "K_CO2": 5,
            "K_iL": 2.5,
            "Y_CO2_G": 0.2,
            "Y_CO2_L": 0.2,
            "C_star": 100,
        }

        k_L_a = 0.02 + 0.06 * (1 - 1 / (1 + 1 * np.exp(-(10 * (scale - 0.5)))))
        k_d = 1 / (1 + np.exp(-(100 * (k_L_a - 0.05))))

        q_GL = p["q_GL_max"] / (p["q_GL1"] + L)
        q_G = q_GL * p["q_G_max"] * G / (p["K_s"] + G)
        q_O = p["q_O_max"] * O / (p["K_O"] + O)
        q_L = p["q_L_max"] * L / (p["K_L"] + L)

        mu = (
            p["mu_max"]
            * G
            / (p["K_s"] + G)
            * O
            / (p["K_O"] + O)
            * CO2
            / (p["K_CO2"] + CO2)
            / (
                (1 + (CO2 / (5 * p["K_iCO2"])))
                * (1 + (G / p["K_iG"]))
                * (1 + (O / p["K_iO"]))
                * (1 + X_v / (5 * p["K_iX"]))
            )
        )

        r_via_to_death = k_d * X_v
        r_death_to_lys = p["k_l"] * Xd
        r_growth = mu * X_v

        MT_O = k_L_a * (p["C_star"] - O)
        r_o_cons = q_O * X_v
        r_g_cons = q_G * X_v
        r_L_cons = q_L * X_v

        dScale_dt = 0
        dG_dt = -r_g_cons
        dO_dt = MT_O - r_o_cons
        dX_dt = r_growth - r_via_to_death
        dXd_dt = r_via_to_death - r_death_to_lys
        dL_dt = p["Y_L_G"] * r_g_cons - r_L_cons
        dCO2_dt = (
            p["Y_CO2_G"] * r_g_cons
            + p["Y_CO2_L"] * r_L_cons
            - 2 * CO2 / (p["K_CO2"] + CO2)
        )

        Q_pmax = 10 * X_v / (X_v + Xd + 1e-10) + L / (L + G + 1e-10)
        dP_dt = (Q_pmax / (10 + P) + r_L_cons) / (1 + (Xd / 2.5) ** 2)

        return np.array(
            [dScale_dt, dG_dt, dO_dt, dX_dt, dXd_dt, dP_dt, dL_dt, dCO2_dt]
        )

    def _integrate_batch(
        self,
        x0: np.ndarray,
        time_grid: np.ndarray, # verify that it is between 0 to 1
        feed_schedule: list[tuple[float, float]] | None,
    ) -> np.ndarray:
        """
        Piecewise integration of one batch with bolus feeds.

        Args:
            x0: initial state vector
            time_grid: observation time points
            feed_schedule: list of (time, amount) for G bolus feeds, sorted by time

        Returns:
            Solution array of shape (n_species, n_time_points)
        """
        if feed_schedule is None or len(feed_schedule) == 0:
            solution = odeint(self.ode, x0, time_grid, rtol=1e-4)
            return np.clip(solution.T, 0, None)

        feed_times = [f[0] for f in feed_schedule]
        feed_amounts = [f[1] for f in feed_schedule]

        # Build segment boundaries: [0, feed_1, feed_2, ..., feed_n, 1.0]
        boundaries = sorted(set([time_grid[0]] + feed_times + [time_grid[-1]]))

        state = x0.copy()
        all_solutions = []
        all_times = []

        for seg_idx in range(len(boundaries) - 1):
            t_start = boundaries[seg_idx]
            t_end = boundaries[seg_idx + 1]

            # Get time grid points in this segment
            mask = (time_grid >= t_start) & (time_grid <= t_end)
            t_segment = time_grid[mask]

            # Ensure segment has at least start and end
            if len(t_segment) == 0:
                t_segment = np.array([t_start, t_end])
            elif t_segment[0] > t_start:
                t_segment = np.concatenate([[t_start], t_segment])
            if t_segment[-1] < t_end:
                t_segment = np.concatenate([t_segment, [t_end]])

            # Integrate this segment
            seg_solution = odeint(self.ode, state, t_segment, rtol=1e-4)
            seg_solution = np.clip(seg_solution, 0, None)

            # Collect points that are on the original time grid
            for i, t_val in enumerate(t_segment):
                if np.any(np.isclose(time_grid, t_val, atol=1e-10)):
                    if not any(np.isclose(t_val, at, atol=1e-10) for at in all_times):
                        all_solutions.append(seg_solution[i])
                        all_times.append(t_val)

            # Update state to end of segment
            state = seg_solution[-1].copy()

            # Apply bolus feed if this boundary is a feed time
            if t_end in feed_times:
                feed_idx = feed_times.index(t_end)
                state[1] += feed_amounts[feed_idx]  # G is index 1

        solution = np.array(all_solutions).T
        return solution

    def simulate(
        self,
        n_batch: int = 50,
        n_time_points: int = 15,
        n_feeds_range: tuple[int, int] = (3, 8),
        feed_amount_range: tuple[float, float] = (0.5, 2.0),
        possible_feed_times: np.ndarray | None = None,
        with_feed: bool = True,
        eps: float = 0.0,
    ) -> pd.DataFrame:
        """
        Simulate fed-batch bioreactor with variable feed schedules per batch.

        Args:
            n_batch: number of batches to simulate
            n_time_points: time points per batch in [0, 1]
            n_feeds_range: (min, max) number of feed events per batch
            feed_amount_range: (min, max) glucose amount per bolus
            possible_feed_times: candidate feed times (default: 0.1, 0.2, ..., 0.9)
            with_feed: if False, simulate without any feeds
            eps: multiplicative Gaussian noise level

        Returns:
            DataFrame with columns [Batch, Time, <species>, G_Feed (if with_feed)]
        """
        if possible_feed_times is None:
            possible_feed_times = np.arange(0.1, 1.0, 0.1)

        time_grid = np.linspace(0, 1.0, n_time_points)

        # LHS sampling for initial conditions
        lhs = LatinHypercube(
            d=len(self.X_names), scramble=True, strength=1, optimization="random-cd"
        ).random(n=n_batch)
        initial_conditions = lhs * (self.X_max - self.X_min) + self.X_min

        all_batch_dfs = []

        for batch_i, x0_i in enumerate(initial_conditions):
            # Generate feed schedule for this batch
            feed_schedule = None
            if with_feed:
                n_feeds = np.random.randint(
                    n_feeds_range[0], n_feeds_range[1] + 1
                )
                chosen_times = np.sort(
                    np.random.choice(
                        possible_feed_times, size=n_feeds, replace=False
                    )
                )
                amounts = np.random.uniform(
                    feed_amount_range[0], feed_amount_range[1], size=n_feeds
                )
                feed_schedule = list(zip(chosen_times.tolist(), amounts.tolist()))

            # Integrate with piecewise bolus
            solution = self._integrate_batch(x0_i, time_grid, feed_schedule)

            # Add noise (skip Scale which is constant)
            for species_idx in range(1, solution.shape[0]):
                noise = np.random.normal(scale=eps, size=solution.shape[1])
                solution[species_idx] *= 1 + noise
            solution = np.clip(solution, 0, None)

            # Build batch DataFrame
            batch_name = f"Batch_{batch_i}"
            df = pd.DataFrame()
            df["Batch"] = [batch_name] * n_time_points
            df["Time"] = time_grid
            for col_idx, col_name in enumerate(self.X_names):
                df[col_name] = solution[col_idx]

            # Add feed data
            if with_feed and feed_schedule:
                df_feeds = pd.DataFrame()
                df_feeds["Time"] = [f[0] for f in feed_schedule]
                df_feeds["Batch"] = [batch_name] * len(feed_schedule)
                df_feeds["G_Feed"] = [f[1] for f in feed_schedule]
                df = df.merge(
                    df_feeds, on=["Time", "Batch"], how="outer"
                ).sort_values("Time")

            all_batch_dfs.append(df)

        result = pd.concat(all_batch_dfs).reset_index(drop=True)
        numeric_cols = ["Time"] + self.X_names
        if "G_Feed" in result.columns:
            numeric_cols.append("G_Feed")
        result[numeric_cols] = result[numeric_cols].astype("double")

        return result


if __name__ == "__main__":
    import os
    N_BATCH = 10000
    N_TIME_POINTS = 30
    EPS = 0.005  # lowest noise setting

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(
        out_dir, f"{N_BATCH}trajectories_{N_TIME_POINTS}points_{EPS}noise.csv"
    )

    sim = FedBatchSim()  # no seed -> output differs each run
    df = sim.simulate(
        n_batch=N_BATCH,
        n_time_points=N_TIME_POINTS,
        with_feed=False,
        eps=EPS,
    )
    df["Noise_Level"] = EPS

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows ({N_BATCH} batches) to {out_path}")
