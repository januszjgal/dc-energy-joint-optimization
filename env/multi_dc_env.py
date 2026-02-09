"""Multi-DC Gymnasium environment for geo-distributed energy optimization.

The agent receives incoming VM demand each timestep and decides how to
distribute it across N data centers.  Each DC has local solar availability,
electricity prices, and a power model.  The goal is to minimize total grid
energy cost while maintaining service quality (low backlog).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.dc_site import DataCenterSite
from env.power_model import PowerModel


class MultiDCEnv(gym.Env):
    """Gymnasium environment for multi-DC workload routing.

    Observation space (per DC: 5 features, global: 2):
        Per DC: [cpu_demand_local, backlog, energy_price, solar_fraction, current_load]
        Global: [total_incoming_demand, hour_of_day_normalized]
        Total:  5 * N + 2

    Action space: Box(0, 1, shape=(N,))
        Allocation fractions for each DC (softmax-normalized internally).

    Reward:
        Negative of total cost across all DCs:
        cost = sum over DCs of (price * grid_power + backlog_weight * backlog)
        bonus for renewable utilization
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sites: list[DataCenterSite],
        power_model: PowerModel,
        max_steps: int | None = None,
        backlog_weight: float = 1.5,
        capacity_penalty_weight: float = 5.0,
        renewable_bonus_weight: float = 0.2,
        steps_per_day: int = 288,
    ):
        super().__init__()

        self.sites = sites
        self.power_model = power_model
        self.n_dc = len(sites)

        # Determine episode length
        min_timesteps = min(s.num_timesteps for s in sites)
        self.max_steps = max_steps if max_steps else min_timesteps

        # Reward weights
        self.backlog_weight = backlog_weight
        self.capacity_penalty_weight = capacity_penalty_weight
        self.renewable_bonus_weight = renewable_bonus_weight
        self.steps_per_day = steps_per_day

        # Observation: 5 features per DC + 2 global
        obs_dim = 5 * self.n_dc + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: raw values in [-1, 1] (SB3-recommended symmetric range).
        # Mapped to allocation fractions via softmax in step().
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_dc,), dtype=np.float32
        )

        self.step_index = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.step_index = 0
        for site in self.sites:
            site.reset()
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        t = self.step_index

        # Convert [-1, 1] actions to allocation fractions via softmax
        action = np.asarray(action, dtype=np.float64)
        exp_a = np.exp(action - action.max())  # numerically stable softmax
        fractions = exp_a / exp_a.sum()

        # Total incoming demand = sum of all cells' demand at this timestep
        total_demand = sum(s.get_local_demand(t) for s in self.sites)

        # Distribute demand + process backlogs
        total_cost = 0.0
        total_renewable_used = 0.0
        total_power = 0.0
        info_per_dc = []

        for i, site in enumerate(self.sites):
            # Demand assigned to this DC
            assigned = fractions[i] * total_demand
            total_to_serve = assigned + site.backlog

            # Serve what we can (up to capacity)
            served = min(total_to_serve, site.capacity)
            new_backlog = total_to_serve - served

            # Power consumption
            power_util = self.power_model.compute(served)
            power_mw = power_util * site.rated_power_mw

            # Solar supply
            solar_mw = site.solar_capacity_mw * site.get_solar_fraction(t)

            # Grid power (what we need from the grid)
            grid_mw = max(0.0, power_mw - solar_mw)
            renewable_used = min(power_mw, solar_mw)

            # Energy cost for this 5-min interval
            interval_hours = 5.0 / 60.0  # 5 minutes in hours
            energy_cost = site.get_price(t) * grid_mw * 1000.0 * interval_hours  # price is $/kWh

            # Backlog penalty
            backlog_cost = self.backlog_weight * new_backlog

            # Capacity violation penalty
            cap_penalty = self.capacity_penalty_weight * max(0.0, served - site.capacity)

            dc_cost = energy_cost + backlog_cost + cap_penalty

            # Update site state
            site.backlog = new_backlog
            site.current_load = served

            total_cost += dc_cost
            total_renewable_used += renewable_used
            total_power += power_mw

            info_per_dc.append({
                "name": site.name,
                "assigned": float(assigned),
                "served": float(served),
                "backlog": float(new_backlog),
                "grid_mw": float(grid_mw),
                "solar_mw": float(solar_mw),
                "energy_cost": float(energy_cost),
            })

        # Renewable bonus
        renewable_frac = (
            total_renewable_used / total_power if total_power > 0 else 0.0
        )
        renewable_bonus = self.renewable_bonus_weight * renewable_frac

        reward = -total_cost + renewable_bonus

        self.step_index += 1
        terminated = self.step_index >= self.max_steps
        truncated = False

        info = {
            "total_demand": float(total_demand),
            "fractions": fractions.tolist(),
            "total_cost": float(total_cost),
            "renewable_frac": float(renewable_frac),
            "per_dc": info_per_dc,
        }

        obs = self._get_obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        t = self.step_index
        obs_parts = []

        total_demand = 0.0
        for site in self.sites:
            demand = site.get_local_demand(t)
            total_demand += demand
            obs_parts.extend([
                demand,
                site.backlog,
                site.get_price(t),
                site.get_solar_fraction(t),
                site.current_load,
            ])

        # Global features
        hour_of_day = (t % self.steps_per_day) / self.steps_per_day  # [0, 1)
        obs_parts.extend([total_demand, hour_of_day])

        return np.array(obs_parts, dtype=np.float32)
