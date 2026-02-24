"""Multi-DC Gymnasium environment for geo-distributed energy optimization.

The agent receives incoming VM demand each timestep and decides how to
distribute it across N data centers.  Each DC has local solar availability,
electricity prices, and a power model.  The goal is to minimize total grid
energy cost while maintaining service quality (low backlog).

When *batch_enabled* is True the environment splits each DC's demand into
an immediate service component and a deferrable batch component.  The agent
then controls both **spatial routing** (which DC) and **temporal scheduling**
(when to drain the batch pool) to align batch execution with renewable
availability and low prices.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.dc_site import DataCenterSite
from env.power_model import PowerModel


class MultiDCEnv(gym.Env):
    """Gymnasium environment for multi-DC workload routing.

    **Legacy mode** (batch_enabled=False):
        Observation: 5*N + 2
        Action:      N  (spatial routing only)

    **Batch mode** (batch_enabled=True):
        Observation: 7*N + 3
        Action:      2*N (spatial routing + temporal drain rates)
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
        # Batch scheduling parameters
        batch_enabled: bool = False,
        flexibility_factor: float = 1.0,
        deadline_penalty_weight: float = 2.0,
        urgency_horizon_steps: int = 12,
        interval_seconds: int = 300,
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

        # Batch parameters
        self.batch_enabled = batch_enabled
        self.flexibility_factor = flexibility_factor
        self.deadline_penalty_weight = deadline_penalty_weight
        self.urgency_horizon_steps = urgency_horizon_steps
        self.interval_seconds = interval_seconds

        # Precompute per-site deadline offsets (in timesteps)
        self._deadline_offsets: list[int] = []
        if self.batch_enabled:
            for site in self.sites:
                offset = max(
                    1,
                    math.ceil(
                        site.batch_mean_duration_sec
                        * (1.0 + self.flexibility_factor)
                        / self.interval_seconds
                    ),
                )
                self._deadline_offsets.append(offset)

        # Observation & action spaces
        if self.batch_enabled:
            obs_dim = 7 * self.n_dc + 3
            action_dim = 2 * self.n_dc
        else:
            obs_dim = 5 * self.n_dc + 2
            action_dim = self.n_dc

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        self.step_index = 0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.step_index = 0
        for site in self.sites:
            site.reset()
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step (dispatcher)
    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.batch_enabled:
            return self._step_batch(action)
        return self._step_legacy(action)

    # ------------------------------------------------------------------
    # Legacy step (unchanged logic)
    # ------------------------------------------------------------------

    def _step_legacy(
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
            energy_cost = site.get_price(t) * grid_mw * 1000.0 * interval_hours

            # Backlog penalty
            backlog_cost = self.backlog_weight * new_backlog

            # Capacity violation penalty
            cap_penalty = self.capacity_penalty_weight * max(
                0.0, served - site.capacity
            )

            dc_cost = energy_cost + backlog_cost + cap_penalty

            # Update site state
            site.backlog = new_backlog
            site.current_load = served

            total_cost += dc_cost
            total_renewable_used += renewable_used
            total_power += power_mw

            info_per_dc.append(
                {
                    "name": site.name,
                    "assigned": float(assigned),
                    "served": float(served),
                    "backlog": float(new_backlog),
                    "grid_mw": float(grid_mw),
                    "solar_mw": float(solar_mw),
                    "energy_cost": float(energy_cost),
                }
            )

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

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Batch step (temporal + spatial scheduling)
    # ------------------------------------------------------------------

    def _step_batch(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        t = self.step_index
        N = self.n_dc
        action = np.asarray(action, dtype=np.float64)

        # Parse action: first N = spatial routing, last N = batch drain rates
        routing_logits = action[:N]
        drain_logits = action[N:]

        # Spatial routing via softmax
        exp_a = np.exp(routing_logits - routing_logits.max())
        fractions = exp_a / exp_a.sum()

        # Batch drain rates via sigmoid: map [-1, 1] -> (0, 1)
        drain_rates = 1.0 / (1.0 + np.exp(-drain_logits))

        # --- Phase 1: Inject new batch demand into pools ---
        for i, site in enumerate(self.sites):
            new_batch = site.get_batch_demand(t)
            if new_batch > 0:
                site.batch_pool.add(new_batch, t + self._deadline_offsets[i])

        # --- Phase 2: Expire overdue batch entries (deadline violations) ---
        expired_per_dc = []
        for site in self.sites:
            expired_per_dc.append(site.batch_pool.expire(t))

        # --- Phase 3: Drain batch pools ---
        batch_drained = []
        for i, site in enumerate(self.sites):
            batch_drained.append(site.batch_pool.drain(drain_rates[i]))

        # --- Phase 4: Route service demand spatially ---
        total_service = sum(s.get_service_demand(t) for s in self.sites)

        # --- Phase 5: Per-DC cost computation ---
        total_cost = 0.0
        total_renewable_used = 0.0
        total_power = 0.0
        info_per_dc = []

        for i, site in enumerate(self.sites):
            # Service demand assigned to this DC
            service_assigned = fractions[i] * total_service
            service_to_serve = service_assigned + site.backlog

            # Total work this timestep: service + batch drained
            batch_work = batch_drained[i]
            total_work = service_to_serve + batch_work

            # Serve up to capacity — service gets priority
            served = min(total_work, site.capacity)
            service_served = min(service_to_serve, served)
            batch_served = served - service_served
            new_backlog = service_to_serve - service_served

            # Return unserved batch to pool (preserving urgency via near deadline)
            batch_unserved = batch_work - batch_served
            if batch_unserved > 1e-9:
                site.batch_pool.add(batch_unserved, t + 1)

            # Power consumption (on total served work)
            power_util = self.power_model.compute(served)
            power_mw = power_util * site.rated_power_mw

            # Solar supply
            solar_mw = site.solar_capacity_mw * site.get_solar_fraction(t)

            # Grid power
            grid_mw = max(0.0, power_mw - solar_mw)
            renewable_used = min(power_mw, solar_mw)

            # Energy cost for this 5-min interval
            interval_hours = 5.0 / 60.0
            energy_cost = site.get_price(t) * grid_mw * 1000.0 * interval_hours

            # Service backlog penalty
            backlog_cost = self.backlog_weight * new_backlog

            # Capacity violation penalty
            cap_penalty = self.capacity_penalty_weight * max(
                0.0, served - site.capacity
            )

            # Deadline violation penalty
            deadline_cost = self.deadline_penalty_weight * expired_per_dc[i]

            dc_cost = energy_cost + backlog_cost + cap_penalty + deadline_cost

            # Update site state
            site.backlog = new_backlog
            site.current_load = served

            total_cost += dc_cost
            total_renewable_used += renewable_used
            total_power += power_mw

            info_per_dc.append(
                {
                    "name": site.name,
                    "service_assigned": float(service_assigned),
                    "service_served": float(service_served),
                    "batch_drained": float(batch_work),
                    "batch_served": float(batch_served),
                    "served": float(served),
                    "backlog": float(new_backlog),
                    "batch_pool_size": float(site.batch_pool.total_demand),
                    "batch_expired": float(expired_per_dc[i]),
                    "drain_rate": float(drain_rates[i]),
                    "grid_mw": float(grid_mw),
                    "solar_mw": float(solar_mw),
                    "energy_cost": float(energy_cost),
                    "deadline_cost": float(deadline_cost),
                }
            )

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
            "total_demand": float(total_service),
            "total_batch_expired": float(sum(expired_per_dc)),
            "total_batch_pool": float(
                sum(s.batch_pool.total_demand for s in self.sites)
            ),
            "fractions": fractions.tolist(),
            "drain_rates": drain_rates.tolist(),
            "total_cost": float(total_cost),
            "renewable_frac": float(renewable_frac),
            "per_dc": info_per_dc,
        }

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        t = self.step_index
        obs_parts: list[float] = []

        if self.batch_enabled:
            total_service = 0.0
            total_batch_pool = 0.0
            for site in self.sites:
                svc = site.get_service_demand(t)
                total_service += svc
                pool_size = site.batch_pool.total_demand
                total_batch_pool += pool_size
                urgency = site.batch_pool.urgency(
                    t, self.urgency_horizon_steps
                )
                obs_parts.extend(
                    [
                        svc,
                        pool_size,
                        urgency,
                        site.backlog,
                        site.get_price(t),
                        site.get_solar_fraction(t),
                        site.current_load,
                    ]
                )
            hour_of_day = (t % self.steps_per_day) / self.steps_per_day
            obs_parts.extend([total_service, total_batch_pool, hour_of_day])
        else:
            total_demand = 0.0
            for site in self.sites:
                demand = site.get_local_demand(t)
                total_demand += demand
                obs_parts.extend(
                    [
                        demand,
                        site.backlog,
                        site.get_price(t),
                        site.get_solar_fraction(t),
                        site.current_load,
                    ]
                )
            hour_of_day = (t % self.steps_per_day) / self.steps_per_day
            obs_parts.extend([total_demand, hour_of_day])

        return np.array(obs_parts, dtype=np.float32)
