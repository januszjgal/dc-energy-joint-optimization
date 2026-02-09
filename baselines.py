"""Baseline policies for comparison against the trained PPO agent.

Each baseline implements a `predict(obs, env)` method returning an action
in the same format as the MultiDCEnv action space.
"""

from __future__ import annotations

import numpy as np

from env.multi_dc_env import MultiDCEnv


class RoundRobinPolicy:
    """Equal allocation to all DCs at every timestep."""

    name = "Round Robin"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        return np.ones(env.n_dc, dtype=np.float32) / env.n_dc


class CheapestFirstPolicy:
    """Route all demand to the DC with the lowest current energy price."""

    name = "Cheapest Price First"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        prices = [site.get_price(t) for site in env.sites]
        action = np.zeros(env.n_dc, dtype=np.float32)
        action[int(np.argmin(prices))] = 1.0
        return action


class FollowTheSunPolicy:
    """Route all demand to the DC with the highest current solar availability."""

    name = "Follow the Sun"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        solar = [site.get_solar_fraction(t) for site in env.sites]
        action = np.zeros(env.n_dc, dtype=np.float32)
        action[int(np.argmax(solar))] = 1.0
        return action


class LocalOnlyPolicy:
    """Each DC handles only its own cell's workload -- no cross-DC routing.

    The allocation matches each DC's proportional share of total demand.
    """

    name = "Local Only (No Routing)"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        demands = np.array(
            [site.get_local_demand(t) for site in env.sites], dtype=np.float32
        )
        total = demands.sum()
        if total > 0:
            return demands / total
        return np.ones(env.n_dc, dtype=np.float32) / env.n_dc


class RandomPolicy:
    """Uniform random allocation each timestep."""

    name = "Random"

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        raw = self.rng.random(env.n_dc).astype(np.float32)
        return raw / raw.sum()


ALL_BASELINES = [
    RoundRobinPolicy,
    CheapestFirstPolicy,
    FollowTheSunPolicy,
    LocalOnlyPolicy,
    RandomPolicy,
]
