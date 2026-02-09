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
        # Equal logits → uniform softmax fractions
        return np.zeros(env.n_dc, dtype=np.float32)


class CheapestFirstPolicy:
    """Route all demand to the DC with the lowest current energy price."""

    name = "Cheapest Price First"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        prices = [site.get_price(t) for site in env.sites]
        # Large logit at cheapest DC so softmax concentrates allocation there
        action = np.full(env.n_dc, -1.0, dtype=np.float32)
        action[int(np.argmin(prices))] = 1.0
        return action


class FollowTheSunPolicy:
    """Route all demand to the DC with the highest current solar availability."""

    name = "Follow the Sun"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        solar = [site.get_solar_fraction(t) for site in env.sites]
        # Large logit at sunniest DC
        action = np.full(env.n_dc, -1.0, dtype=np.float32)
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
            # Convert desired fractions to log-space (inverse of softmax)
            fracs = demands / total
            fracs = np.clip(fracs, 1e-6, None)
            logits = np.log(fracs)
            # Center within [-1, 1]
            logits = logits - logits.mean()
            logits = np.clip(logits, -1.0, 1.0)
            return logits.astype(np.float32)
        return np.zeros(env.n_dc, dtype=np.float32)


class RandomPolicy:
    """Uniform random allocation each timestep."""

    name = "Random"

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        # Random logits in [-1, 1]
        return self.rng.uniform(-1.0, 1.0, size=env.n_dc).astype(np.float32)


ALL_BASELINES = [
    RoundRobinPolicy,
    CheapestFirstPolicy,
    FollowTheSunPolicy,
    LocalOnlyPolicy,
    RandomPolicy,
]
