"""Baseline policies for comparison against the trained PPO agent.

Each baseline implements a `predict(obs, env)` method returning an action
in the same format as the MultiDCEnv action space.
"""

from __future__ import annotations

import numpy as np

from env.multi_dc_env import MultiDCEnv


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _with_drain(routing: np.ndarray, drain: np.ndarray, env: MultiDCEnv) -> np.ndarray:
    """Append drain logits to routing logits when batch mode is active."""
    if getattr(env, "batch_enabled", False):
        return np.concatenate([routing, drain.astype(np.float32)])
    return routing


# ------------------------------------------------------------------
# Existing baselines (extended for batch mode)
# ------------------------------------------------------------------


class RoundRobinPolicy:
    """Equal allocation to all DCs at every timestep."""

    name = "Round Robin"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        routing = np.zeros(env.n_dc, dtype=np.float32)
        # sigmoid(0) = 0.5 → drain half the batch pool each step
        drain = np.zeros(env.n_dc, dtype=np.float32)
        return _with_drain(routing, drain, env)


class CheapestFirstPolicy:
    """Route all demand to the DC with the lowest current energy price."""

    name = "Cheapest Price First"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        prices = np.array(
            [site.get_price(t) for site in env.sites], dtype=np.float32
        )
        routing = np.full(env.n_dc, -1.0, dtype=np.float32)
        routing[int(np.argmin(prices))] = 1.0

        # Drain more at cheap DCs, less at expensive ones
        if prices.max() > prices.min():
            norm = (prices - prices.min()) / (prices.max() - prices.min())
            drain = (1.0 - norm) * 2.0 - 1.0  # cheap → +1, expensive → -1
        else:
            drain = np.zeros(env.n_dc, dtype=np.float32)
        return _with_drain(routing, drain, env)


class FollowTheSunPolicy:
    """Route all demand to the DC with the highest current solar availability."""

    name = "Follow the Sun"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        solar = np.array(
            [site.get_solar_fraction(t) for site in env.sites], dtype=np.float32
        )
        routing = np.full(env.n_dc, -1.0, dtype=np.float32)
        routing[int(np.argmax(solar))] = 1.0

        # Drain proportional to solar availability
        if solar.max() > 0:
            drain = solar / solar.max() * 2.0 - 1.0
        else:
            drain = np.full(env.n_dc, -1.0, dtype=np.float32)
        return _with_drain(routing, drain, env)


class LocalOnlyPolicy:
    """Each DC handles only its own cell's workload -- no cross-DC routing."""

    name = "Local Only (No Routing)"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        demands = np.array(
            [site.get_local_demand(t) for site in env.sites], dtype=np.float32
        )
        total = demands.sum()
        if total > 0:
            fracs = demands / total
            fracs = np.clip(fracs, 1e-6, None)
            logits = np.log(fracs)
            logits = logits - logits.mean()
            routing = np.clip(logits, -1.0, 1.0).astype(np.float32)
        else:
            routing = np.zeros(env.n_dc, dtype=np.float32)

        # Drain most immediately (no temporal optimization)
        drain = np.ones(env.n_dc, dtype=np.float32)  # sigmoid(1) ≈ 0.73
        return _with_drain(routing, drain, env)


class RandomPolicy:
    """Uniform random allocation each timestep."""

    name = "Random"

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        if getattr(env, "batch_enabled", False):
            return self.rng.uniform(-1.0, 1.0, size=2 * env.n_dc).astype(
                np.float32
            )
        return self.rng.uniform(-1.0, 1.0, size=env.n_dc).astype(np.float32)


# ------------------------------------------------------------------
# New temporal baselines (batch mode only, but safe in legacy mode)
# ------------------------------------------------------------------


class DrainImmediatelyPolicy:
    """Route evenly, drain all batch work immediately.

    Isolates the value of temporal scheduling: this baseline does no
    temporal optimization, so any improvement by PPO comes from learning
    *when* to execute batch work.
    """

    name = "Drain Immediately"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        routing = np.zeros(env.n_dc, dtype=np.float32)
        # sigmoid(5.0) ≈ 0.993 → drain almost everything
        drain = np.full(env.n_dc, 5.0, dtype=np.float32)
        return _with_drain(routing, drain, env)


class DeferToSunPolicy:
    """Defer batch work to periods of high solar availability.

    A heuristic temporal strategy: drain proportional to how much sun
    is available right now.  When it's dark, accumulate batch work in
    the pool; when the sun is up, drain aggressively.
    """

    name = "Defer to Sun"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        solar = np.array(
            [site.get_solar_fraction(t) for site in env.sites], dtype=np.float32
        )
        routing = np.zeros(env.n_dc, dtype=np.float32)
        # Map solar [0, 1] to sigmoid input [-3, 3]:
        #   0 sun → sigmoid(-3) ≈ 0.05 (hold)
        #   1 sun → sigmoid(+3) ≈ 0.95 (drain)
        drain = solar * 6.0 - 3.0
        return _with_drain(routing, drain, env)


ALL_BASELINES = [
    RoundRobinPolicy,
    CheapestFirstPolicy,
    FollowTheSunPolicy,
    LocalOnlyPolicy,
    RandomPolicy,
    DrainImmediatelyPolicy,
    DeferToSunPolicy,
]
