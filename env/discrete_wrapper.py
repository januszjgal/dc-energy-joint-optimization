"""Discrete action wrapper for DQN training on the MultiDCEnv.

DQN requires a discrete action space. This wrapper maps a finite set of
canonical routing allocations to the continuous action space of MultiDCEnv,
giving DQN a 253 × 3 = 759 action grid (routing × drain) in batch mode.

This is a generic discretization, not CFWS's flattened-index VM-migration
scheme — CFWS operates at per-PM granularity over `R × n × m` migration
decisions (see thesis_overview.md §8.5 for the distinction).
"""

from __future__ import annotations

from itertools import product

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.multi_dc_env import MultiDCEnv


def _build_routing_actions(n_dc: int, granularity: int = 5) -> np.ndarray:
    """Build a set of discrete routing allocations.

    Creates allocations where each DC gets one of `granularity` levels
    of demand (0%, 25%, 50%, 75%, 100%), then normalizes.  Also includes
    single-DC-only allocations and uniform allocation.

    Returns array of shape (num_actions, n_dc) with softmax-input logits.
    """
    actions = []

    # Uniform allocation
    actions.append(np.zeros(n_dc, dtype=np.float32))

    # Single-DC allocations (route everything to one DC)
    for i in range(n_dc):
        a = np.full(n_dc, -2.0, dtype=np.float32)
        a[i] = 2.0
        actions.append(a)

    # Grid allocations: each DC gets a level from {0, 0.25, 0.5, 0.75, 1.0}
    levels = np.linspace(0.0, 1.0, granularity)
    for combo in product(levels, repeat=n_dc):
        combo = np.array(combo, dtype=np.float32)
        if combo.sum() < 1e-6:
            continue  # skip all-zero
        # Convert proportions to logits
        combo = combo / combo.sum()
        combo = np.clip(combo, 1e-6, None)
        logits = np.log(combo)
        logits = logits - logits.mean()
        logits = np.clip(logits, -2.0, 2.0).astype(np.float32)
        actions.append(logits)

    # Deduplicate (round to avoid float noise)
    unique = {}
    for a in actions:
        key = tuple(np.round(a, 3))
        if key not in unique:
            unique[key] = a
    actions = list(unique.values())

    return np.array(actions, dtype=np.float32)


def _build_drain_actions(n_dc: int) -> np.ndarray:
    """Build discrete drain rate options.

    Three strategies: drain nothing, drain half, drain all.
    """
    return np.array(
        [
            np.full(n_dc, -3.0, dtype=np.float32),  # ~5% drain
            np.zeros(n_dc, dtype=np.float32),  # ~50% drain
            np.full(n_dc, 3.0, dtype=np.float32),  # ~95% drain
        ],
        dtype=np.float32,
    )


class DiscretizedMultiDCEnv(gym.Wrapper):
    """Wraps MultiDCEnv with a Discrete action space for DQN.

    In legacy mode: actions index into routing allocations.
    In batch mode: actions index into (routing, drain) combinations.
    """

    def __init__(self, env: MultiDCEnv, granularity: int = 5):
        super().__init__(env)
        self.routing_actions = _build_routing_actions(env.n_dc, granularity)

        if env.batch_enabled:
            self.drain_actions = _build_drain_actions(env.n_dc)
            # Combined action space: routing × drain
            n_routing = len(self.routing_actions)
            n_drain = len(self.drain_actions)
            self.n_actions = n_routing * n_drain
            self.action_space = spaces.Discrete(self.n_actions)
        else:
            self.drain_actions = None
            self.n_actions = len(self.routing_actions)
            self.action_space = spaces.Discrete(self.n_actions)

    def _decode_action(self, discrete_action: int) -> np.ndarray:
        """Convert discrete action index to continuous action array."""
        if self.env.batch_enabled:
            n_routing = len(self.routing_actions)
            routing_idx = discrete_action % n_routing
            drain_idx = discrete_action // n_routing
            drain_idx = min(drain_idx, len(self.drain_actions) - 1)
            routing = self.routing_actions[routing_idx]
            drain = self.drain_actions[drain_idx]
            return np.concatenate([routing, drain])
        else:
            idx = min(discrete_action, len(self.routing_actions) - 1)
            return self.routing_actions[idx]

    def step(self, action):
        continuous_action = self._decode_action(int(action))
        return self.env.step(continuous_action)
