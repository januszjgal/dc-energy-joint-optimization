"""CFWS-style flattened-index discrete action wrapper for MultiDCEnv.

Adapts the action-encoding idea from Zhao et al. 2025 (CFWS, IEEE TSC 10(1))
to our cell-aggregate multi-DC routing formulation. CFWS's original
encoding is `R × n × m → 1D` for (VM, dest_DC, dest_PM) VM-migration
decisions; our env has no VMs/PMs, so we adapt the *spirit* of the trick —
a small, semantically-meaningful action set decoded from a single discrete
index via hash-map (division/modulo) — to our routing+drain formulation.

Action set: 48 actions (= n_dc² × 3 drain levels for n_dc=4).
Decode:
    action_id ∈ [0, 47]
    src_dc      = action_id // (n_dc * n_drain_levels)        ∈ [0, n_dc)
    rest        = action_id  % (n_dc * n_drain_levels)
    dst_dc      = rest // n_drain_levels                       ∈ [0, n_dc)
    drain_level = rest  % n_drain_levels                       ∈ [0, 3)

Semantics per action:
    if src_dc == dst_dc:    # "no migration"
        routing fractions = uniform (1/n_dc each)
    else:                   # "migrate fraction TRANSFER_AMOUNT from src to dst"
        fractions[src] -= TRANSFER_AMOUNT
        fractions[dst] += TRANSFER_AMOUNT
        others = uniform
    drain (all DCs) = sigmoid(DRAIN_LOGITS[drain_level])
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.multi_dc_env import MultiDCEnv

# Fraction of load moved from src to dst in a migration action.
TRANSFER_AMOUNT = 0.15

# Drain logits per intensity level (passed through sigmoid in the env).
#   sigmoid(-3) ≈ 0.05 (hold)
#   sigmoid( 0)  = 0.50 (half)
#   sigmoid(+3) ≈ 0.95 (flush)
DRAIN_LOGITS = (-3.0, 0.0, 3.0)


def _fractions_to_logits(fractions: np.ndarray) -> np.ndarray:
    """Convert allocation fractions to softmax-compatible logits.

    The env applies softmax to routing_logits, so log(fractions) recovers
    the fractions after softmax (up to a constant shift).
    """
    safe = np.clip(fractions, 1e-6, None)
    logits = np.log(safe)
    return (logits - logits.mean()).astype(np.float32)


class CFWSStyleDiscretizedEnv(gym.Wrapper):
    """Wraps MultiDCEnv with a CFWS-style flattened-index action space.

    The underlying MultiDCEnv exposes a continuous Box action space (routing
    logits + optional drain logits); this wrapper converts a single Discrete
    action_id into the appropriate continuous vector before calling step().
    """

    def __init__(self, env: MultiDCEnv):
        super().__init__(env)
        self.n_dc = env.n_dc
        self.n_drain_levels = len(DRAIN_LOGITS)
        self.n_actions = self.n_dc * self.n_dc * self.n_drain_levels
        self.action_space = spaces.Discrete(self.n_actions)

    def _decode_action(self, action_id: int) -> tuple[int, int, int]:
        per_src = self.n_dc * self.n_drain_levels
        src_dc = action_id // per_src
        rest = action_id % per_src
        dst_dc = rest // self.n_drain_levels
        drain_level = rest % self.n_drain_levels
        return src_dc, dst_dc, drain_level

    def _build_continuous_action(self, action_id: int) -> np.ndarray:
        src, dst, drain_lvl = self._decode_action(int(action_id))

        # Start from uniform allocation
        fractions = np.full(self.n_dc, 1.0 / self.n_dc, dtype=np.float64)
        if src != dst:
            fractions[src] -= TRANSFER_AMOUNT
            fractions[dst] += TRANSFER_AMOUNT
            fractions = np.clip(fractions, 1e-6, None)  # numerical safety

        routing_logits = _fractions_to_logits(fractions)
        drain_logit = DRAIN_LOGITS[drain_lvl]

        if self.env.batch_enabled:
            drain_logits = np.full(self.n_dc, drain_logit, dtype=np.float32)
            parts = [routing_logits, drain_logits]
            if getattr(self.env, "batch_spatial_routing", False):
                # Route batch like service (the migration action applies to both).
                parts.append(routing_logits)
            return np.concatenate(parts)
        return routing_logits

    def step(self, action):
        return self.env.step(self._build_continuous_action(action))


__all__ = ["CFWSStyleDiscretizedEnv", "TRANSFER_AMOUNT", "DRAIN_LOGITS"]
