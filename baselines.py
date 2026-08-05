"""Baseline policies for comparison against the trained RL agent.

Under the demand-smoothing formulation (no on-site solar; reward includes
a peak-contribution penalty against grid net demand), the heuristics are
renamed from the old solar-supply framing:

  Follow the Sun     -> Avoid the Ramp           (route to slack grid)
  Defer to Sun       -> Defer to Low Net Demand  (drain at trough)
  GreenSlot          -> Trough-Slot Lookahead    (foresighted defer)
"""

from __future__ import annotations

import numpy as np

from env.multi_dc_env import MultiDCEnv


def _with_drain(
    routing: np.ndarray,
    drain: np.ndarray,
    env: MultiDCEnv,
    batch_routing: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the env action vector for a baseline.

    Spatial-only mode -> ``routing`` only. Spatial+temporal mode ->
    ``[routing, drain]``. With batch spatial routing enabled, the action is
    ``[routing, drain, batch_routing]``;
    ``batch_routing`` defaults to the service ``routing`` (route deferred work
    the same way as service — e.g. toward the slack/cheap grid).
    """
    if not getattr(env, "batch_enabled", False):
        return routing
    parts = [routing, drain.astype(np.float32)]
    if getattr(env, "batch_spatial_routing", False):
        br = routing if batch_routing is None else batch_routing
        parts.append(np.asarray(br, dtype=np.float32))
    return np.concatenate(parts)


# ----------------------------------------------------------------------
# Simple baselines
# ----------------------------------------------------------------------


class RoundRobinPolicy:
    name = "Round Robin"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        routing = np.zeros(env.n_dc, dtype=np.float32)
        drain = np.zeros(env.n_dc, dtype=np.float32)  # sigmoid(0) = 0.5
        return _with_drain(routing, drain, env)


class CheapestFirstPolicy:
    name = "Cheapest Price First"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        prices = np.array(
            [site.get_price(t) for site in env.sites], dtype=np.float32
        )
        routing = np.full(env.n_dc, -1.0, dtype=np.float32)
        routing[int(np.argmin(prices))] = 1.0

        if prices.max() > prices.min():
            norm = (prices - prices.min()) / (prices.max() - prices.min())
            drain = (1.0 - norm) * 2.0 - 1.0  # cheap → +1, expensive → -1
        else:
            drain = np.zeros(env.n_dc, dtype=np.float32)
        return _with_drain(routing, drain, env)


class AvoidTheRampPolicy:
    """Route demand inversely to current grid net demand.

    The DC whose grid is currently most slack gets the most load; the DC
    whose grid is approaching peak gets the least. Drain inversely
    proportional to net demand.

    Replaces the old "Follow the Sun" baseline under the demand-smoothing
    formulation (the old version routed to whichever DC had the most local
    solar, which made sense only when DCs had on-site solar self-consumption).
    """

    name = "Avoid the Ramp"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        nd = np.array(
            [site.get_net_demand(t) for site in env.sites], dtype=np.float32
        )

        # Spatial: route most to the lowest-net-demand DC
        routing = np.full(env.n_dc, -1.0, dtype=np.float32)
        routing[int(np.argmin(nd))] = 1.0

        # Drain inversely: low net demand → drain hard, high → hold
        # Map nd [0,1] → drain logit [+3, -3]
        drain = (1.0 - nd) * 6.0 - 3.0
        return _with_drain(routing, drain, env)


class LocalOnlyPolicy:
    """Each DC handles only its own cell's workload."""

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

        drain = np.ones(env.n_dc, dtype=np.float32)  # sigmoid(1) ≈ 0.73
        return _with_drain(routing, drain, env)


class StatusQuoPolicy:
    """No-optimization reference: serve each cell's own load locally, immediately.

    The closest in-framework stand-in for "the workload run as-is" — no grid-aware
    routing (each DC serves its own demand) and no temporal deferral (batch drained
    immediately). This is the counterfactual PPO's savings are measured against: a
    CICS-style load-shaper switched OFF, leaving Borg's raw aggregate to run where
    and when it arrived. Scored by the same cost model as every other policy, so the
    relative comparison is robust to the power model's absolute error.
    """

    name = "Status Quo (local, no deferral)"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        demands = np.array(
            [site.get_local_demand(t) for site in env.sites], dtype=np.float32
        )
        total = demands.sum()
        if total > 0:
            fracs = np.clip(demands / total, 1e-6, None)
            logits = np.log(fracs)
            routing = np.clip(logits - logits.mean(), -1.0, 1.0).astype(np.float32)
        else:
            routing = np.zeros(env.n_dc, dtype=np.float32)
        drain = np.full(env.n_dc, 5.0, dtype=np.float32)  # sigmoid(5) ≈ 0.993, immediate
        return _with_drain(routing, drain, env)


class RandomPolicy:
    name = "Random"

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        if getattr(env, "batch_enabled", False):
            mult = 3 if getattr(env, "batch_spatial_routing", False) else 2
            return self.rng.uniform(-1.0, 1.0, size=mult * env.n_dc).astype(np.float32)
        return self.rng.uniform(-1.0, 1.0, size=env.n_dc).astype(np.float32)


# ----------------------------------------------------------------------
# Temporal baselines
# ----------------------------------------------------------------------


class DrainImmediatelyPolicy:
    """Route evenly, drain all batch work immediately.

    Isolates the value of temporal scheduling.
    """

    name = "Drain Immediately"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        routing = np.zeros(env.n_dc, dtype=np.float32)
        drain = np.full(env.n_dc, 5.0, dtype=np.float32)  # sigmoid(5) ≈ 0.993
        return _with_drain(routing, drain, env)


class DeferToLowNetDemandPolicy:
    """Equal routing; drain inversely proportional to current net demand.

    When the grid is slack (midday solar trough), drain aggressively;
    when net demand is high (evening ramp), hold the pool. Replaces the
    old "Defer to Sun" baseline.
    """

    name = "Defer to Low Net Demand"

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        nd = np.array(
            [site.get_net_demand(t) for site in env.sites], dtype=np.float32
        )
        routing = np.zeros(env.n_dc, dtype=np.float32)
        # nd ∈ [0,1] → drain logit ∈ [+3, -3]
        drain = (1.0 - nd) * 6.0 - 3.0
        return _with_drain(routing, drain, env)


class TroughSlotLookaheadPolicy:
    """Net-demand-trough lookahead scheduling (GreenSlot-style adaptation).

    Looks ahead N timesteps to compute average grid net demand per DC.
    Routes proportional to *inverse* net demand. For temporal scheduling,
    drains when current net demand is below the lookahead average (this
    is a "trough slot"); defers when above.

    Replaces the old GreenSlot baseline, which operated on solar surplus
    rather than grid demand troughs.
    """

    name = "Trough-Slot Lookahead"

    def __init__(self, lookahead: int = 36):
        """lookahead: future timesteps to consider (default 36 = 3 hours)."""
        self.lookahead = lookahead

    def predict(self, obs: np.ndarray, env: MultiDCEnv) -> np.ndarray:
        t = env.step_index
        N = env.n_dc

        avg_nd = np.zeros(N, dtype=np.float32)
        current_nd = np.zeros(N, dtype=np.float32)
        for i, site in enumerate(env.sites):
            current_nd[i] = site.get_net_demand(t)
            total = current_nd[i]
            count = 1
            for dt in range(1, self.lookahead + 1):
                future_t = t + dt
                if future_t < site.num_timesteps:
                    total += site.get_net_demand(future_t)
                    count += 1
            avg_nd[i] = total / count

        # Spatial: route most to the DC with the lowest lookahead net demand
        # (most slack on average over the next 3 hours)
        inv = 1.0 - avg_nd  # higher = more slack
        if inv.sum() > 1e-6:
            weights = inv / inv.sum()
            weights = np.clip(weights, 1e-6, None)
            routing = np.log(weights).astype(np.float32)
            routing = routing - routing.mean()
            routing = np.clip(routing, -1.0, 1.0)
        else:
            routing = np.zeros(N, dtype=np.float32)

        # Temporal: drain when current nd < lookahead average (trough slot)
        drain = np.zeros(N, dtype=np.float32)
        for i in range(N):
            if avg_nd[i] > 1e-6:
                # Ratio < 1 means current is more slack than average → drain now
                # Map: ratio=0.5 → +3, ratio=2 → -3
                ratio = current_nd[i] / avg_nd[i]
                drain[i] = np.clip((1.0 - ratio) * 3.0, -3.0, 3.0)
            else:
                drain[i] = 3.0  # grid uniformly slack → drain

        return _with_drain(routing, drain, env)


ALL_BASELINES = [
    StatusQuoPolicy,
    RoundRobinPolicy,
    CheapestFirstPolicy,
    AvoidTheRampPolicy,
    LocalOnlyPolicy,
    RandomPolicy,
    DrainImmediatelyPolicy,
    DeferToLowNetDemandPolicy,
    TroughSlotLookaheadPolicy,
]
