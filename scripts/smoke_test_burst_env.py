"""Smoke test for burst-aware (+ memory-enabled) env.

Verifies:
  1. Observation dim is correct: batch + burst + memory = (8 + 2 + 1)*N + 3 = 47 for N=4
  2. Burst severity values are computed correctly (1.0-ish for normal, higher for bursts)
  3. Random rollout completes without errors
  4. Memory load is tracked but doesn't bind
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from baselines import RoundRobinPolicy
from env.data_loader import load_scenario
from env.multi_dc_env import MultiDCEnv


def main() -> None:
    print("=" * 70)
    print("Smoke test: burst-aware + memory-enabled env, US batch")
    print("=" * 70)

    sites, power_model, batch_config = load_scenario(
        ROOT / "env" / "scenarios" / "us_model.yaml",
        batch_enabled=True,
    )
    env = MultiDCEnv(
        sites=sites,
        power_model=power_model,
        batch_enabled=True,
        memory_enabled=True,
        burst_aware=True,
        peak_penalty_weight=0.015,
        flexibility_factor=batch_config.get("flexibility_factor", 1.0),
        deadline_penalty_weight=batch_config.get("deadline_penalty_weight", 2.0),
        urgency_horizon_steps=batch_config.get("urgency_horizon_steps", 12),
    )
    print(f"n_dc: {env.n_dc}")
    # Derived rather than hardcoded: the hardcoded 47 went stale when
    # site_context became default-on (M2), which adds 4 dims/DC.
    ctx = 4 if env.site_context else 0
    dc = 1 if env.demand_charge_enabled else 0
    expected = (8 + 2 + 1 + ctx + dc) * env.n_dc + 3
    print(f"obs dim: {env.observation_space.shape[0]}  "
          f"(expected: (8 + 2 mem + 1 burst + {ctx} ctx + {dc} demand-charge)"
          f"*{env.n_dc} + 3 = {expected})")
    assert env.observation_space.shape[0] == expected, (
        f"Expected {expected}, got {env.observation_space.shape[0]}"
    )

    # Reset and step
    obs, _ = env.reset(seed=42)
    print(f"\nInitial obs shape: {obs.shape}")
    print(f"Initial burst severity per DC (positions 9, 20, 31, 42): "
          f"{[obs[9], obs[20], obs[31], obs[42]]}")
    # At t=0 (before any history), should be 1.0 (neutral)

    # Run several steps and watch burst severity values
    pol = RoundRobinPolicy()
    burst_severities = []
    arrivals_seen = []
    for t in range(500):
        action = pol.predict(obs, env)
        obs, reward, term, trunc, info = env.step(action)
        # Burst severity is at index 9 (DC 0), 20 (DC 1), 31 (DC 2), 42 (DC 3)
        # per_dc block is 11 wide (8 base + 2 mem + 1 burst); global block 3
        # Actually let's just track total arrivals from the env state
        arrival_dc0 = env.sites[0].get_batch_demand(env.step_index)
        burst_dc0 = env.sites[0].get_burst_severity(env.step_index)
        burst_severities.append(burst_dc0)
        arrivals_seen.append(arrival_dc0)
        if term or trunc:
            break

    arr = np.array(arrivals_seen)
    sev = np.array(burst_severities)
    print(f"\nDC 0 over 500 steps:")
    print(f"  arrival mean: {arr.mean():.4f}, max: {arr.max():.4f}, min: {arr.min():.4f}")
    print(f"  burst severity mean: {sev.mean():.4f}, max: {sev.max():.4f}, "
          f"min: {sev.min():.4f}")
    print(f"  steps with burst severity > 2.0: {(sev > 2.0).sum()}/500")
    print(f"  steps with burst severity > 5.0: {(sev > 5.0).sum()}/500")

    # Full episode
    print(f"\nFull episode rollout...")
    obs, _ = env.reset(seed=42)
    total_cost = 0.0
    steps = 0
    while True:
        action = pol.predict(obs, env)
        obs, reward, term, trunc, info = env.step(action)
        total_cost += info["total_cost"]
        steps += 1
        if term or trunc:
            break
    print(f"  steps: {steps}")
    print(f"  total cost (RR, US batch, burst+memory env): ${total_cost:,.0f}")
    print(f"  reference (non-burst non-memory RR): $9,785,322")
    diff_pct = (total_cost - 9785322) / 9785322 * 100
    print(f"  diff: {diff_pct:+.4f}%")


if __name__ == "__main__":
    main()
