"""Focused follow-up: does PPO-burst+mem differentiate drain rates burst vs non-burst?

Step 1 finding: baseline PPO learned burst-aware spatial routing implicitly,
but did NOT differentiate temporal behavior (drain rate same in burst vs
non-burst windows). Step 2 added a burst_severity feature; this script
checks whether the new PPO uses that signal on the temporal axis.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import PPO

from env.data_loader import load_scenario
from env.multi_dc_env import MultiDCEnv


SEED = 42
ALPHA = 0.015
BURST_PCTL = 95


def run_ppo(scenario_path: Path, model_path: Path, burst_aware: bool, memory: bool):
    sites, power_model, batch_config = load_scenario(
        scenario_path, batch_enabled=True, seed=SEED
    )
    env = MultiDCEnv(
        sites=sites, power_model=power_model, batch_enabled=True,
        peak_penalty_weight=ALPHA,
        flexibility_factor=batch_config.get("flexibility_factor", 1.0),
        deadline_penalty_weight=batch_config.get("deadline_penalty_weight", 2.0),
        urgency_horizon_steps=batch_config.get("urgency_horizon_steps", 12),
        burst_aware=burst_aware, memory_enabled=memory,
    )
    model = PPO.load(model_path)
    obs, _ = env.reset(seed=SEED)
    recs = []
    t = 0
    while True:
        arrival = sum(env.sites[i].get_batch_demand(t) for i in range(env.n_dc))
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        recs.append({
            "arrival": float(arrival),
            "drain_rates": [float(d) for d in info["drain_rates"]],
            "fractions": [float(f) for f in info["fractions"]],
            "net_demands": [float(d["net_demand"]) for d in info["per_dc"]],
        })
        t += 1
        if term or trunc:
            break
    return recs


def summarize(label, recs):
    arrivals = np.array([r["arrival"] for r in recs])
    thresh = np.percentile(arrivals, BURST_PCTL)
    burst_mask = arrivals > thresh

    drain_per_step = np.array([np.mean(r["drain_rates"]) for r in recs])
    hhi_per_step = np.array([sum(f * f for f in r["fractions"]) for r in recs])
    nd_routed_per_step = np.array([sum(f * n for f, n in zip(r["fractions"], r["net_demands"])) for r in recs])

    print(f"\n--- {label} ---")
    print(f"  drain rate (burst):     {drain_per_step[burst_mask].mean():.4f}")
    print(f"  drain rate (non-burst): {drain_per_step[~burst_mask].mean():.4f}")
    print(f"  drain delta (burst - non-burst): {drain_per_step[burst_mask].mean() - drain_per_step[~burst_mask].mean():+.4f}")
    print(f"  HHI (burst):     {hhi_per_step[burst_mask].mean():.4f}")
    print(f"  HHI (non-burst): {hhi_per_step[~burst_mask].mean():.4f}")
    print(f"  ND-weighted route (burst):     {nd_routed_per_step[burst_mask].mean():.4f}")
    print(f"  ND-weighted route (non-burst): {nd_routed_per_step[~burst_mask].mean():.4f}")


def main():
    configs = [
        ("US batch",
         ROOT / "env/scenarios/us_model.yaml",
         ROOT / "models/ppo_us_model_batch.zip",
         ROOT / "models/ppo_us_model_batch_burst.zip"),
        ("Global batch",
         ROOT / "env/scenarios/global_model.yaml",
         ROOT / "models/ppo_global_model_batch.zip",
         ROOT / "models/ppo_global_model_batch_burst.zip"),
    ]
    for name, scenario, baseline, burst in configs:
        print("=" * 70)
        print(name)
        print("=" * 70)
        b_recs = run_ppo(scenario, baseline, burst_aware=False, memory=False)
        summarize("PPO baseline", b_recs)
        u_recs = run_ppo(scenario, burst, burst_aware=True, memory=True)
        summarize("PPO-burst+mem", u_recs)


if __name__ == "__main__":
    main()
