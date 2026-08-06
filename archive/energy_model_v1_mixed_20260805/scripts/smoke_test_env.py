"""Smoke test the refactored environment + run baselines for alpha calibration.

For each scenario + mode, runs a quick baseline (Round Robin) and prints:
  - total cost components (energy_cost, peak_penalty)
  - peak penalty at α=1 (raw) so we can pick α to hit a target ratio
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


def run(scenario: Path, batch_enabled: bool, alpha: float) -> dict:
    sites, power_model, batch_config = load_scenario(
        scenario, batch_enabled=batch_enabled
    )
    env = MultiDCEnv(
        sites=sites,
        power_model=power_model,
        batch_enabled=batch_enabled,
        peak_penalty_weight=alpha,
        flexibility_factor=batch_config.get("flexibility_factor", 1.0),
        deadline_penalty_weight=batch_config.get("deadline_penalty_weight", 2.0),
        urgency_horizon_steps=batch_config.get("urgency_horizon_steps", 12),
    )
    pol = RoundRobinPolicy()
    obs, _ = env.reset(seed=42)
    total_energy = 0.0
    total_peak = 0.0
    total_cost = 0.0
    total_grid = 0.0
    nd_weighted = 0.0
    steps = 0
    while True:
        action = pol.predict(obs, env)
        obs, reward, term, trunc, info = env.step(action)
        total_energy += info.get("total_energy_cost", 0.0)
        total_peak += info.get("total_peak_penalty", 0.0)
        total_cost += info["total_cost"]
        total_grid += info.get("total_grid_mw", 0.0)
        for dc in info["per_dc"]:
            nd_weighted += dc["grid_mw"] * dc["net_demand"]
        steps += 1
        if term or trunc:
            break
    return {
        "obs_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.shape[0],
        "steps": steps,
        "total_energy_cost": total_energy,
        "total_peak_penalty_at_alpha": total_peak,
        "total_cost": total_cost,
        "total_grid_mw_steps": total_grid,
        "nd_weighted_load": nd_weighted,
    }


def main() -> None:
    print("=" * 70)
    print("Smoke test + alpha calibration probe")
    print("=" * 70)
    for scenario_path, label in [
        (ROOT / "env" / "scenarios" / "us_model.yaml", "us_model"),
        (ROOT / "env" / "scenarios" / "global_model.yaml", "global_model"),
    ]:
        for batch_enabled in (False, True):
            print(f"\n--- {label}  batch={batch_enabled} ---")
            res = run(scenario_path, batch_enabled, alpha=1.0)
            for k, v in res.items():
                if isinstance(v, float):
                    print(f"  {k:<35} {v:>16,.2f}")
                else:
                    print(f"  {k:<35} {v:>16}")

            # Target: peak penalty == 0.20 * energy_cost
            target_ratio = 0.20
            energy = res["total_energy_cost"]
            raw_peak = res["total_peak_penalty_at_alpha"]
            if raw_peak > 0:
                alpha_suggest = (target_ratio * energy) / raw_peak
                print(
                    f"  alpha to hit {target_ratio*100:.0f}% peak/energy ratio: "
                    f"{alpha_suggest:.6f}"
                )


if __name__ == "__main__":
    main()
