"""Run baselines-only evaluation under the new env (no trained RL model needed).

Quick way to preview the new policy rankings while RL training is in flight.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

from baselines import ALL_BASELINES
from env.data_loader import load_scenario
from env.multi_dc_env import MultiDCEnv


def run_episode(env: MultiDCEnv, policy) -> dict:
    obs, _ = env.reset(seed=42)
    total_cost = total_energy = total_peak = nd_weighted = total_grid = 0.0
    total_expired = 0.0
    while True:
        action = policy.predict(obs, env)
        obs, reward, term, trunc, info = env.step(action)
        total_cost += info["total_cost"]
        total_energy += info.get("total_energy_cost", 0.0)
        total_peak += info.get("total_peak_penalty", 0.0)
        total_grid += info.get("total_grid_mw", 0.0)
        total_expired += info.get("total_batch_expired", 0.0)
        for dc in info["per_dc"]:
            nd_weighted += dc["grid_mw"] * dc["net_demand"]
        if term or trunc:
            break
    return {
        "total_cost": total_cost,
        "total_energy_cost": total_energy,
        "total_peak_penalty": total_peak,
        "total_grid_mw": total_grid,
        "nd_weighted": nd_weighted,
        "total_expired": total_expired,
    }


def evaluate(scenario: Path, batch_enabled: bool, alpha: float = 0.015) -> None:
    print(f"\n{'='*82}")
    print(f"Scenario: {scenario.name}    batch={batch_enabled}    alpha={alpha}")
    print("=" * 82)

    sites, power_model, batch_config = load_scenario(
        scenario, batch_enabled=batch_enabled
    )

    rows = []
    for cls in ALL_BASELINES:
        env = MultiDCEnv(
            sites=sites,
            power_model=power_model,
            batch_enabled=batch_enabled,
            peak_penalty_weight=alpha,
            flexibility_factor=batch_config.get("flexibility_factor", 1.0),
            deadline_penalty_weight=batch_config.get("deadline_penalty_weight", 2.0),
            urgency_horizon_steps=batch_config.get("urgency_horizon_steps", 12),
        )
        # Re-create sites so backlog/pool state is fresh per policy
        sites_fresh, _, _ = load_scenario(scenario, batch_enabled=batch_enabled)
        env.sites = sites_fresh

        pol = cls()
        result = run_episode(env, pol)
        rows.append((pol.name, result))
        print(
            f"  {pol.name:<28}  cost={result['total_cost']:>13,.0f}  "
            f"energy={result['total_energy_cost']:>12,.0f}  "
            f"peak={result['total_peak_penalty']:>10,.0f}  "
            f"nd_weighted={result['nd_weighted']:>12,.0f}  "
            f"expired={result['total_expired']:>8.1f}"
        )

    rows.sort(key=lambda r: r[1]["total_cost"])
    print(f"\n  --- Ranked by total cost ---")
    for i, (name, r) in enumerate(rows):
        print(f"  {i+1:>2}. {name:<28}  total_cost = ${r['total_cost']:,.0f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.015)
    args = parser.parse_args()

    for sp in [
        ROOT / "env" / "scenarios" / "us_model.yaml",
        ROOT / "env" / "scenarios" / "global_model.yaml",
    ]:
        for batch in (False, True):
            evaluate(sp, batch, args.alpha)


if __name__ == "__main__":
    main()
