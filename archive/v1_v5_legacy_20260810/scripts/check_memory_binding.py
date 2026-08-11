"""Quick check: how often does memory bind in our env?

Runs Round Robin on US batch with memory_enabled=True and reports:
  - How many timesteps had served clipped by memory constraint
  - Per-DC binding frequency
  - Difference in total cost vs memory_enabled=False
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


def run(memory_enabled: bool) -> dict:
    sites, power_model, batch_config = load_scenario(
        ROOT / "env" / "scenarios" / "us_model_v2_2025.yaml",
        batch_enabled=True,
    )
    env = MultiDCEnv(
        sites=sites,
        power_model=power_model,
        batch_enabled=True,
        memory_enabled=memory_enabled,
        peak_penalty_weight=0.015,
        flexibility_factor=batch_config.get("flexibility_factor", 1.0),
        deadline_penalty_weight=batch_config.get("deadline_penalty_weight", 2.0),
        urgency_horizon_steps=batch_config.get("urgency_horizon_steps", 12),
    )
    pol = RoundRobinPolicy()
    obs, _ = env.reset(seed=42)
    total_cost = 0.0
    memory_binds_per_step = 0
    per_dc_mem_load = [[] for _ in range(env.n_dc)]
    while True:
        action = pol.predict(obs, env)
        obs, reward, term, trunc, info = env.step(action)
        total_cost += info["total_cost"]
        for i, dc in enumerate(env.sites):
            per_dc_mem_load[i].append(dc.current_memory_load)
            # Check if memory was the binding constraint (would have clipped serve)
            mem_required = dc.current_load * dc.memory_cpu_ratio
            if mem_required > dc.memory_capacity * 0.99:
                memory_binds_per_step += 1 / env.n_dc  # fractional contribution
        if term or trunc:
            break
    return {
        "memory_enabled": memory_enabled,
        "total_cost": total_cost,
        "memory_binds_per_step": memory_binds_per_step,
        "per_dc_mem_capacity": [s.memory_capacity for s in env.sites],
        "per_dc_avg_mem_load": [np.mean(loads) for loads in per_dc_mem_load],
        "per_dc_max_mem_load": [np.max(loads) for loads in per_dc_mem_load],
    }


def main() -> None:
    print("=" * 70)
    print("Memory-binding diagnostic: US batch, Round Robin policy")
    print("=" * 70)

    off = run(memory_enabled=False)
    on = run(memory_enabled=True)

    print(f"\nmemory_enabled=False:  total_cost = ${off['total_cost']:>13,.0f}")
    print(f"memory_enabled=True:   total_cost = ${on['total_cost']:>13,.0f}")
    diff_pct = (on['total_cost'] - off['total_cost']) / off['total_cost'] * 100
    print(f"Diff:                  {diff_pct:+.4f}%")
    print()
    print(f"Memory binding events (in memory_enabled=True run): {on['memory_binds_per_step']:.1f}")
    print(f"Memory capacities per DC: {[f'{x:.3f}' for x in on['per_dc_mem_capacity']]}")
    print(f"Avg memory load per DC:   {[f'{x:.3f}' for x in on['per_dc_avg_mem_load']]}")
    print(f"Max memory load per DC:   {[f'{x:.3f}' for x in on['per_dc_max_mem_load']]}")

    print()
    if abs(diff_pct) < 0.5 and on['memory_binds_per_step'] < 10:
        print("VERDICT: Memory does NOT meaningfully bind. Flipping it has near-zero")
        print("         effect on baseline behavior. Confound concern is fictional;")
        print("         safe to enable for burst-aware retraining.")
    elif abs(diff_pct) < 5.0:
        print("VERDICT: Memory binds occasionally but cost impact is small.")
        print("         Confound is mild — proceed but caveat results.")
    else:
        print("VERDICT: Memory binds frequently and meaningfully changes cost.")
        print("         Real confound — recommend rerunning all baselines or")
        print("         doing an orthogonal sweep.")


if __name__ == "__main__":
    main()
