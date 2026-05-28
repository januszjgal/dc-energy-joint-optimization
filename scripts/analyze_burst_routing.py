"""Diagnostic analysis: do batch-arrival bursts drive disproportionate cost,
and does PPO route bursts differently from non-burst timesteps?

Burst definition: top 5% of timesteps by *aggregate* batch CPU arrival
across all DCs. Heavy-tail in the arrival distributions (Tirmazi §7)
means a small fraction of timesteps deliver a large fraction of total
batch work; the question is whether (a) those timesteps drive a
disproportionate share of total cost, and (b) policy decisions during
those timesteps are meaningfully different from off-burst routing.

For each batch-mode scenario (US batch, Global batch), this runs PPO,
DQN-routing-grid, DQN-flatidx, and Round Robin to full episode and
computes:
  - Burst share of total cost per policy
  - Per-policy cost per step in burst vs non-burst windows
  - Advantage vs Round Robin in each regime
  - PPO spatial concentration (HHI of routing fractions) during burst vs non-burst
  - PPO net-demand-weighted routing during burst vs non-burst
  - PPO drain-rate behavior during burst vs non-burst
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import DQN, PPO

from baselines import RoundRobinPolicy
from env.cfws_style_wrapper import CFWSStyleDiscretizedEnv
from env.data_loader import load_scenario
from env.discrete_wrapper import DiscretizedMultiDCEnv
from env.multi_dc_env import MultiDCEnv


BURST_PERCENTILE = 95  # top 5% of timesteps by aggregate batch arrival
ALPHA = 0.015
SEED = 42


CONFIGS = [
    {
        "name": "US batch",
        "scenario": "env/scenarios/us_model.yaml",
        "ppo": "models/ppo_us_model_batch.zip",
        "dqn": "models/dqn_us_model_batch.zip",
        "dqn_flat": "models/dqn_us_model_batch_flatidx.zip",
        # Step 2 burst-aware + memory variants
        "ppo_burst": "models/ppo_us_model_batch_burst.zip",
        "dqn_flat_burst": "models/dqn_us_model_batch_flatidx_burst.zip",
    },
    {
        "name": "Global batch",
        "scenario": "env/scenarios/global_model.yaml",
        "ppo": "models/ppo_global_model_batch.zip",
        "dqn": "models/dqn_global_model_batch.zip",
        "dqn_flat": "models/dqn_global_model_batch_flatidx.zip",
        "ppo_burst": "models/ppo_global_model_batch_burst.zip",
        "dqn_flat_burst": "models/dqn_global_model_batch_flatidx_burst.zip",
    },
]


def make_env(
    scenario_path: Path,
    action_scheme: str | None = None,
    burst_aware: bool = False,
    memory_enabled: bool = False,
):
    sites, power_model, batch_config = load_scenario(
        scenario_path, batch_enabled=True, seed=SEED
    )
    base = MultiDCEnv(
        sites=sites,
        power_model=power_model,
        batch_enabled=True,
        peak_penalty_weight=ALPHA,
        flexibility_factor=batch_config.get("flexibility_factor", 1.0),
        deadline_penalty_weight=batch_config.get("deadline_penalty_weight", 2.0),
        urgency_horizon_steps=batch_config.get("urgency_horizon_steps", 12),
        burst_aware=burst_aware,
        memory_enabled=memory_enabled,
    )
    if action_scheme == "cfws":
        return CFWSStyleDiscretizedEnv(base), base
    if action_scheme == "routing-grid":
        return DiscretizedMultiDCEnv(base), base
    return base, base


def run_episode(env, base_env, predict_fn, is_sb3: bool) -> list[dict]:
    """Run one full episode; capture per-step batch arrival + cost + routing."""
    obs, _ = env.reset(seed=SEED)
    records: list[dict] = []
    t = 0
    while True:
        arrivals = [base_env.sites[i].get_batch_demand(t) for i in range(base_env.n_dc)]
        total_arrival = sum(arrivals)
        if is_sb3:
            action, _ = predict_fn(obs, deterministic=True)
        else:
            action = predict_fn(obs, base_env)
        obs, reward, term, trunc, info = env.step(action)
        records.append({
            "t": t,
            "arrival_total": float(total_arrival),
            "total_cost": float(info["total_cost"]),
            "energy_cost": float(info.get("total_energy_cost", 0.0)),
            "peak_penalty": float(info.get("total_peak_penalty", 0.0)),
            "fractions": [float(f) for f in info["fractions"]],
            "drain_rates": [float(d) for d in info.get("drain_rates", [])],
            "net_demands": [float(d["net_demand"]) for d in info["per_dc"]],
            "grid_mws": [float(d["grid_mw"]) for d in info["per_dc"]],
        })
        t += 1
        if term or trunc:
            break
    return records


def fmt_pct(x: float) -> str:
    return f"{100*x:+.2f}%"


def analyze_one(cfg: dict) -> dict:
    print(f"\n{'='*82}\n{cfg['name']}\n{'='*82}")

    scenario = ROOT / cfg["scenario"]
    policies: list[tuple[str, list[dict]]] = []

    # Round Robin (reference baseline)
    env, base = make_env(scenario)
    print("  running Round Robin...")
    policies.append(("Round Robin", run_episode(env, base, RoundRobinPolicy().predict, is_sb3=False)))

    # PPO (baseline: no burst, no memory)
    env, base = make_env(scenario)
    print("  running PPO (baseline)...")
    policies.append(("PPO", run_episode(env, base, PPO.load(ROOT / cfg["ppo"]).predict, is_sb3=True)))

    # PPO burst-aware + memory (Step 2)
    burst_ppo = ROOT / cfg["ppo_burst"]
    if burst_ppo.exists():
        env, base = make_env(scenario, burst_aware=True, memory_enabled=True)
        print(f"  running PPO-burst+mem...")
        policies.append(("PPO-burst+mem", run_episode(env, base, PPO.load(burst_ppo).predict, is_sb3=True)))
    else:
        print(f"  skipping PPO-burst+mem ({burst_ppo.name} not found)")

    # DQN routing-grid (baseline)
    env, base = make_env(scenario, action_scheme="routing-grid")
    print("  running DQN-routing-grid...")
    policies.append(("DQN-routing-grid", run_episode(env, base, DQN.load(ROOT / cfg["dqn"]).predict, is_sb3=True)))

    # DQN flat-idx (baseline)
    env, base = make_env(scenario, action_scheme="cfws")
    print("  running DQN-flatidx...")
    policies.append(("DQN-flatidx", run_episode(env, base, DQN.load(ROOT / cfg["dqn_flat"]).predict, is_sb3=True)))

    # DQN flat-idx burst-aware + memory (Step 2)
    burst_dqn = ROOT / cfg["dqn_flat_burst"]
    if burst_dqn.exists():
        env, base = make_env(scenario, action_scheme="cfws", burst_aware=True, memory_enabled=True)
        print(f"  running DQN-flatidx-burst+mem...")
        policies.append(("DQN-flatidx-burst+mem", run_episode(env, base, DQN.load(burst_dqn).predict, is_sb3=True)))
    else:
        print(f"  skipping DQN-flatidx-burst+mem ({burst_dqn.name} not found)")

    # Arrivals are identical across policies (same seed). Use Round Robin's as canonical.
    arrivals = np.array([r["arrival_total"] for r in policies[0][1]])
    burst_threshold = float(np.percentile(arrivals, BURST_PERCENTILE))
    burst_mask = arrivals > burst_threshold
    n_burst = int(burst_mask.sum())
    n_total = len(arrivals)

    arrival_burst_mean = float(arrivals[burst_mask].mean()) if n_burst else 0.0
    arrival_nonburst_mean = float(arrivals[~burst_mask].mean())
    arrival_ratio = arrival_burst_mean / arrival_nonburst_mean if arrival_nonburst_mean else float("inf")

    print(f"\nBurst threshold (top {100-BURST_PERCENTILE}% by arrival): {burst_threshold:.4f}")
    print(f"  burst timesteps: {n_burst}/{n_total} ({100*n_burst/n_total:.1f}%)")
    print(f"  arrival magnitude: burst-mean {arrival_burst_mean:.4f} vs non-burst-mean {arrival_nonburst_mean:.4f}  ({arrival_ratio:.1f}x)")

    # --- Per-policy cost analysis ---
    print(f"\nCost decomposition (burst = top {100-BURST_PERCENTILE}% arrival timesteps):")
    print(f"  {'Policy':<22} {'Total $':>14} {'Burst $':>14} {'Burst %':>8} {'$/burst step':>14} {'$/non-burst step':>18}")
    print("  " + "-" * 100)

    per_policy = {}
    for name, recs in policies:
        costs = np.array([r["total_cost"] for r in recs])
        total = float(costs.sum())
        burst_total = float(costs[burst_mask].sum())
        nonburst_total = float(costs[~burst_mask].sum())
        burst_share = burst_total / total if total else 0.0
        burst_per_step = burst_total / n_burst if n_burst else 0.0
        nonburst_per_step = nonburst_total / (n_total - n_burst) if (n_total - n_burst) else 0.0
        per_policy[name] = {
            "total": total, "burst_total": burst_total, "nonburst_total": nonburst_total,
            "burst_share": burst_share, "burst_per_step": burst_per_step,
            "nonburst_per_step": nonburst_per_step,
        }
        print(f"  {name:<22} {total:>13,.0f} {burst_total:>13,.0f} {100*burst_share:>7.1f}% {burst_per_step:>13,.0f} {nonburst_per_step:>17,.0f}")

    # --- Advantage analysis vs Round Robin ---
    rr = per_policy["Round Robin"]
    print(f"\nAdvantage vs Round Robin (positive = cheaper than Round Robin):")
    print(f"  {'Policy':<22} {'Total adv':>11} {'Burst adv':>11} {'Off-burst adv':>15}")
    print("  " + "-" * 70)
    adv_results = {}
    for name, _ in policies[1:]:
        s = per_policy[name]
        total_adv = (rr["total"] - s["total"]) / rr["total"] if rr["total"] else 0
        burst_adv = (rr["burst_per_step"] - s["burst_per_step"]) / rr["burst_per_step"] if rr["burst_per_step"] else 0
        nonburst_adv = (rr["nonburst_per_step"] - s["nonburst_per_step"]) / rr["nonburst_per_step"] if rr["nonburst_per_step"] else 0
        adv_results[name] = {"total": total_adv, "burst": burst_adv, "nonburst": nonburst_adv}
        print(f"  {name:<22} {fmt_pct(total_adv):>10} {fmt_pct(burst_adv):>10} {fmt_pct(nonburst_adv):>14}")

    # --- PPO behavior decomposition ---
    ppo_recs = next(r for n, r in policies if n == "PPO")
    fracs = np.array([r["fractions"] for r in ppo_recs])
    nds = np.array([r["net_demands"] for r in ppo_recs])
    drains = np.array([r["drain_rates"] for r in ppo_recs])

    # HHI: sum of fraction² across DCs. uniform 1/4 -> HHI=0.25; fully concentrated -> HHI=1.0
    hhi = (fracs ** 2).sum(axis=1)
    nd_weighted_route = (fracs * nds).sum(axis=1)
    mean_drain = drains.mean(axis=1)

    print(f"\nPPO behavior — burst vs non-burst:")
    print(f"  Spatial concentration (HHI of routing fractions; 0.25 uniform, 1.0 fully concentrated):")
    print(f"    burst:     {hhi[burst_mask].mean():.4f}")
    print(f"    non-burst: {hhi[~burst_mask].mean():.4f}")
    print(f"  Net-demand-weighted routing (Σ fraction_i × net_demand_i; lower = routing toward slack grids):")
    print(f"    burst:     {nd_weighted_route[burst_mask].mean():.4f}")
    print(f"    non-burst: {nd_weighted_route[~burst_mask].mean():.4f}")
    print(f"  Mean drain rate across DCs (1=flush, 0=hold):")
    print(f"    burst:     {mean_drain[burst_mask].mean():.4f}")
    print(f"    non-burst: {mean_drain[~burst_mask].mean():.4f}")

    return {
        "scenario": cfg["name"],
        "burst_percentile": BURST_PERCENTILE,
        "n_burst": n_burst, "n_total": n_total,
        "arrival_ratio": arrival_ratio,
        "per_policy": per_policy,
        "advantage_vs_round_robin": adv_results,
        "ppo_behavior": {
            "hhi_burst": float(hhi[burst_mask].mean()),
            "hhi_nonburst": float(hhi[~burst_mask].mean()),
            "nd_weighted_burst": float(nd_weighted_route[burst_mask].mean()),
            "nd_weighted_nonburst": float(nd_weighted_route[~burst_mask].mean()),
            "drain_burst": float(mean_drain[burst_mask].mean()),
            "drain_nonburst": float(mean_drain[~burst_mask].mean()),
        },
    }


def main() -> None:
    results = {}
    for cfg in CONFIGS:
        results[cfg["name"]] = analyze_one(cfg)

    out = ROOT / "output" / "burst_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved analysis to {out}")


if __name__ == "__main__":
    main()
