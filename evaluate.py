"""Evaluate a trained PPO agent against all baselines.

Runs one full episode for the PPO agent and each baseline policy,
collects per-timestep metrics, and produces a summary report with plots.

Usage:
    python evaluate.py --scenario env/scenarios/us_model.yaml --model models/ppo_us_model
    python evaluate.py --scenario env/scenarios/us_model.yaml --model models/ppo_us_model_batch --batch-mode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO

from baselines import ALL_BASELINES
from env.data_loader import load_scenario
from env.multi_dc_env import REFERENCE_DEMAND_CHARGE_RATE, MultiDCEnv


def run_episode(
    env: MultiDCEnv,
    predict_fn,
    is_sb3: bool = False,
) -> tuple[float, list[dict[str, Any]]]:
    """Run one full episode and collect per-step info.

    predict_fn: either a baseline's predict(obs, env) or SB3 model.predict(obs).
    is_sb3: if True, calls predict_fn(obs) instead of predict_fn(obs, env).
    """
    obs, _ = env.reset()
    total_reward = 0.0
    history: list[dict[str, Any]] = []

    while True:
        if is_sb3:
            action, _ = predict_fn(obs, deterministic=True)
        else:
            action = predict_fn(obs, env)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        history.append(info)

        if terminated or truncated:
            break

    return total_reward, history


def compute_summary(history: list[dict], batch_enabled: bool = False) -> dict:
    """Compute summary metrics from episode history."""
    total_cost = sum(h["total_cost"] for h in history)
    total_energy_cost = sum(h.get("total_energy_cost", 0.0) for h in history)
    total_peak_penalty = sum(h.get("total_peak_penalty", 0.0) for h in history)
    total_grid_mw = sum(h.get("total_grid_mw", 0.0) for h in history)

    # Peak-weighted grid consumption: how much load was drawn at the
    # duck-curve neck (averaged net demand across DCs, weighted by grid_mw).
    # Quantifies a policy's contribution to grid stress without baking in
    # the α weight, making it directly comparable across calibrations.
    n_dc = len(history[0]["per_dc"])
    nd_weighted_load = 0.0
    for h in history:
        for dc in h["per_dc"]:
            nd_weighted_load += dc.get("grid_mw", 0.0) * dc.get("net_demand", 0.0)

    dc_costs = {
        history[0]["per_dc"][i]["name"]: sum(
            h["per_dc"][i]["energy_cost"] for h in history
        )
        for i in range(n_dc)
    }
    dc_peak = {
        history[0]["per_dc"][i]["name"]: sum(
            h["per_dc"][i].get("peak_penalty", 0.0) for h in history
        )
        for i in range(n_dc)
    }
    dc_backlogs = {
        history[0]["per_dc"][i]["name"]: float(np.mean(
            [h["per_dc"][i]["backlog"] for h in history]
        ))
        for i in range(n_dc)
    }

    # Demand charge (EVALUATION-ONLY unless demand_charge_rate > 0 in the env).
    # Commercial/industrial customers are billed on the highest demand interval
    # of the billing period, per meter — so the billed quantity is the SUM of
    # per-site peaks, not the fleet coincident peak. Reported for every policy
    # regardless of whether the term was in the reward, because it is a real
    # cost the operator pays and the shaped peak penalty Φ does not represent
    # it (Φ is a per-step quadratic grid-stress shadow price; see §3.3).
    per_dc_billed_peak = {
        history[0]["per_dc"][i]["name"]: float(max(
            h["per_dc"][i].get("grid_mw", 0.0) for h in history
        ))
        for i in range(n_dc)
    }
    billed_peak_sum_mw = float(sum(per_dc_billed_peak.values()))
    # Episodes are ~31 days, so the episode max is the monthly billed peak.
    demand_charge_ref = billed_peak_sum_mw * 1000.0 * REFERENCE_DEMAND_CHARGE_RATE
    # In-reward charge actually paid (0.0 when the term is disabled).
    total_demand_charge = sum(h.get("total_demand_charge", 0.0) for h in history)

    # Normalized power-draw profile (demand-smoothing KPI). load_factor = mean/peak
    # aggregate grid draw; higher = flatter (less peaky). Status Quo should have the
    # lowest load_factor; the optimizer should raise it by shaving peaks.
    agg_grid_mw = [h.get("total_grid_mw", 0.0) for h in history]
    peak_grid_mw = float(max(agg_grid_mw)) if agg_grid_mw else 0.0
    mean_grid_mw = float(np.mean(agg_grid_mw)) if agg_grid_mw else 0.0
    load_factor = mean_grid_mw / peak_grid_mw if peak_grid_mw > 0 else 0.0

    # Cost-component breakdown + backlog audit (peer-review M4): every policy is
    # scored on the full shaped objective, so report the components separately,
    # plus terminal state (finite-horizon leakage check) and served-vs-demand.
    total_backlog_cost = sum(
        sum(dc.get("backlog_cost", 0.0) for dc in h["per_dc"]) for h in history
    )
    total_capacity_cost = sum(
        sum(dc.get("capacity_cost", 0.0) for dc in h["per_dc"]) for h in history
    )
    terminal_backlog = float(sum(dc.get("backlog", 0.0) for dc in history[-1]["per_dc"]))
    terminal_batch_pool = float(history[-1].get("total_batch_pool", 0.0))
    demand_total = float(sum(h.get("total_demand", 0.0) for h in history))
    served_key = "service_served" if batch_enabled else "served"
    served_total = float(
        sum(sum(dc.get(served_key, 0.0) for dc in h["per_dc"]) for h in history)
    )
    max_backlog = float(max(
        sum(dc.get("backlog", 0.0) for dc in h["per_dc"]) for h in history
    ))

    summary: dict[str, Any] = {
        "total_cost": float(total_cost),
        "peak_grid_mw": peak_grid_mw,
        "mean_grid_mw": mean_grid_mw,
        "load_factor": load_factor,
        "total_energy_cost": float(total_energy_cost),
        "total_backlog_cost": float(total_backlog_cost),
        "total_capacity_cost": float(total_capacity_cost),
        "terminal_backlog": terminal_backlog,
        "terminal_batch_pool": terminal_batch_pool,
        "max_backlog": max_backlog,
        "service_demand_total": demand_total,
        "service_served_total": served_total,
        "total_peak_penalty": float(total_peak_penalty),
        "total_demand_charge": float(total_demand_charge),
        "billed_peak_sum_mw": billed_peak_sum_mw,
        "demand_charge_ref": float(demand_charge_ref),
        "demand_charge_ref_rate": float(REFERENCE_DEMAND_CHARGE_RATE),
        "total_grid_mw_steps": float(total_grid_mw),
        "nd_weighted_load": float(nd_weighted_load),
        "per_dc_energy_cost": dc_costs,
        "per_dc_peak_penalty": dc_peak,
        "per_dc_billed_peak_mw": per_dc_billed_peak,
        "per_dc_avg_backlog": dc_backlogs,
    }

    if batch_enabled:
        total_expired = sum(h.get("total_batch_expired", 0) for h in history)
        total_deadline_cost = sum(
            sum(dc.get("deadline_cost", 0) for dc in h["per_dc"])
            for h in history
        )
        avg_batch_pool = float(
            np.mean([h.get("total_batch_pool", 0) for h in history])
        )
        avg_drain_rates = {
            history[0]["per_dc"][i]["name"]: float(
                np.mean([h["per_dc"][i].get("drain_rate", 0) for h in history])
            )
            for i in range(n_dc)
        }
        summary.update(
            {
                "total_batch_expired": total_expired,
                "total_deadline_cost": total_deadline_cost,
                "avg_batch_pool_size": avg_batch_pool,
                "avg_drain_rates": avg_drain_rates,
            }
        )

    return summary


def _make_env(
    scenario_path: Path,
    batch_enabled: bool = False,
    flexibility_factor: float = 1.0,
    deadline_penalty_weight: float = 2.0,
    memory_enabled: bool = False,
    dynamic_arrivals: bool = True,
    seed: int = 42,
    peak_penalty_weight: float = 0.0,
    demand_charge_rate: float = 0.0,
    demand_charge_period_steps: int = 288,
    batch_spatial_routing: bool = True,
) -> MultiDCEnv:
    """Create environment for evaluation."""
    sites, power_model, batch_config = load_scenario(
        scenario_path,
        batch_enabled=batch_enabled,
        dynamic_arrivals=dynamic_arrivals,
        seed=seed,
    )
    ff = batch_config.get("flexibility_factor", flexibility_factor)
    dp = batch_config.get("deadline_penalty_weight", deadline_penalty_weight)
    uh = batch_config.get("urgency_horizon_steps", 12)
    return MultiDCEnv(
        sites=sites,
        power_model=power_model,
        batch_enabled=batch_enabled,
        flexibility_factor=ff,
        deadline_penalty_weight=dp,
        urgency_horizon_steps=uh,
        memory_enabled=memory_enabled,
        peak_penalty_weight=peak_penalty_weight,
        demand_charge_rate=demand_charge_rate,
        demand_charge_period_steps=demand_charge_period_steps,
        batch_spatial_routing=batch_spatial_routing,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO vs baselines")
    parser.add_argument(
        "--scenario", type=Path, required=True, help="Scenario YAML config"
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="Path to saved PPO model"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for report and plots",
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Enable batch scheduling mode for evaluation",
    )
    parser.add_argument(
        "--flexibility-factor",
        type=float,
        default=1.0,
        help="Deadline flexibility factor (default: 1.0)",
    )
    parser.add_argument(
        "--deadline-penalty",
        type=float,
        default=2.0,
        help="Deadline violation penalty weight (default: 2.0)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable memory as a constraint dimension",
    )
    parser.add_argument(
        "--no-dynamic-arrivals",
        action="store_true",
        help="Disable dynamic batch arrivals (use static fraction split)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="RL algorithm used for the trained model (default: ppo)",
    )
    parser.add_argument(
        "--dqn-model",
        type=Path,
        default=None,
        help="Path to a second (DQN routing-grid) model for comparison",
    )
    parser.add_argument(
        "--dqn-flatidx-model",
        type=Path,
        default=None,
        help="Path to a third (DQN CFWS-style flattened-index) model for comparison",
    )
    parser.add_argument(
        "--peak-penalty-weight",
        type=float,
        default=0.0,
        help="Peak-contribution penalty weight α (must match training value)",
    )
    parser.add_argument(
        "--demand-charge-rate",
        type=float,
        default=0.0,
        help="Demand charge in $/kW-month, billed on the highest demand "
             "interval of each billing period (the real commercial tariff "
             "term). Default 0.0 = not in the reward; evaluation reports the "
             "charge at a reference rate either way",
    )
    parser.add_argument(
        "--demand-charge-period-steps",
        type=int,
        default=288,
        help="Billing window in steps (default 288 = daily). Monthly (8640) "
             "is the realistic tariff but far exceeds the gamma=0.99 credit "
             "horizon (~100 steps)",
    )
    parser.add_argument(
        "--no-batch-spatial-routing",
        dest="batch_spatial_routing",
        action="store_false",
        help="Disable spatial routing of drained batch work (batch mode only); "
             "drained batch executes at its home DC. Must match how the model "
             "was trained (action space differs: 3N with routing, 2N without).",
    )
    args = parser.parse_args(argv)

    scenario_name = args.scenario.stem
    if args.batch_mode:
        scenario_name += "_batch"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    # Load primary trained model
    print(f"Loading {args.algorithm.upper()} model from {args.model}...")
    if args.algorithm == "dqn":
        from env.discrete_wrapper import DiscretizedMultiDCEnv

        rl_model = DQN.load(args.model)
        rl_label = "DQN"
    else:
        rl_model = PPO.load(args.model)
        rl_label = "PPO"

    # Evaluate primary model
    print(f"Evaluating {rl_label} agent...")
    env = _make_env(
        args.scenario,
        batch_enabled=args.batch_mode,
        flexibility_factor=args.flexibility_factor,
        deadline_penalty_weight=args.deadline_penalty,
        memory_enabled=args.memory,
        dynamic_arrivals=not args.no_dynamic_arrivals,
        peak_penalty_weight=args.peak_penalty_weight,
        demand_charge_rate=args.demand_charge_rate,
        demand_charge_period_steps=args.demand_charge_period_steps,
        batch_spatial_routing=args.batch_spatial_routing,
    )
    if args.algorithm == "dqn":
        env = DiscretizedMultiDCEnv(env)
    rl_reward, rl_history = run_episode(env, rl_model.predict, is_sb3=True)
    rl_summary = compute_summary(rl_history, batch_enabled=args.batch_mode)
    print(f"  {rl_label} total cost: {rl_summary['total_cost']:.2f}")
    results[rl_label] = {"reward": rl_reward, "summary": rl_summary, "history": rl_history}

    # Optionally load a second model (DQN routing-grid for comparison)
    if args.dqn_model is not None and args.algorithm != "dqn":
        from env.discrete_wrapper import DiscretizedMultiDCEnv

        print(f"Loading DQN (routing-grid) model from {args.dqn_model}...")
        dqn_model = DQN.load(args.dqn_model)
        env2 = _make_env(
            args.scenario,
            batch_enabled=args.batch_mode,
            flexibility_factor=args.flexibility_factor,
            deadline_penalty_weight=args.deadline_penalty,
            memory_enabled=args.memory,
            dynamic_arrivals=not args.no_dynamic_arrivals,
            peak_penalty_weight=args.peak_penalty_weight,
        demand_charge_rate=args.demand_charge_rate,
        demand_charge_period_steps=args.demand_charge_period_steps,
        )
        env2 = DiscretizedMultiDCEnv(env2)
        dqn_reward, dqn_history = run_episode(env2, dqn_model.predict, is_sb3=True)
        dqn_summary = compute_summary(dqn_history, batch_enabled=args.batch_mode)
        print(f"  DQN total cost: {dqn_summary['total_cost']:.2f}")
        results["DQN"] = {"reward": dqn_reward, "summary": dqn_summary, "history": dqn_history}

    # Optionally load a third model (CFWS-style flattened-index DQN)
    if args.dqn_flatidx_model is not None:
        from env.cfws_style_wrapper import CFWSStyleDiscretizedEnv

        print(f"Loading DQN (CFWS flat-index) model from {args.dqn_flatidx_model}...")
        dqn_flat = DQN.load(args.dqn_flatidx_model)
        env3 = _make_env(
            args.scenario,
            batch_enabled=args.batch_mode,
            flexibility_factor=args.flexibility_factor,
            deadline_penalty_weight=args.deadline_penalty,
            memory_enabled=args.memory,
            dynamic_arrivals=not args.no_dynamic_arrivals,
            peak_penalty_weight=args.peak_penalty_weight,
        demand_charge_rate=args.demand_charge_rate,
        demand_charge_period_steps=args.demand_charge_period_steps,
        )
        env3 = CFWSStyleDiscretizedEnv(env3)
        df_reward, df_history = run_episode(env3, dqn_flat.predict, is_sb3=True)
        df_summary = compute_summary(df_history, batch_enabled=args.batch_mode)
        print(f"  DQN-flatidx total cost: {df_summary['total_cost']:.2f}")
        results["DQN-flatidx"] = {
            "reward": df_reward, "summary": df_summary, "history": df_history,
        }

    ppo_summary = rl_summary  # for dc_names later

    for baseline_cls in ALL_BASELINES:
        baseline = baseline_cls()
        print(f"Evaluating {baseline.name}...")
        env = _make_env(
            args.scenario,
            batch_enabled=args.batch_mode,
            flexibility_factor=args.flexibility_factor,
            deadline_penalty_weight=args.deadline_penalty,
            peak_penalty_weight=args.peak_penalty_weight,
        demand_charge_rate=args.demand_charge_rate,
        demand_charge_period_steps=args.demand_charge_period_steps,
            batch_spatial_routing=args.batch_spatial_routing,
        )
        reward, history = run_episode(env, baseline.predict, is_sb3=False)
        summary = compute_summary(history, batch_enabled=args.batch_mode)
        results[baseline.name] = {
            "reward": reward,
            "summary": summary,
            "history": history,
        }
        print(f"  {baseline.name} total cost: {summary['total_cost']:.2f}")

    # --- Generate Report ---
    print("\nGenerating report...")

    # Summary table
    report_lines = [
        f"# Evaluation Report: {scenario_name}\n",
        "## Summary\n",
    ]

    if args.batch_mode:
        report_lines.extend(
            [
                "| Policy | Total Cost | Energy Cost | Peak Penalty | ND-weighted Load | Batch Expired | Deadline Cost | Avg Pool Size |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name, data in results.items():
            s = data["summary"]
            report_lines.append(
                f"| {name} | {s['total_cost']:.2f} | "
                f"{s.get('total_energy_cost', 0):.2f} | "
                f"{s.get('total_peak_penalty', 0):.2f} | "
                f"{s.get('nd_weighted_load', 0):.0f} | "
                f"{s.get('total_batch_expired', 0):.4f} | "
                f"{s.get('total_deadline_cost', 0):.2f} | "
                f"{s.get('avg_batch_pool_size', 0):.4f} |"
            )
    else:
        report_lines.extend(
            [
                "| Policy | Total Cost | Energy Cost | Peak Penalty | ND-weighted Load | Total Grid (MW-steps) |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name, data in results.items():
            s = data["summary"]
            report_lines.append(
                f"| {name} | {s['total_cost']:.2f} | "
                f"{s.get('total_energy_cost', 0):.2f} | "
                f"{s.get('total_peak_penalty', 0):.2f} | "
                f"{s.get('nd_weighted_load', 0):.0f} | "
                f"{s['total_grid_mw_steps']:.2f} |"
            )

    report_lines.append("\n## Per-DC Energy Cost Breakdown\n")
    dc_names = list(ppo_summary["per_dc_energy_cost"].keys())
    header = "| Policy | " + " | ".join(dc_names) + " |"
    sep = "| --- " * (len(dc_names) + 1) + "|"
    report_lines.extend([header, sep])

    for name, data in results.items():
        costs = data["summary"]["per_dc_energy_cost"]
        row = (
            f"| {name} | "
            + " | ".join(f"{costs[dc]:.2f}" for dc in dc_names)
            + " |"
        )
        report_lines.append(row)

    # --- Plots ---

    # 1. Cumulative cost comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, data in results.items():
        costs = [h["total_cost"] for h in data["history"]]
        ax.plot(
            np.cumsum(costs),
            label=name,
            linewidth=1.5 if name == "PPO" else 0.8,
        )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Cost ($)")
    ax.set_title(f"Cumulative Cost: {scenario_name}")
    ax.legend()
    fig.tight_layout()
    cost_path = args.output_dir / f"{scenario_name}_cumulative_cost.png"
    fig.savefig(cost_path, dpi=150)
    plt.close(fig)

    # 2. Per-policy load draw weighted by current grid net demand.
    # A policy that successfully smooths demand keeps this line low when
    # the duck-curve neck (net demand) is high.
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, data in results.items():
        nd_weighted = []
        for h in data["history"]:
            total = 0.0
            for dc in h["per_dc"]:
                total += dc.get("grid_mw", 0.0) * dc.get("net_demand", 0.0)
            nd_weighted.append(total)
        window = 12  # 1-hour smoothing
        smoothed = pd.Series(nd_weighted).rolling(window, min_periods=1).mean()
        ax.plot(
            smoothed,
            label=name,
            linewidth=1.5 if name in ("PPO", "DQN") else 0.8,
        )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Σ grid_mw × net_demand_norm (smoothed)")
    ax.set_title(f"DC Contribution to Grid Stress: {scenario_name}")
    ax.legend()
    fig.tight_layout()
    renew_path = args.output_dir / f"{scenario_name}_peak_contribution.png"
    fig.savefig(renew_path, dpi=150)
    plt.close(fig)

    # 3. PPO allocation heatmap
    n_dc = len(dc_names)
    fractions_matrix = np.array(
        [h["fractions"] for h in rl_history]
    )  # (T, n_dc)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(
        fractions_matrix.T,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    ax.set_yticks(range(n_dc))
    ax.set_yticklabels(dc_names)
    ax.set_xlabel("Timestep")
    ax.set_title(f"PPO Allocation Heatmap: {scenario_name}")
    fig.colorbar(im, ax=ax, label="Allocation Fraction")
    fig.tight_layout()
    heatmap_path = args.output_dir / f"{scenario_name}_allocation_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    plot_refs = [
        f"![Cumulative Cost]({cost_path.name})\n",
        f"![Peak Contribution]({renew_path.name})\n",
        f"![Allocation Heatmap]({heatmap_path.name})\n",
    ]

    # 4 & 5. Batch-specific plots
    if args.batch_mode:
        # Batch pool size over time
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        pool_sizes = [h.get("total_batch_pool", 0) for h in rl_history]
        ax1.plot(pool_sizes, label="Total Batch Pool", color="tab:blue")
        ax1.set_ylabel("Batch Pool Size (norm CPU)")
        ax1.set_title(f"Batch Pool Evolution: {scenario_name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for i in range(n_dc):
            drain_rates = [
                h["per_dc"][i].get("drain_rate", 0) for h in rl_history
            ]
            ax2.plot(drain_rates, label=dc_names[i], alpha=0.7)
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Drain Rate")
        ax2.set_title("Per-DC Drain Rates (PPO)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        batch_path = args.output_dir / f"{scenario_name}_batch_pool.png"
        fig.savefig(batch_path, dpi=150)
        plt.close(fig)
        plot_refs.append(f"![Batch Pool]({batch_path.name})\n")

        # Drain rate heatmap
        drain_matrix = np.array(
            [h.get("drain_rates", [0] * n_dc) for h in rl_history]
        )
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(
            drain_matrix.T,
            aspect="auto",
            cmap="YlGnBu",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_yticks(range(n_dc))
        ax.set_yticklabels(dc_names)
        ax.set_xlabel("Timestep")
        ax.set_title(f"PPO Drain Rate Heatmap: {scenario_name}")
        fig.colorbar(im, ax=ax, label="Drain Rate")
        fig.tight_layout()
        drain_path = args.output_dir / f"{scenario_name}_drain_heatmap.png"
        fig.savefig(drain_path, dpi=150)
        plt.close(fig)
        plot_refs.append(f"![Drain Heatmap]({drain_path.name})\n")

    # Write report
    report_lines.extend(["\n## Plots\n"] + plot_refs)

    report_path = args.output_dir / f"{scenario_name}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")

    # Save raw results as JSON (for further analysis)
    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            "reward": data["reward"],
            "summary": data["summary"],
        }
    json_path = args.output_dir / f"{scenario_name}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Raw results saved to {json_path}")


if __name__ == "__main__":
    main()
