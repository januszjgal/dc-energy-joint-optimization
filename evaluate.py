"""Evaluate a trained PPO agent against all baselines.

Runs one full episode for the PPO agent and each baseline policy,
collects per-timestep metrics, and produces a summary report with plots.

Usage:
    python evaluate.py --scenario env/scenarios/us_model.yaml --model models/ppo_us_model
    python evaluate.py --scenario env/scenarios/global_model.yaml --model models/ppo_global_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from baselines import ALL_BASELINES
from env.data_loader import load_scenario
from env.multi_dc_env import MultiDCEnv


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


def compute_summary(history: list[dict]) -> dict:
    """Compute summary metrics from episode history."""
    total_cost = sum(h["total_cost"] for h in history)
    avg_renewable = np.mean([h["renewable_frac"] for h in history])

    # Per-DC metrics
    n_dc = len(history[0]["per_dc"])
    dc_costs = {
        history[0]["per_dc"][i]["name"]: sum(
            h["per_dc"][i]["energy_cost"] for h in history
        )
        for i in range(n_dc)
    }
    dc_backlogs = {
        history[0]["per_dc"][i]["name"]: np.mean(
            [h["per_dc"][i]["backlog"] for h in history]
        )
        for i in range(n_dc)
    }

    total_grid = sum(
        sum(h["per_dc"][i]["grid_mw"] for h in history) for i in range(n_dc)
    )

    return {
        "total_cost": total_cost,
        "avg_renewable_frac": avg_renewable,
        "total_grid_mw_steps": total_grid,
        "per_dc_energy_cost": dc_costs,
        "per_dc_avg_backlog": dc_backlogs,
    }


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
    args = parser.parse_args(argv)

    scenario_name = args.scenario.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load environment
    sites, power_model = load_scenario(args.scenario)

    # Load trained model
    print(f"Loading model from {args.model}...")
    ppo_model = PPO.load(args.model)

    # Evaluate PPO
    print("Evaluating PPO agent...")
    env = MultiDCEnv(sites=sites, power_model=power_model)
    ppo_reward, ppo_history = run_episode(env, ppo_model.predict, is_sb3=True)
    ppo_summary = compute_summary(ppo_history)
    print(f"  PPO total cost: {ppo_summary['total_cost']:.2f}")

    # Evaluate baselines
    results = {"PPO": {"reward": ppo_reward, "summary": ppo_summary, "history": ppo_history}}

    for baseline_cls in ALL_BASELINES:
        baseline = baseline_cls()
        print(f"Evaluating {baseline.name}...")
        env = MultiDCEnv(sites=sites, power_model=power_model)
        reward, history = run_episode(env, baseline.predict, is_sb3=False)
        summary = compute_summary(history)
        results[baseline.name] = {"reward": reward, "summary": summary, "history": history}
        print(f"  {baseline.name} total cost: {summary['total_cost']:.2f}")

    # --- Generate Report ---
    print("\nGenerating report...")

    # Summary table
    report_lines = [
        f"# Evaluation Report: {scenario_name}\n",
        "## Summary\n",
        "| Policy | Total Cost | Avg Renewable % | Total Grid (MW-steps) |",
        "| --- | --- | --- | --- |",
    ]
    for name, data in results.items():
        s = data["summary"]
        report_lines.append(
            f"| {name} | {s['total_cost']:.2f} | "
            f"{s['avg_renewable_frac']*100:.1f}% | "
            f"{s['total_grid_mw_steps']:.2f} |"
        )

    report_lines.append("\n## Per-DC Energy Cost Breakdown\n")
    dc_names = list(ppo_summary["per_dc_energy_cost"].keys())
    header = "| Policy | " + " | ".join(dc_names) + " |"
    sep = "| --- " * (len(dc_names) + 1) + "|"
    report_lines.extend([header, sep])

    for name, data in results.items():
        costs = data["summary"]["per_dc_energy_cost"]
        row = f"| {name} | " + " | ".join(f"{costs[dc]:.2f}" for dc in dc_names) + " |"
        report_lines.append(row)

    # --- Plots ---

    # 1. Cumulative cost comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, data in results.items():
        costs = [h["total_cost"] for h in data["history"]]
        ax.plot(np.cumsum(costs), label=name, linewidth=1.5 if name == "PPO" else 0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Cost ($)")
    ax.set_title(f"Cumulative Cost: {scenario_name}")
    ax.legend()
    fig.tight_layout()
    cost_path = args.output_dir / f"{scenario_name}_cumulative_cost.png"
    fig.savefig(cost_path, dpi=150)
    plt.close(fig)

    # 2. Renewable utilization over time
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, data in results.items():
        renew = [h["renewable_frac"] for h in data["history"]]
        # Smooth with rolling average
        window = 12  # 1-hour window
        smoothed = pd.Series(renew).rolling(window, min_periods=1).mean()
        ax.plot(smoothed, label=name, linewidth=1.5 if name == "PPO" else 0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Renewable Fraction")
    ax.set_title(f"Renewable Utilization: {scenario_name}")
    ax.legend()
    fig.tight_layout()
    renew_path = args.output_dir / f"{scenario_name}_renewable.png"
    fig.savefig(renew_path, dpi=150)
    plt.close(fig)

    # 3. PPO allocation heatmap
    n_dc = len(dc_names)
    fractions_matrix = np.array(
        [h["fractions"] for h in ppo_history]
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

    # Write report
    report_lines.extend([
        "\n## Plots\n",
        f"![Cumulative Cost]({cost_path.name})\n",
        f"![Renewable Utilization]({renew_path.name})\n",
        f"![Allocation Heatmap]({heatmap_path.name})\n",
    ])

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
