"""Evaluate a trained PPO agent against the active baseline set.

Runs one full episode for the PPO agent and each baseline policy,
collects per-timestep metrics, and produces a summary report with plots.

The frozen v2 campaign must supply objective/completion flags from its protocol;
ad hoc defaults are not a valid headline evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from baselines import (
    DrainImmediatelyPolicy,
    RoundRobinPolicy,
    StatusQuoPolicy,
)
from env.data_loader import load_scenario
from env.multi_dc_env import REFERENCE_DEMAND_CHARGE_RATE, MultiDCEnv
from env.reward import RewardConfig
from env.safe_multi_dc_env import SafeMultiDCEnv
from env.safety_layer import SafetyConfig


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


def run_normalized_episode(
    env: MultiDCEnv,
    model: PPO,
    vecnormalize_path: Path,
) -> tuple[float, list[dict[str, Any]]]:
    """Evaluate a PPO model with its frozen observation-normalization state."""
    max_steps = env.max_steps
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    obs = vec_env.reset()
    total_reward = 0.0
    history: list[dict[str, Any]] = []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        total_reward += float(rewards[0])
        history.append(infos[0])
        if bool(dones[0]):
            break
    vec_env.close()
    if len(history) != max_steps:
        raise RuntimeError(
            "normalized evaluation ended before the full episode: "
            f"{len(history)} != {max_steps}"
        )
    return total_reward, history


def compute_ramp_metrics(
    history: list[dict[str, Any]],
    horizon_steps: int,
) -> dict[str, Any] | None:
    """Compare raw-grid and grid-plus-DC upward ramps across physical sites."""
    if len(history) <= horizon_steps:
        return None

    per_dc: dict[str, Any] = {}
    all_base: list[np.ndarray] = []
    all_combined: list[np.ndarray] = []
    n_dc = len(history[0]["per_dc"])

    for i in range(n_dc):
        name = history[0]["per_dc"][i]["name"]
        raw = [h["per_dc"][i].get("net_demand_mw") for h in history]
        if any(value is None for value in raw):
            return None
        base = np.asarray(raw, dtype=np.float64)
        dc_load = np.asarray(
            [h["per_dc"][i].get("grid_mw", 0.0) for h in history],
            dtype=np.float64,
        )
        combined = base + dc_load
        base_up = np.maximum(
            base[horizon_steps:] - base[:-horizon_steps],
            0.0,
        )
        combined_up = np.maximum(
            combined[horizon_steps:] - combined[:-horizon_steps],
            0.0,
        )
        all_base.append(base_up)
        all_combined.append(combined_up)
        per_dc[name] = {
            "base_max_up_mw": float(base_up.max()),
            "with_dc_max_up_mw": float(combined_up.max()),
            "max_up_delta_mw": float(
                combined_up.max() - base_up.max()
            ),
            "base_p95_up_mw": float(np.percentile(base_up, 95)),
            "with_dc_p95_up_mw": float(
                np.percentile(combined_up, 95)
            ),
            "p95_up_delta_mw": float(
                np.percentile(combined_up, 95)
                - np.percentile(base_up, 95)
            ),
        }

    base_all = np.concatenate(all_base)
    combined_all = np.concatenate(all_combined)
    return {
        "interpretation": (
            "Distribution across separate physical sites; regional net demand "
            "is not summed into a fictitious global grid."
        ),
        "base_max_up_mw": float(base_all.max()),
        "with_dc_max_up_mw": float(combined_all.max()),
        "max_up_delta_mw": float(
            combined_all.max() - base_all.max()
        ),
        "base_p95_up_mw": float(np.percentile(base_all, 95)),
        "with_dc_p95_up_mw": float(
            np.percentile(combined_all, 95)
        ),
        "p95_up_delta_mw": float(
            np.percentile(combined_all, 95)
            - np.percentile(base_all, 95)
        ),
        "per_dc": per_dc,
    }


def compute_summary(history: list[dict], batch_enabled: bool = False) -> dict:
    """Compute summary metrics from episode history."""
    first_step = history[0]
    total_cost = sum(h["total_cost"] for h in history)
    total_energy_cost = sum(h.get("total_energy_cost", 0.0) for h in history)
    total_peak_penalty = sum(h.get("total_peak_penalty", 0.0) for h in history)
    total_grid_mw = sum(h.get("total_grid_mw", 0.0) for h in history)
    total_reward_training_cost = sum(
        h.get("reward_training_cost", h.get("total_cost", 0.0))
        for h in history
    )
    total_reward_idle_cost = sum(
        h.get("reward_idle_cost", 0.0) for h in history
    )
    total_reward_potential_delta = sum(
        h.get("reward_potential_delta", 0.0) for h in history
    )
    total_reward_penalty_adjustment = sum(
        h.get("reward_penalty_adjustment", 0.0) for h in history
    )

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

    # Full-episode reference charge. Billing is per meter, so the quantity is
    # the sum of site maxima rather than the coincident fleet peak. It is
    # reported even when disabled because Φ is a different grid-stress term.
    per_dc_billed_peak = {
        history[0]["per_dc"][i]["name"]: float(max(
            h["per_dc"][i].get("grid_mw", 0.0) for h in history
        ))
        for i in range(n_dc)
    }
    billed_peak_sum_mw = float(sum(per_dc_billed_peak.values()))
    # The 31-day study episode is treated as one reference billing cycle.
    demand_charge_ref = billed_peak_sum_mw * 1000.0 * REFERENCE_DEMAND_CHARGE_RATE
    # In-reward charge actually paid under the environment's configured
    # period(s) and rate (0.0 when the term is disabled).
    total_demand_charge = sum(h.get("total_demand_charge", 0.0) for h in history)
    demand_charge_enabled = bool(first_step.get("demand_charge_enabled", False))
    demand_charge_rate = float(first_step.get("demand_charge_rate", 0.0))
    demand_charge_period_steps = int(
        first_step.get("demand_charge_period_steps", len(history))
    )
    demand_charge_period_count = len({
        int(h.get("demand_charge_period_index", 0)) for h in history
    })
    economic_penalty_floor = float(
        first_step.get("economic_penalty_floor", 0.0)
    )

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
        "demand_charge_enabled": demand_charge_enabled,
        "demand_charge_rate": demand_charge_rate,
        "demand_charge_rate_unit": first_step.get(
            "demand_charge_rate_unit", "USD_per_kW_per_billing_period"
        ),
        "demand_charge_period_steps": demand_charge_period_steps,
        "demand_charge_period_count": demand_charge_period_count,
        "demand_charge_penalty_guard": bool(
            first_step.get("demand_charge_penalty_guard", False)
        ),
        "economic_penalty_floor": economic_penalty_floor,
        "configured_backlog_weight": float(
            first_step.get("configured_backlog_weight", 25.0)
        ),
        "effective_backlog_weight": float(
            first_step.get("effective_backlog_weight", 25.0)
        ),
        "configured_deadline_penalty_weight": float(
            first_step.get("configured_deadline_penalty_weight", 2.0)
        ),
        "effective_deadline_penalty_weight": float(
            first_step.get("effective_deadline_penalty_weight", 2.0)
        ),
        "reward_scale": float(first_step.get("reward_scale", 1.0)),
        "reward_training_cost": float(total_reward_training_cost),
        "reward_idle_cost_subtracted": float(total_reward_idle_cost),
        "reward_potential_delta": float(total_reward_potential_delta),
        "reward_penalty_adjustment": float(
            total_reward_penalty_adjustment
        ),
        "reward_idle_cost_subtraction": bool(
            first_step.get("reward_idle_cost_subtraction", False)
        ),
        "reward_urgency_potential_weight": float(
            first_step.get("reward_urgency_potential_weight", 0.0)
        ),
        "batch_completion_weight": float(
            first_step.get(
                "batch_completion_weight",
                first_step.get("effective_deadline_penalty_weight", 0.0),
            )
        ),
        "reward_service_backlog_weight": float(
            first_step.get(
                "reward_service_backlog_weight",
                first_step.get("effective_backlog_weight", 0.0),
            )
        ),
        "reward_batch_completion_weight": float(
            first_step.get(
                "reward_batch_completion_weight",
                first_step.get("effective_deadline_penalty_weight", 0.0),
            )
        ),
        "observe_episode_progress": bool(
            first_step.get("observe_episode_progress", False)
        ),
        "deadline_bucket_edges": list(
            first_step.get("deadline_bucket_edges", [])
        ),
        "batch_completion_shaping_enabled": bool(
            first_step.get("batch_completion_shaping_enabled", False)
        ),
        "billed_peak_sum_mw": billed_peak_sum_mw,
        "demand_charge_ref": float(demand_charge_ref),
        "demand_charge_ref_rate": float(REFERENCE_DEMAND_CHARGE_RATE),
        "demand_charge_ref_period_steps": len(history),
        "total_grid_mw_steps": float(total_grid_mw),
        "nd_weighted_load": float(nd_weighted_load),
        "per_dc_energy_cost": dc_costs,
        "per_dc_peak_penalty": dc_peak,
        "per_dc_billed_peak_mw": per_dc_billed_peak,
        "per_dc_avg_backlog": dc_backlogs,
    }
    ramp_metrics = {}
    for label, steps in (("1h", 12), ("3h", 36)):
        metrics = compute_ramp_metrics(history, steps)
        if metrics is not None:
            ramp_metrics[label] = metrics
    summary["physical_ramp_metrics"] = ramp_metrics

    if bool(first_step.get("safety_enabled", False)):
        intervention_values = [
            bool(
                row.get(
                    "safety_emergency_intervened",
                    row.get("safety_intervened", False),
                )
            )
            for row in history
        ]
        projection_values = np.asarray(
            [
                row.get("safety_projection_l2", 0.0)
                for row in history
            ],
            dtype=np.float64,
        )
        decoder_adjusted_values = [
            bool(row.get("safety_decoder_adjusted", False)) for row in history
        ]
        decoder_adjustment_values = np.asarray(
            [
                row.get("safety_decoder_adjustment_l2", 0.0)
                for row in history
            ],
            dtype=np.float64,
        )
        emergency_adjustment_values = np.asarray(
            [
                row.get(
                    "safety_emergency_adjustment_l2",
                    row.get("safety_projection_l2", 0.0),
                )
                for row in history
            ],
            dtype=np.float64,
        )
        mandatory_values = np.asarray(
            [
                row.get("safety_mandatory_batch", 0.0)
                for row in history
            ],
            dtype=np.float64,
        )
        binding_counts: dict[str, int] = {}
        for row in history:
            binding = row.get(
                "safety_binding_deadline_steps_remaining"
            )
            if binding is not None:
                key = str(int(binding))
                binding_counts[key] = binding_counts.get(key, 0) + 1
        slack_values = [
            float(row["safety_minimum_deadline_slack"])
            for row in history
            if math.isfinite(
                float(
                    row.get(
                        "safety_minimum_deadline_slack",
                        math.inf,
                    )
                )
            )
        ]
        summary["safety"] = {
            "enabled": True,
            "infeasibility_certificates": 0,
            "intervention_count": int(sum(intervention_values)),
            "intervention_rate": float(np.mean(intervention_values)),
            "emergency_intervention_count": int(sum(intervention_values)),
            "emergency_intervention_rate": float(np.mean(intervention_values)),
            "mean_projection_l2": float(projection_values.mean()),
            "max_projection_l2": float(projection_values.max()),
            "decoder_adjustment_count": int(sum(decoder_adjusted_values)),
            "decoder_adjustment_rate": float(np.mean(decoder_adjusted_values)),
            "mean_decoder_adjustment_l2": float(
                decoder_adjustment_values.mean()
            ),
            "max_decoder_adjustment_l2": float(
                decoder_adjustment_values.max()
            ),
            "mean_emergency_adjustment_l2": float(
                emergency_adjustment_values.mean()
            ),
            "max_emergency_adjustment_l2": float(
                emergency_adjustment_values.max()
            ),
            "mandatory_step_count": int(
                np.count_nonzero(mandatory_values > 1e-12)
            ),
            "mandatory_step_rate": float(
                np.mean(mandatory_values > 1e-12)
            ),
            "mean_mandatory_batch": float(mandatory_values.mean()),
            "max_mandatory_batch": float(mandatory_values.max()),
            "binding_deadline_histogram": dict(
                sorted(binding_counts.items(), key=lambda item: int(item[0]))
            ),
            "minimum_deadline_slack": (
                float(min(slack_values)) if slack_values else None
            ),
            "exact_zero_drain_count": int(
                sum(
                    row.get("safety_exact_zero_drain_count", 0)
                    for row in history
                )
            ),
            "exact_full_drain_count": int(
                sum(
                    row.get("safety_exact_full_drain_count", 0)
                    for row in history
                )
            ),
            "negative_flush_step_count": int(
                sum(
                    bool(
                        row.get(
                            "safety_negative_flush_active",
                            False,
                        )
                    )
                    for row in history
                )
            ),
            "negative_flush_activation_rate": float(
                np.mean(
                    [
                        bool(
                            row.get(
                                "safety_negative_flush_active",
                                False,
                            )
                        )
                        for row in history
                    ]
                )
            ),
            "max_transport_conservation_error": float(
                max(
                    row.get(
                        "safety_transport_conservation_error",
                        0.0,
                    )
                    for row in history
                )
            ),
            "service_envelope_total": float(
                first_step["safety_service_envelope_total"]
            ),
            "batch_arrival_envelope_total": float(
                first_step["safety_batch_arrival_envelope_total"]
            ),
            "guaranteed_carried_batch_capacity": float(
                first_step[
                    "safety_guaranteed_carried_batch_capacity"
                ]
            ),
            "envelope_id": str(first_step["safety_envelope_id"]),
            "envelope_scope": str(
                first_step["safety_envelope_scope"]
            ),
        }

    if batch_enabled:
        total_expired = sum(h.get("total_batch_expired", 0) for h in history)
        total_deadline_cost = sum(
            sum(dc.get("deadline_cost", 0) for dc in h["per_dc"])
            for h in history
        )
        total_terminal_batch_cost = sum(
            h.get("total_terminal_batch_cost", 0.0) for h in history
        )
        total_batch_accounting_cost = sum(
            h.get("total_batch_accounting_cost", 0.0) for h in history
        )
        batch_arrival_total = float(sum(
            sum(dc.get("batch_arrival", 0.0) for dc in h["per_dc"])
            for h in history
        ))
        batch_completed_total = float(sum(
            sum(dc.get("batch_drained", 0.0) for dc in h["per_dc"])
            for h in history
        ))
        batch_completion_fraction = (
            batch_completed_total / batch_arrival_total
            if batch_arrival_total > 0.0
            else 1.0
        )
        work_demand_total = demand_total + batch_arrival_total
        work_completed_total = served_total + batch_completed_total
        work_completed_fraction = (
            work_completed_total / work_demand_total
            if work_demand_total > 0.0
            else 1.0
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
                "total_terminal_batch_cost": total_terminal_batch_cost,
                "total_batch_accounting_cost": total_batch_accounting_cost,
                "total_unfinished_batch_cost": (
                    total_batch_accounting_cost
                    if first_step.get(
                        "batch_completion_shaping_enabled", False
                    )
                    else total_deadline_cost + total_terminal_batch_cost
                ),
                "batch_arrival_total": batch_arrival_total,
                "batch_completed_total": batch_completed_total,
                "batch_completion_fraction": batch_completion_fraction,
                "work_demand_total": work_demand_total,
                "work_completed_total": work_completed_total,
                "work_completed_fraction": work_completed_fraction,
                "avg_batch_pool_size": avg_batch_pool,
                "avg_drain_rates": avg_drain_rates,
            }
        )

    return summary


def _make_env(
    scenario_path: Path,
    batch_enabled: bool = False,
    flexibility_factor: float | None = None,
    deadline_penalty_weight: float | None = None,
    memory_enabled: bool = False,
    dynamic_arrivals: bool = True,
    seed: int = 42,
    peak_penalty_weight: float = 0.0,
    demand_charge_rate: float = 0.0,
    demand_charge_period_steps: int | None = None,
    allow_multiple_demand_charge_periods: bool = False,
    enforce_batch_completion: bool = False,
    completion_penalty_weight: float | None = None,
    batch_spatial_routing: bool = True,
    reward_config: RewardConfig | None = None,
    observe_episode_progress: bool = False,
    deadline_bucket_edges: tuple[int, ...] | None = None,
    safety_config: SafetyConfig | None = None,
) -> MultiDCEnv:
    """Create environment for evaluation."""
    sites, power_model, batch_config = load_scenario(
        scenario_path,
        batch_enabled=batch_enabled,
        dynamic_arrivals=dynamic_arrivals,
        seed=seed,
    )
    ff = (
        batch_config.get("flexibility_factor", 1.0)
        if flexibility_factor is None
        else flexibility_factor
    )
    dp = (
        batch_config.get("deadline_penalty_weight", 2.0)
        if deadline_penalty_weight is None
        else deadline_penalty_weight
    )
    uh = batch_config.get("urgency_horizon_steps", 12)
    environment_class = (
        SafeMultiDCEnv if safety_config is not None else MultiDCEnv
    )
    environment_kwargs: dict[str, Any] = {}
    if safety_config is not None:
        environment_kwargs["safety_config"] = safety_config
    return environment_class(
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
        allow_multiple_demand_charge_periods=(
            allow_multiple_demand_charge_periods
        ),
        enforce_batch_completion=enforce_batch_completion,
        completion_penalty_weight=completion_penalty_weight,
        batch_spatial_routing=batch_spatial_routing,
        reward_config=reward_config,
        observe_episode_progress=observe_episode_progress,
        deadline_bucket_edges=deadline_bucket_edges,
        **environment_kwargs,
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
        default=None,
        help="Override scenario deadline flexibility.",
    )
    parser.add_argument(
        "--deadline-penalty",
        type=float,
        default=None,
        help="Override scenario deadline penalty weight.",
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
        "--peak-penalty-weight",
        type=float,
        default=0.0,
        help="Peak-contribution penalty weight alpha (must match training value)",
    )
    parser.add_argument(
        "--demand-charge-rate",
        type=float,
        default=0.0,
        help="Demand charge in $/kW per configured billing period, billed on "
             "the highest demand interval. Default 0.0 = not in the reward; "
             "evaluation reports a full-episode reference charge either way.",
    )
    parser.add_argument(
        "--demand-charge-period-steps",
        type=int,
        default=None,
        help="Billing period in steps. Default: the full episode is one "
             "billing cycle. A shorter value is a different tariff and needs "
             "a rate quoted for that period; it must divide max_steps exactly.",
    )
    parser.add_argument(
        "--allow-multiple-demand-charge-periods",
        action="store_true",
        help="Acknowledge a full configured rate charged once per shorter "
             "billing period.",
    )
    parser.add_argument(
        "--enforce-batch-completion",
        action="store_true",
        help="Evaluate with gamma-equivalent dense completion accounting and "
             "terminal-pool value. Batch mode only.",
    )
    parser.add_argument(
        "--completion-penalty",
        type=float,
        default=None,
        help="Fixed completion coefficient used during guarded training.",
    )
    parser.add_argument(
        "--no-batch-spatial-routing",
        dest="batch_spatial_routing",
        action="store_false",
        help="Disable spatial routing of drained batch work (batch mode only); "
             "drained batch executes at its home DC. Must match how the model "
             "was trained (action space differs: 3N with routing, 2N without).",
    )
    parser.add_argument(
        "--v3-recovery",
        action="store_true",
        help="Use the opt-in 81D joint PPO recovery state and reward config.",
    )
    parser.add_argument(
        "--v4-safety",
        action="store_true",
        help="Evaluate with the hard v4 feasibility projector active.",
    )
    parser.add_argument(
        "--negative-demand-flush",
        action="store_true",
        help="Enable the optional v4 negative-demand flush ablation.",
    )
    parser.add_argument(
        "--service-envelope-total",
        type=float,
        default=2.25,
    )
    parser.add_argument(
        "--batch-arrival-envelope-total",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--future-fleet-capacity-total",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--vecnormalize-path",
        type=Path,
        default=None,
        help="Saved VecNormalize statistics for a normalized v3 model.",
    )
    parser.add_argument(
        "--service-backlog-weight",
        type=float,
        default=1000.0,
        help="V3 training reward service weight; evaluation remains at 1000.",
    )
    parser.add_argument(
        "--batch-completion-weight",
        type=float,
        default=1000.0,
        help="V3 training reward batch weight; evaluation remains at 1000.",
    )
    parser.add_argument(
        "--subtract-idle-cost",
        action="store_true",
        help="Use the v3 action-independent idle-cost reward baseline.",
    )
    parser.add_argument(
        "--urgency-potential-weight",
        type=float,
        default=0.0,
        help="V3 policy-invariant urgency-potential coefficient.",
    )
    args = parser.parse_args(argv)
    if args.enforce_batch_completion and not args.batch_mode:
        parser.error("--enforce-batch-completion requires --batch-mode")
    if (
        args.completion_penalty is not None
        and not args.enforce_batch_completion
    ):
        parser.error(
            "--completion-penalty requires --enforce-batch-completion"
        )
    recovery_enabled = bool(args.v3_recovery or args.v4_safety)
    if recovery_enabled and (
        not args.batch_mode or not args.enforce_batch_completion
    ):
        parser.error(
            "--v3-recovery requires --batch-mode and "
            "--enforce-batch-completion"
        )
    if recovery_enabled and not args.batch_spatial_routing:
        parser.error(
            "--v3-recovery requires the joint batch-routing head; "
            "--no-batch-spatial-routing is incompatible"
        )
    if recovery_enabled and args.completion_penalty is not None:
        parser.error(
            "--v3-recovery uses separate reward weights; omit "
            "--completion-penalty"
        )
    if args.vecnormalize_path is not None:
        if not recovery_enabled:
            parser.error(
                "--vecnormalize-path requires --v3-recovery or --v4-safety"
            )
        if not args.vecnormalize_path.exists():
            parser.error(
                f"VecNormalize file does not exist: {args.vecnormalize_path}"
            )

    recovery_reward = (
        RewardConfig(
            service_backlog_weight=args.service_backlog_weight,
            batch_completion_weight=args.batch_completion_weight,
            evaluation_service_backlog_weight=1000.0,
            evaluation_batch_completion_weight=1000.0,
            reward_scale=1e-4,
            subtract_idle_cost=args.subtract_idle_cost,
            urgency_potential_weight=args.urgency_potential_weight,
        )
        if recovery_enabled
        else None
    )
    deadline_bucket_edges = (
        (1, 3, 6, 12, 24) if recovery_enabled else None
    )
    safety_config = (
        SafetyConfig(
            service_envelope_total=args.service_envelope_total,
            batch_arrival_envelope_total=(
                args.batch_arrival_envelope_total
            ),
            future_fleet_capacity_total=(
                args.future_fleet_capacity_total
            ),
            negative_demand_flush=args.negative_demand_flush,
        )
        if args.v4_safety
        else None
    )

    scenario_name = args.scenario.stem
    if args.batch_mode:
        scenario_name += "_batch"
    if args.demand_charge_rate > 0.0:
        scenario_name += "_demand_charge"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    # Load primary trained model
    print(f"Loading PPO model from {args.model}...")
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
        allow_multiple_demand_charge_periods=(
            args.allow_multiple_demand_charge_periods
        ),
        enforce_batch_completion=args.enforce_batch_completion,
        completion_penalty_weight=args.completion_penalty,
        batch_spatial_routing=args.batch_spatial_routing,
        reward_config=recovery_reward,
        observe_episode_progress=recovery_enabled,
        deadline_bucket_edges=deadline_bucket_edges,
        safety_config=safety_config,
    )
    if args.vecnormalize_path is not None:
        rl_reward, rl_history = run_normalized_episode(
            env,
            rl_model,
            args.vecnormalize_path,
        )
    else:
        rl_reward, rl_history = run_episode(
            env,
            rl_model.predict,
            is_sb3=True,
        )
    rl_summary = compute_summary(rl_history, batch_enabled=args.batch_mode)
    print(f"  {rl_label} total cost: {rl_summary['total_cost']:.2f}")
    results[rl_label] = {"reward": rl_reward, "summary": rl_summary, "history": rl_history}

    ppo_summary = rl_summary  # for dc_names later

    baseline_classes = [StatusQuoPolicy, RoundRobinPolicy]
    if args.batch_mode:
        baseline_classes.append(DrainImmediatelyPolicy)
    for baseline_cls in baseline_classes:
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
            allow_multiple_demand_charge_periods=(
                args.allow_multiple_demand_charge_periods
            ),
            enforce_batch_completion=args.enforce_batch_completion,
            completion_penalty_weight=args.completion_penalty,
            batch_spatial_routing=args.batch_spatial_routing,
            reward_config=recovery_reward,
            observe_episode_progress=recovery_enabled,
            deadline_bucket_edges=deadline_bucket_edges,
            safety_config=safety_config,
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
    if rl_summary["demand_charge_enabled"]:
        report_lines.append(
            "Demand charge is included in total cost at "
            f"${rl_summary['demand_charge_rate']:g}/kW per "
            f"{rl_summary['demand_charge_period_steps']}-step billing period "
            f"({rl_summary['demand_charge_period_count']} period(s)). "
            "Tariff-aware backlog/expiry floor: "
            f"${rl_summary['economic_penalty_floor']:,.2f}/unit; "
            f"RL reward scale: {rl_summary['reward_scale']:g}.\n"
        )
        charge_heading = "Demand Charge (In Reward)"
    else:
        report_lines.append(
            "Demand charge is not included in total cost. The table reports "
            f"the full-episode reference charge at "
            f"${REFERENCE_DEMAND_CHARGE_RATE:g}/kW per billing cycle.\n"
        )
        charge_heading = "Reference Demand Charge"

    def demand_charge_for_report(summary: dict[str, Any]) -> float:
        if summary["demand_charge_enabled"]:
            return float(summary["total_demand_charge"])
        return float(summary["demand_charge_ref"])

    if args.batch_mode:
        report_lines.extend(
            [
                f"| Policy | Total Cost | Energy Cost | Peak Penalty | Full-Cycle Billed Peak (MW) | {charge_heading} | ND-weighted Load | Batch Complete | Batch Expired | Unfinished Batch Cost | Avg Pool Size |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name, data in results.items():
            s = data["summary"]
            report_lines.append(
                f"| {name} | {s['total_cost']:.2f} | "
                f"{s.get('total_energy_cost', 0):.2f} | "
                f"{s.get('total_peak_penalty', 0):.2f} | "
                f"{s.get('billed_peak_sum_mw', 0):.2f} | "
                f"{demand_charge_for_report(s):.2f} | "
                f"{s.get('nd_weighted_load', 0):.0f} | "
                f"{s.get('batch_completion_fraction', 0):.4f} | "
                f"{s.get('total_batch_expired', 0):.4f} | "
                f"{s.get('total_unfinished_batch_cost', 0):.2f} | "
                f"{s.get('avg_batch_pool_size', 0):.4f} |"
            )
    else:
        report_lines.extend(
            [
                f"| Policy | Total Cost | Energy Cost | Peak Penalty | Full-Cycle Billed Peak (MW) | {charge_heading} | ND-weighted Load | Total Grid (MW-steps) |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name, data in results.items():
            s = data["summary"]
            report_lines.append(
                f"| {name} | {s['total_cost']:.2f} | "
                f"{s.get('total_energy_cost', 0):.2f} | "
                f"{s.get('total_peak_penalty', 0):.2f} | "
                f"{s.get('billed_peak_sum_mw', 0):.2f} | "
                f"{demand_charge_for_report(s):.2f} | "
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
            linewidth=1.5 if name == "PPO" else 0.8,
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
