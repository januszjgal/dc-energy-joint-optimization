"""Evaluation-only policies and complete v6 metric reporting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from env.ramp_v6.environment import RampAwareEnv


@dataclass(frozen=True)
class CausalPolicyContext:
    """Current workload state exposed to hand-written evaluation comparators."""

    n_sites: int
    service_arrivals: np.ndarray
    queued_batch_by_origin: np.ndarray
    action_bound: float


Policy = Callable[[np.ndarray, CausalPolicyContext], np.ndarray]


def status_quo_policy(
    _: np.ndarray, context: CausalPolicyContext
) -> np.ndarray:
    """Route current service/batch in origin proportions and drain immediately."""
    service_logits = _bounded_logits(
        context.service_arrivals, context.action_bound
    )
    batch_logits = _bounded_logits(
        context.queued_batch_by_origin, context.action_bound
    )
    return np.concatenate(
        [
            service_logits,
            np.asarray([context.action_bound]),
            batch_logits,
        ]
    )


def flat_preference_policy(
    _: np.ndarray, context: CausalPolicyContext
) -> np.ndarray:
    """Deterministic pure preference smoke policy, not an analytic controller."""
    return np.zeros(2 * context.n_sites + 1, dtype=np.float64)


def run_episode(
    env: RampAwareEnv,
    policy: Policy,
) -> tuple[float, list[dict[str, Any]]]:
    observation, _ = env.reset()
    total_reward = 0.0
    history: list[dict[str, Any]] = []
    while True:
        context = CausalPolicyContext(
            n_sites=env.n_sites,
            service_arrivals=env.workload.service_arrivals[env._step].copy(),
            queued_batch_by_origin=env.queue.by_origin(env.n_sites).copy(),
            action_bound=float(env.action_space.high[0]),
        )
        action = policy(observation, context)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        history.append(info)
        if terminated or truncated:
            break
    return total_reward, history


def _bounded_logits(weights: np.ndarray, bound: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    logits = np.log(np.maximum(weights, np.finfo(np.float64).tiny))
    logits -= float(np.max(logits))
    return np.clip(logits, -bound, bound)


def summarize(
    history: list[dict[str, Any]],
    *,
    total_reward: float,
) -> dict[str, Any]:
    if not history:
        raise ValueError("history cannot be empty")
    markets = sorted(history[0]["per_market"])
    per_market: dict[str, Any] = {}
    all_forecast_errors: list[float] = []
    all_forecast_impacts: list[float] = []
    for market in markets:
        market_result: dict[str, Any] = {}
        for horizon in (1, 3):
            windows = [
                item["per_market"][market]["windows"][f"{horizon}h"]
                for item in history
            ]
            native_mw = np.asarray(
                [window["native_ramp_mw"] for window in windows]
            )
            adjusted_mw = np.asarray(
                [window["adjusted_ramp_mw"] for window in windows]
            )
            native_fraction = np.asarray(
                [window["native_fraction_s_per_hour"] for window in windows]
            )
            adjusted_fraction = np.asarray(
                [window["adjusted_fraction_s_per_hour"] for window in windows]
            )
            impacts = np.asarray(
                [window["incremental_squared_impact"] for window in windows]
            )
            tails = np.asarray(
                [window["incremental_tail_burden"] for window in windows]
            )
            market_result[f"{horizon}h"] = {
                "incremental_ramp_impact_sum": float(impacts.sum()),
                "incremental_ramp_impact_mean": float(impacts.mean()),
                "native_max_up_mw": float(np.max(native_mw)),
                "adjusted_max_up_mw": float(np.max(adjusted_mw)),
                "native_p95_abs_mw": float(np.quantile(np.abs(native_mw), 0.95)),
                "adjusted_p95_abs_mw": float(
                    np.quantile(np.abs(adjusted_mw), 0.95)
                ),
                "native_max_up_fraction_s_per_hour": float(
                    np.max(native_fraction)
                ),
                "adjusted_max_up_fraction_s_per_hour": float(
                    np.max(adjusted_fraction)
                ),
                "native_p95_abs_fraction_s_per_hour": float(
                    np.quantile(np.abs(native_fraction), 0.95)
                ),
                "adjusted_p95_abs_fraction_s_per_hour": float(
                    np.quantile(np.abs(adjusted_fraction), 0.95)
                ),
                "incremental_high_tail_burden_sum": float(tails.sum()),
            }
        market_rows = [item["per_market"][market] for item in history]
        market_result.update(
            {
                "da_energy_cost_usd": float(
                    sum(row["da_energy_cost_usd"] for row in market_rows)
                ),
                "power_over_frozen_s_mean": float(
                    np.mean([row["power_over_frozen_s"] for row in market_rows])
                ),
                "power_over_gross_demand_mean": float(
                    np.mean(
                        [row["power_over_gross_demand"] for row in market_rows]
                    )
                ),
                "power_over_market_scale_mean": float(
                    np.mean(
                        [row["power_over_market_scale"] for row in market_rows]
                    )
                ),
            }
        )
        for item in history:
            error = item["per_market"][market]["forecast_errors"].get(
                "net_h1_abs_error_mw"
            )
            if error is not None:
                all_forecast_errors.append(float(error))
                all_forecast_impacts.append(
                    float(
                        item["per_market"][market]["windows"]["1h"][
                            "incremental_squared_impact"
                        ]
                    )
                )
        per_market[market] = market_result

    forecast_strata = _forecast_error_strata(
        np.asarray(all_forecast_errors), np.asarray(all_forecast_impacts)
    )
    total_service = float(sum(item["service_completed"] for item in history))
    total_service_arrived = float(sum(item["service_arrived"] for item in history))
    total_batch = float(sum(item["batch_completed"] for item in history))
    total_batch_arrived = float(sum(item["batch_arrived"] for item in history))
    min_slack = min(
        min(item["capacity_slack"]) for item in history
    )
    terminal = history[-1]["batch_queue"]
    return {
        "protocol_id": history[0]["protocol_id"],
        "total_scalar_reward": total_reward,
        "macro_incremental_ramp_impact_mean": float(
            np.mean(
                [
                    item["weighted_incremental_ramp_impact"]
                    for item in history
                ]
            )
        ),
        "macro_incremental_high_tail_burden_mean": float(
            np.mean(
                [
                    item["weighted_incremental_tail_burden"]
                    for item in history
                ]
            )
        ),
        "da_energy_cost_usd": float(
            sum(item["da_energy_cost_usd"] for item in history)
        ),
        "workload_safety": {
            "service_completed": total_service,
            "service_arrived": total_service_arrived,
            "service_conservation_error": total_service_arrived - total_service,
            "batch_completed": total_batch,
            "batch_arrived": total_batch_arrived,
            "terminal_batch_queue": terminal["queued"],
            "batch_conservation_error": terminal["conservation_error"],
            "deadline_missed_work": float(
                sum(item["deadline_missed_work"] for item in history)
            ),
            "minimum_capacity_slack": min_slack,
            "terminal_tail_new_arrivals": float(
                sum(
                    item["new_arrivals"]
                    for item in history
                    if item["terminal_tail_active"]
                )
            ),
        },
        "per_market": per_market,
        "forecast_error_strata": forecast_strata,
    }


def no_proxy_summary(controlled: dict[str, Any]) -> dict[str, Any]:
    """Construct the explicit P=0 comparator from native metrics."""
    per_market: dict[str, Any] = {}
    for market, result in controlled["per_market"].items():
        per_market[market] = {
            f"{horizon}h": {
                "native_max_up_mw": result[f"{horizon}h"]["native_max_up_mw"],
                "adjusted_max_up_mw": result[f"{horizon}h"]["native_max_up_mw"],
                "native_p95_abs_mw": result[f"{horizon}h"]["native_p95_abs_mw"],
                "adjusted_p95_abs_mw": result[f"{horizon}h"][
                    "native_p95_abs_mw"
                ],
                "incremental_ramp_impact_sum": 0.0,
                "incremental_high_tail_burden_sum": 0.0,
            }
            for horizon in (1, 3)
        }
    return {
        "definition": "No data-center proxy: P_t=0 in every market-hour.",
        "macro_incremental_ramp_impact_mean": 0.0,
        "macro_incremental_high_tail_burden_mean": 0.0,
        "da_energy_cost_usd": 0.0,
        "per_market": per_market,
    }


def compare_to_status_quo(
    candidate: dict[str, Any],
    status_quo: dict[str, Any],
    cost_budget_fraction: float,
) -> dict[str, Any]:
    baseline_cost = float(status_quo["da_energy_cost_usd"])
    candidate_cost = float(candidate["da_energy_cost_usd"])
    allowed_increase = cost_budget_fraction * abs(baseline_cost)
    limit = baseline_cost + allowed_increase
    return {
        "status_quo_da_cost_usd": baseline_cost,
        "candidate_da_cost_usd": candidate_cost,
        "cost_delta_fraction": (
            (candidate_cost - baseline_cost) / abs(baseline_cost)
            if baseline_cost != 0.0
            else None
        ),
        "cost_budget_fraction": cost_budget_fraction,
        "cost_budget_limit_usd": limit,
        "cost_budget_compliant": (
            candidate_cost - baseline_cost <= allowed_increase + 1e-9
        ),
        "ramp_impact_delta_vs_status_quo": (
            candidate["macro_incremental_ramp_impact_mean"]
            - status_quo["macro_incremental_ramp_impact_mean"]
        ),
        "pareto_improves_ramp_without_higher_cost": (
            candidate["macro_incremental_ramp_impact_mean"]
            <= status_quo["macro_incremental_ramp_impact_mean"]
            and candidate_cost <= baseline_cost
        ),
    }


def _forecast_error_strata(
    errors: np.ndarray, impacts: np.ndarray
) -> dict[str, Any]:
    if len(errors) == 0:
        return {}
    q1, q2 = np.quantile(errors, [1.0 / 3.0, 2.0 / 3.0])
    result: dict[str, Any] = {}
    masks = {
        "low": errors <= q1,
        "medium": (errors > q1) & (errors <= q2),
        "high": errors > q2,
    }
    for name, mask in masks.items():
        result[name] = {
            "count": int(mask.sum()),
            "mean_abs_forecast_error_mw": (
                float(np.mean(errors[mask])) if mask.any() else None
            ),
            "mean_incremental_1h_ramp_impact": (
                float(np.mean(impacts[mask])) if mask.any() else None
            ),
        }
    return result
