"""Deterministic validation evaluation for policy and status quo.

Each validation window is one continuous episode (a calendar month). Both the
policy and the status-quo comparator run the same window with the same
arrivals, power models, grid trajectory, and forecasts. Daily figures are
produced afterwards by grouping the hourly results by UTC date.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.ramp_v6.projection import (
    SEMANTIC_ADJUSTMENT_COORDINATE_ID,
    SEMANTIC_ADJUSTMENT_UNITS,
)
from env.ramp_v6.objective import JointObjective
from ramp_rl.contract import (
    EnvRequest, RampEnvAdapter, RampEnvironmentFactory, environment_identity,
)


IMPROVEMENT_DEFINITION = (
    "Mean monthly J(status quo) - J(policy). J combines summed regional "
    "incremental squared ramp impacts and regional monthly net-load peaks using fixed positive "
    "training references and the declared weights. Positive favors the policy; "
    "neither component is guaranteed to improve individually."
)


def _bootstrap_interval(
    values: list[float], groups: list[str], *, draws: int = 500, seed: int = 20260808
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        grouped[group].append(float(value))
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    samples = [
        mean(value for key in rng.choice(keys, size=len(keys), replace=True) for value in grouped[str(key)])
        for _ in range(draws)
    ]
    return {"mean": float(mean(values)), "lower_95": float(np.quantile(samples, .025)),
            "upper_95": float(np.quantile(samples, .975))}


def _cluster_bootstrap_interval(
    values: list[float],
    groups: list[str],
    *,
    unit: str,
    draws: int,
) -> dict[str, Any]:
    group_count = len(set(groups))
    if group_count < 2:
        return {
            "estimable": False,
            "group_count": group_count,
            "mean": float(mean(values)),
            "unit": unit,
            "reason": "at least two independent groups are required",
        }
    interval = _bootstrap_interval(values, groups, draws=draws)
    interval.update(
        {
            "estimable": True,
            "group_count": group_count,
            "unit": unit,
        }
    )
    return interval


def _collect_per_market_metrics(
    infos: list[dict[str, Any]],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    incremental: dict[str, list[float]] = defaultdict(list)
    abs_h1: dict[str, list[float]] = defaultdict(list)
    for info in infos:
        per_market = info.get("per_market", {})
        if not per_market:
            raise ValueError("evaluation requires per-market step metrics")
        for market, row in per_market.items():
            window = row["windows"]["1h"]
            incremental[str(market)].append(float(window["incremental_squared_impact"]))
            abs_h1[str(market)].append(abs(float(window["adjusted_fraction_s_per_hour"])))
    return dict(incremental), dict(abs_h1)


def _daily_means(dates: list[str], values: list[float]) -> "OrderedDict[str, float]":
    buckets: "OrderedDict[str, list[float]]" = OrderedDict()
    for date, value in zip(dates, values):
        buckets.setdefault(date, []).append(float(value))
    return OrderedDict((date, float(mean(bucket))) for date, bucket in buckets.items())


def _run_episode(
    *,
    factory: RampEnvironmentFactory,
    split: str,
    seed: int,
    window_id: str,
    model: PPO | None,
    normalization_path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request = EnvRequest(split=split, seed=seed, window_id=window_id, training=False)
    infos: list[dict[str, Any]] = []
    if model is None:
        adapter = RampEnvAdapter(factory(request), request)
        try:
            _, reset_info = adapter.reset(seed=seed)
            while True:
                action = adapter.evaluation_action("status_quo")
                _, _, terminated, _, info = adapter.step(action)
                infos.append(info)
                if terminated:
                    break
        finally:
            adapter.close()
        return infos, reset_info

    def make() -> RampEnvAdapter:
        local_request = EnvRequest(split=split, seed=seed, window_id=window_id, training=False)
        return RampEnvAdapter(factory(local_request), local_request)

    base_vec = DummyVecEnv([make])
    vec = VecNormalize.load(normalization_path, base_vec)
    vec.training = False
    vec.norm_reward = False
    try:
        observation = vec.reset()
        reset_info = base_vec.reset_infos[0]
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, dones, step_infos = vec.step(action)
            infos.append(step_infos[0])
            if bool(dones[0]):
                break
    finally:
        vec.close()
    return infos, reset_info


def _episode(
    *,
    factory: RampEnvironmentFactory,
    split: str,
    seed: int,
    window_id: str,
    model: PPO | None,
    normalization_path: Path | None,
) -> dict[str, Any]:
    infos, reset_info = _run_episode(
        factory=factory, split=split, seed=seed, window_id=window_id,
        model=model, normalization_path=normalization_path,
    )
    context = reset_info["episode_context"]
    per_market_incremental, abs_h1_by_market = _collect_per_market_metrics(infos)
    # The objective uses the monthly sum; hourly means remain diagnostics.
    incremental = [float(info["incremental_ramp_impact"]) for info in infos]
    ramp_impact_sum = float(sum(incremental))
    dates = [str(info["utc_date"]) for info in infos]
    daily = _daily_means(dates, incremental)
    semantic_adjustment_values = [float(info["semantic_adjustment_l2"]) for info in infos]
    objective = JointObjective.from_dict(context["objective"])
    ramp_mean_squared = float(mean(float(info["ramp_squared_score"]) for info in infos))
    peaks = {
        market: {
            "adjusted_peak_mw": float(row["running_adjusted_peak_mw"]),
            "native_peak_mw": float(row["running_native_peak_mw"]),
            "normalized_peak": float(row["running_adjusted_peak_mw"]) / float(row["market_scale_mw"]),
            "peak_impact": (
                float(row["running_adjusted_peak_mw"]) - float(row["running_native_peak_mw"])
            ) / float(row["market_scale_mw"]),
        }
        for market, row in infos[-1]["per_market"].items()
    }
    normalized_peak = sum(row["normalized_peak"] for row in peaks.values())
    peak_impact_sum = sum(row["peak_impact"] for row in peaks.values())
    joint_J = objective.score(ramp_impact_sum, peak_impact_sum)
    raw_return = float(sum(info["scalar_reward"] for info in infos))
    if not np.isclose(raw_return, -joint_J, rtol=1e-9, atol=1e-12):
        raise RuntimeError("raw monthly return does not equal the negative joint objective")
    return {
        "window_id": window_id,
        "evaluation_seed": seed,
        "month_id": str(context.get("month_id", window_id)),
        "future_realized_features_exposed": bool(context["future_realized_features_exposed"]),
        "step_count": len(infos),
        "day_count": len(daily),
        "utc_dates": dates,
        "ramp_h1": [float(info["ramp_h1_adjusted"]) for info in infos],
        "incremental": incremental,
        "objective": objective.as_dict(),
        "J": joint_J,
        "raw_undiscounted_return": raw_return,
        "ramp_mean_squared": ramp_mean_squared,
        "ramp_impact_sum": ramp_impact_sum,
        "normalized_peak": normalized_peak,
        "peak_impact_sum": peak_impact_sum,
        "regional_peaks": peaks,
        "per_market_ramp_mean_squared": {
            market: float(mean(
                float(info["per_market"][market]["windows"]["1h"]["adjusted_fraction_s_per_hour"])**2
                for info in infos
            ))
            for market in peaks
        },
        "per_market_ramp_impact_sum": {
            market: float(sum(values))
            for market, values in per_market_incremental.items()
        },
        "ramp_impact_J": ramp_impact_sum,
        "daily_incremental": dict(daily),
        "per_market_incremental": per_market_incremental,
        "abs_adjusted_ramp_h1_by_market": abs_h1_by_market,
        "service_unserved": sum(float(info["service_unserved"]) for info in infos),
        "batch_unfinished": float(infos[-1]["batch_unfinished"]),
        "batch_expired": sum(float(info["batch_expired"]) for info in infos),
        "certificate_violations": sum(int(info["certificate_violations"]) for info in infos),
        "terminal_work": float(infos[-1]["terminal_work"]),
        "emergency_count": sum(bool(info["emergency_feasibility"]) for info in infos),
        "semantic_adjustment_l2_values": semantic_adjustment_values,
        "semantic_adjustment_l2": sum(semantic_adjustment_values),
        "deferrable_pre_service": sum(float(info["deferrable_pre_service"]) for info in infos),
        "ramp_power": sum(float(info["dc_power_during_realized_ramp"]) for info in infos),
    }


def _market_series(episodes: list[dict[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        for market, episode_values in episode["per_market_incremental"].items():
            values[market].extend(episode_values)
    return values


def _daily_table(policy: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Daily ramp-impact diagnostics, not daily or monthly joint objectives."""
    policy_days: dict[str, list[float]] = defaultdict(list)
    baseline_days: dict[str, list[float]] = defaultdict(list)
    for episode in policy:
        for date, value in episode["daily_incremental"].items():
            policy_days[date].append(float(value))
    for episode in baseline:
        for date, value in episode["daily_incremental"].items():
            baseline_days[date].append(float(value))
    if set(policy_days) != set(baseline_days):
        raise ValueError("policy and status quo episodes cover different days")
    rows = []
    for date in sorted(policy_days):
        policy_mean = float(mean(policy_days[date]))
        baseline_mean = float(mean(baseline_days[date]))
        rows.append({
            "date": date,
            "metric": "mean_hourly_sum_of_regional_incremental_ramp_impacts",
            "policy_incremental_ramp_impact": policy_mean,
            "status_quo_incremental_ramp_impact": baseline_mean,
            "ramp_impact_improvement_mean_per_hour": baseline_mean - policy_mean,
        })
    return rows


def _aggregate(
    policy: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    if not policy or len(policy) != len(baseline):
        raise ValueError("policy and status quo episodes must be non-empty and aligned")
    for policy_episode, baseline_episode in zip(policy, baseline):
        if policy_episode["window_id"] != baseline_episode["window_id"]:
            raise ValueError("policy and status quo episodes must be paired by window")
        if policy_episode["utc_dates"] != baseline_episode["utc_dates"]:
            raise ValueError("policy and status quo episodes must cover the same hours")
        if policy_episode["objective"] != baseline_episode["objective"]:
            raise ValueError("policy and status quo objective definitions differ")
    if any(episode["objective"] != policy[0]["objective"] for episode in policy):
        raise ValueError("cannot aggregate different objective weights or references")
    objective = JointObjective.from_dict(policy[0]["objective"])
    impacts = [value for episode in policy for value in episode["incremental"]]
    impact_dates = [date for episode in policy for date in episode["utc_dates"]]
    impact_months = [episode["month_id"] for episode in policy for _ in episode["incremental"]]
    abs_h1 = [
        value
        for episode in policy
        for values in episode["abs_adjusted_ramp_h1_by_market"].values()
        for value in values
    ]
    if not impacts or not abs_h1:
        raise ValueError("evaluation episodes are missing ramp metric observations")
    policy_market_values = _market_series(policy)
    status_quo_market_values = _market_series(baseline)
    if set(policy_market_values) != set(status_quo_market_values):
        raise ValueError("policy and status quo market sets differ")
    policy_market_macro = {key: float(mean(values)) for key, values in policy_market_values.items()}
    status_quo_market_macro = {
        key: float(mean(status_quo_market_values[key])) for key in policy_market_values
    }
    per_market_status_quo_comparison = {
        market: {
            "policy_native_relative_incremental_ramp_impact": policy_market_macro[market],
            "status_quo_native_relative_incremental_ramp_impact": status_quo_market_macro[market],
            "policy_minus_status_quo_incremental_ramp_impact": (
                policy_market_macro[market] - status_quo_market_macro[market]
            ),
            "ramp_impact_improvement_mean_per_hour": status_quo_market_macro[market] - policy_market_macro[market],
        }
        for market in policy_market_values
    }
    for market, row in per_market_status_quo_comparison.items():
        policy_ramp = float(mean(e["per_market_ramp_mean_squared"][market] for e in policy))
        baseline_ramp = float(mean(e["per_market_ramp_mean_squared"][market] for e in baseline))
        policy_impact = float(mean(e["per_market_ramp_impact_sum"][market] for e in policy))
        baseline_impact = float(mean(e["per_market_ramp_impact_sum"][market] for e in baseline))
        policy_peak = float(mean(e["regional_peaks"][market]["peak_impact"] for e in policy))
        baseline_peak = float(mean(e["regional_peaks"][market]["peak_impact"] for e in baseline))
        policy_peak_mw = float(mean(e["regional_peaks"][market]["adjusted_peak_mw"] for e in policy))
        baseline_peak_mw = float(mean(e["regional_peaks"][market]["adjusted_peak_mw"] for e in baseline))
        policy_J = objective.score(policy_impact, policy_peak)
        baseline_J = objective.score(baseline_impact, baseline_peak)
        row.update({
            "policy_ramp_impact_sum": policy_impact,
            "status_quo_ramp_impact_sum": baseline_impact,
            "ramp_impact_improvement_sum": baseline_impact - policy_impact,
            "policy_ramp_mean_squared": policy_ramp,
            "status_quo_ramp_mean_squared": baseline_ramp,
            "ramp_mean_squared_improvement": baseline_ramp - policy_ramp,
            "policy_peak_mw": policy_peak_mw,
            "status_quo_peak_mw": baseline_peak_mw,
            "peak_reduction_mw": baseline_peak_mw - policy_peak_mw,
            "native_peak_mw": float(mean(e["regional_peaks"][market]["native_peak_mw"] for e in policy)),
            "policy_peak_impact": policy_peak,
            "status_quo_peak_impact": baseline_peak,
            "policy_joint_J": policy_J,
            "status_quo_joint_J": baseline_J,
            "improvement": baseline_J - policy_J,
            "policy_outperforms_status_quo": policy_J < baseline_J,
        })
    better_count = sum(row["policy_outperforms_status_quo"] for row in per_market_status_quo_comparison.values())
    market_count = len(per_market_status_quo_comparison)
    if market_count == 0:
        raise ValueError("evaluation contains no market-level impacts")
    status_quo_impacts = [value for episode in baseline for value in episode["incremental"]]
    semantic_adjustment_values = [
        value for episode in policy for value in episode["semantic_adjustment_l2_values"]
    ]
    if not semantic_adjustment_values:
        raise ValueError("evaluation contains no decoder adjustments")
    semantic_positive_count = sum(value > 1e-12 for value in semantic_adjustment_values)
    per_evaluation_seed: dict[int, list[float]] = defaultdict(list)
    for episode in policy:
        per_evaluation_seed[int(episode["evaluation_seed"])].append(episode["J"])
    policy_mean = float(mean(impacts))
    status_quo_mean = float(mean(status_quo_impacts))
    step_count = sum(episode["step_count"] for episode in policy)
    policy_J = float(mean(episode["J"] for episode in policy))
    baseline_J = float(mean(episode["J"] for episode in baseline))
    components = {}
    for name, field, reference, weight in (
        ("ramp", "ramp_impact_sum", objective.ramp_reference, objective.ramp_weight),
        ("net_load_peak", "peak_impact_sum", objective.peak_reference, objective.peak_weight),
    ):
        policy_value = float(mean(episode[field] for episode in policy))
        baseline_value = float(mean(episode[field] for episode in baseline))
        components[name] = {
            "metric": field,
            "policy": policy_value,
            "status_quo": baseline_value,
            "improvement": baseline_value - policy_value,
            "relative_improvement": (
                (baseline_value - policy_value) / baseline_value
                if name == "net_load_peak" and baseline_value > 0.0 else None
            ),
            "relative_improvement_basis": (
                "positive_baseline_peak_impact"
                if name == "net_load_peak" and baseline_value > 0.0
                else "not_applicable_to_signed_or_nonpositive_metric"
            ),
            "training_reference": reference,
            "weight": weight,
            "weighted_improvement": weight * (baseline_value - policy_value) / reference,
        }
    summary = {
        "split": split,
        "objective": objective.as_dict(),
        "mean_joint_J": policy_J,
        "mean_monthly_ramp_impact": components["ramp"]["policy"],
        "mean_ramp_squared": float(mean(episode["ramp_mean_squared"] for episode in policy)),
        "mean_peak_impact": components["net_load_peak"]["policy"],
        "mean_normalized_net_load_peak": float(mean(episode["normalized_peak"] for episode in policy)),
        "episode_count": len(policy),
        "window_ids": [episode["window_id"] for episode in policy],
        "step_count": step_count,
        "day_count": sum(episode["day_count"] for episode in policy),
        "service_unserved": sum(episode["service_unserved"] for episode in policy),
        "batch_unfinished": sum(episode["batch_unfinished"] for episode in policy),
        "batch_expired": sum(episode["batch_expired"] for episode in policy),
        "certificate_violations": sum(episode["certificate_violations"] for episode in policy),
        "terminal_work": sum(episode["terminal_work"] for episode in policy),
        "mean_policy_native_relative_incremental_ramp_impact": policy_mean,
        "per_market_policy_native_relative_incremental_ramp_impact": policy_market_macro,
        "mean_incremental_ramp_impact": policy_mean,
        "per_market_macro": policy_market_macro,
        "physical_ramp_metric_contract": {
            "input": "absolute adjusted ramp magnitude",
            "pooling": "all market-timestep values with equal weight",
            "cross_market_signed_averaging": False,
            "units": "fraction_of_market_training_q95_gross_demand_per_hour",
        },
        "abs_adjusted_ramp_h1_fraction_s_per_hour_p95": float(np.quantile(abs_h1, 0.95)),
        "abs_adjusted_ramp_h1_fraction_s_per_hour_max": float(max(abs_h1)),
        "ramp_h1_adjusted_p95": float(np.quantile(abs_h1, 0.95)),
        "ramp_h1_adjusted_max": float(max(abs_h1)),
        "status_quo_comparison": {
            "baseline_definition": (
                "status quo: every service and batch arrival executes at its "
                "origin site in its arrival hour; no deferral, no routing"
            ),
            "policy_native_relative_mean_incremental_ramp_impact": policy_mean,
            "status_quo_native_relative_mean_incremental_ramp_impact": status_quo_mean,
            "policy_minus_status_quo_mean_incremental_ramp_impact": policy_mean - status_quo_mean,
            "ramp_impact_improvement_mean_per_hour": status_quo_mean - policy_mean,
            "ramp_impact_improvement_J": float(
                sum(episode["ramp_impact_J"] for episode in baseline)
                - sum(episode["ramp_impact_J"] for episode in policy)
            ),
            "improvement": baseline_J - policy_J,
            "policy_J": policy_J,
            "status_quo_J": baseline_J,
            "improvement_J": baseline_J - policy_J,
            "policy_minus_status_quo_joint_J": policy_J - baseline_J,
            "components": components,
            "improvement_definition": IMPROVEMENT_DEFINITION,
            "markets_better_count": int(better_count),
            "market_count": int(market_count),
            "markets_better_share": float(better_count / market_count),
            "every_market_outperforms_status_quo": better_count == market_count,
            "per_market": per_market_status_quo_comparison,
        },
        "daily": _daily_table(policy, baseline),
        "emergency_feasibility_rate": float(
            sum(episode["emergency_count"] for episode in policy) / step_count
        ),
        "semantic_adjustment_l2": float(sum(semantic_adjustment_values)),
        "semantic_adjustment": {
            "coordinate_id": SEMANTIC_ADJUSTMENT_COORDINATE_ID,
            "units": SEMANTIC_ADJUSTMENT_UNITS,
            "sum_l2": float(sum(semantic_adjustment_values)),
            "mean_l2": float(mean(semantic_adjustment_values)),
            "p50_l2": float(np.quantile(semantic_adjustment_values, 0.50)),
            "p95_l2": float(np.quantile(semantic_adjustment_values, 0.95)),
            "max_l2": float(max(semantic_adjustment_values)),
            "positive_adjustment_count": int(semantic_positive_count),
            "decision_count": len(semantic_adjustment_values),
            "adjustment_rate": float(semantic_positive_count / len(semantic_adjustment_values)),
            "emergency_fallback_rate": float(
                sum(episode["emergency_count"] for episode in policy) / step_count
            ),
        },
        "future_leakage_detected": any(episode["future_realized_features_exposed"] for episode in policy),
        "behavior_audit": {
            "deferrable_pre_service": sum(episode["deferrable_pre_service"] for episode in policy),
            "policy_ramp_power": sum(episode["ramp_power"] for episode in policy),
            "status_quo_ramp_power": sum(episode["ramp_power"] for episode in baseline),
        },
        "ramp_impact_bootstrap_by_month": _cluster_bootstrap_interval(
            impacts, impact_months, unit="month", draws=500,
        ),
        "ramp_impact_bootstrap_by_day": _cluster_bootstrap_interval(
            impacts, impact_dates, unit="day", draws=500,
        ),
        "evaluation_seed_interval": {
            "metric": "monthly_joint_J",
            "mean": float(mean(float(mean(values)) for values in per_evaluation_seed.values())),
            "minimum": float(min(float(mean(values)) for values in per_evaluation_seed.values())),
            "maximum": float(max(float(mean(values)) for values in per_evaluation_seed.values())),
            "seed_count": len(per_evaluation_seed),
        },
        "policy_episodes": policy,
        "status_quo_episodes": baseline,
    }
    return summary


def evaluate_checkpoint(
    *,
    factory: RampEnvironmentFactory,
    checkpoint_dir: Path,
    split: str,
    seeds: list[int],
    windows: list[str],
) -> dict[str, Any]:
    if split != "validation":
        raise ValueError("evaluation is validation-only")
    if not seeds or not windows:
        raise ValueError("evaluation requires at least one seed and validation window")
    metadata_path = checkpoint_dir / "training_summary.json"
    if not metadata_path.is_file():
        metadata_path = checkpoint_dir / "state.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    request = EnvRequest(split=split, seed=seeds[0], window_id=windows[0])
    probe = RampEnvAdapter(factory(request), request)
    try:
        if metadata.get("environment_identity") != environment_identity(probe.contract):
            raise ValueError("checkpoint objective, data or observation schema does not match evaluation")
    finally:
        probe.close()
    model = PPO.load(checkpoint_dir / "model.zip", device="cpu")
    normalization_path = checkpoint_dir / "vecnormalize.pkl"
    policy: list[dict[str, Any]] = []
    baseline: list[dict[str, Any]] = []
    for seed in seeds:
        for window in windows:
            policy.append(
                _episode(
                    factory=factory, split=split, seed=seed, window_id=window,
                    model=model, normalization_path=normalization_path,
                )
            )
            baseline.append(
                _episode(
                    factory=factory, split=split, seed=seed, window_id=window,
                    model=None, normalization_path=None,
                )
            )
    return _aggregate(policy, baseline, split)
