"""Deterministic validation evaluation for policy and status quo."""

from __future__ import annotations

from collections import defaultdict
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
from ramp_rl.contract import EnvRequest, RampEnvAdapter, RampEnvironmentFactory


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


def _collect_per_market_metrics(
    infos: list[dict[str, Any]],
) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
]:
    incremental: dict[str, list[float]] = defaultdict(list)
    abs_h1: dict[str, list[float]] = defaultdict(list)
    abs_h3: dict[str, list[float]] = defaultdict(list)
    for info in infos:
        per_market = info.get("per_market", {})
        if per_market:
            for market, row in per_market.items():
                key = str(market)
                incremental[key].append(
                    sum(
                        weight
                        * float(
                            row["windows"][f"{horizon}h"][
                                "incremental_squared_impact"
                            ]
                        )
                        for horizon, weight in ((1, 0.4), (3, 0.6))
                    )
                )
                abs_h1[key].append(
                    abs(
                        float(
                            row["windows"]["1h"][
                                "adjusted_fraction_s_per_hour"
                            ]
                        )
                    )
                )
                abs_h3[key].append(
                    abs(
                        float(
                            row["windows"]["3h"][
                                "adjusted_fraction_s_per_hour"
                            ]
                        )
                    )
                )
        else:
            key = "__aggregate_fixture__"
            abs_h1[key].append(abs(float(info["ramp_h1_adjusted"])))
            abs_h3[key].append(abs(float(info["ramp_h3_adjusted"])))
    return dict(incremental), dict(abs_h1), dict(abs_h3)


def _market_series(
    episodes: list[dict[str, Any]],
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        if episode["per_market_incremental"]:
            for market, episode_values in episode[
                "per_market_incremental"
            ].items():
                values[market].extend(episode_values)
        else:
            values[f"fixture-month-{episode['month']:02d}"].extend(
                episode["incremental"]
            )
    return values


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


def _episode(
    *,
    factory: RampEnvironmentFactory,
    split: str,
    seed: int,
    window_id: str,
    model: PPO | None,
    normalization_path: Path | None,
) -> dict[str, Any]:
    request = EnvRequest(
        split=split,
        seed=seed,
        window_id=window_id,
        training=False,
    )
    adapter = RampEnvAdapter(factory(request), request)
    if model is None:
        tail_emitted_in_steps = bool(
            adapter.contract.get("terminal_tail_emitted_in_step_metrics", False)
        )
        observation, reset_info = adapter.reset(seed=seed)
        infos: list[dict[str, Any]] = []
        while True:
            action = adapter.evaluation_action("status_quo")
            observation, _, terminated, _, info = adapter.step(action)
            infos.append(info)
            if terminated:
                break
        adapter.close()
    else:
        adapter.close()

        def make() -> RampEnvAdapter:
            local_request = EnvRequest(
                split=split,
                seed=seed,
                window_id=window_id,
                training=False,
            )
            return RampEnvAdapter(factory(local_request), local_request)

        base_vec = DummyVecEnv([make])
        tail_emitted_in_steps = bool(
            base_vec.envs[0].contract.get(
                "terminal_tail_emitted_in_step_metrics", False
            )
        )
        vec = VecNormalize.load(normalization_path, base_vec)
        vec.training = False
        vec.norm_reward = False
        observation = vec.reset()
        reset_info = base_vec.reset_infos[0]
        infos = []
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, dones, step_infos = vec.step(action)
            infos.append(step_infos[0])
            if bool(dones[0]):
                break
        vec.close()
    tail_h1 = (
        []
        if tail_emitted_in_steps
        else infos[-1]["terminal_tail_ramp_h1_adjusted"]
    )
    tail_h3 = (
        []
        if tail_emitted_in_steps
        else infos[-1]["terminal_tail_ramp_h3_adjusted"]
    )
    tail_incremental = (
        []
        if tail_emitted_in_steps
        else infos[-1]["terminal_tail_incremental_ramp_impact"]
    )
    tail_energy = (
        [] if tail_emitted_in_steps else infos[-1]["terminal_tail_energy_cost"]
    )
    tail_status_quo_energy = (
        []
        if tail_emitted_in_steps
        else infos[-1]["terminal_tail_status_quo_energy_cost"]
    )
    tail_service_unserved = (
        []
        if tail_emitted_in_steps
        else infos[-1]["terminal_tail_service_unserved"]
    )
    tail_batch_unfinished = (
        []
        if tail_emitted_in_steps
        else infos[-1]["terminal_tail_batch_unfinished"]
    )
    tail_batch_expired = (
        []
        if tail_emitted_in_steps
        else infos[-1]["terminal_tail_batch_expired"]
    )
    tail_certificates = (
        []
        if tail_emitted_in_steps
        else infos[-1]["terminal_tail_certificate_violations"]
    )
    tail_emergency = (
        []
        if tail_emitted_in_steps
        else infos[-1]["terminal_tail_emergency_feasibility"]
    )
    tail_semantic_adjustment = (
        []
        if tail_emitted_in_steps
        else infos[-1]["terminal_tail_semantic_adjustment_l2"]
    )
    (
        per_market_incremental,
        abs_adjusted_h1_by_market,
        abs_adjusted_h3_by_market,
    ) = _collect_per_market_metrics(infos)
    if not tail_emitted_in_steps:
        abs_adjusted_h1_by_market.setdefault(
            "__aggregate_fixture__", []
        ).extend(abs(float(value)) for value in tail_h1)
        abs_adjusted_h3_by_market.setdefault(
            "__aggregate_fixture__", []
        ).extend(abs(float(value)) for value in tail_h3)
    semantic_adjustment_values = [
        float(info["semantic_adjustment_l2"]) for info in infos
    ] + [float(value) for value in tail_semantic_adjustment]
    return {
        "window_id": window_id,
        "evaluation_seed": seed,
        "month": int(reset_info["episode_context"]["month"]),
        "day_group": str(reset_info["episode_context"].get("day", window_id)),
        "future_realized_features_exposed": bool(
            reset_info["episode_context"][
                "future_realized_features_exposed"
            ]
        ),
        "ramp_h1": (
            [float(info["ramp_h1_adjusted"]) for info in infos]
            + [float(value) for value in tail_h1]
        ),
        "ramp_h3": (
            [float(info["ramp_h3_adjusted"]) for info in infos]
            + [float(value) for value in tail_h3]
        ),
        "incremental": (
            [float(info["incremental_ramp_impact"]) for info in infos]
            + [float(value) for value in tail_incremental]
        ),
        "per_market_incremental": dict(per_market_incremental),
        "abs_adjusted_ramp_h1_by_market": abs_adjusted_h1_by_market,
        "abs_adjusted_ramp_h3_by_market": abs_adjusted_h3_by_market,
        "energy_cost": sum(float(info["energy_cost"]) for info in infos)
        + sum(float(value) for value in tail_energy),
        "status_quo_energy_cost": sum(
            float(info["status_quo_energy_cost"]) for info in infos
        )
        + sum(float(value) for value in tail_status_quo_energy),
        "service_unserved": sum(float(info["service_unserved"]) for info in infos)
        + sum(float(value) for value in tail_service_unserved),
        "batch_unfinished": (
            float(tail_batch_unfinished[-1])
            if tail_batch_unfinished
            else float(infos[-1]["batch_unfinished"])
        ),
        "batch_expired": sum(float(info["batch_expired"]) for info in infos)
        + sum(float(value) for value in tail_batch_expired),
        "certificate_violations": sum(
            int(info["certificate_violations"]) for info in infos
        )
        + sum(int(value) for value in tail_certificates),
        "terminal_work": float(infos[-1]["terminal_work"]),
        "emergency_count": sum(
            bool(info["emergency_feasibility"]) for info in infos
        )
        + sum(bool(value) for value in tail_emergency),
        "step_count": len(infos) + len(tail_emergency),
        "semantic_adjustment_l2_values": semantic_adjustment_values,
        "semantic_adjustment_l2": sum(semantic_adjustment_values),
        "deferrable_pre_service": sum(
            float(info["deferrable_pre_service"]) for info in infos
        ),
        "ramp_power": sum(
            float(info["dc_power_during_realized_ramp"])
            for info in infos
        ),
    }


def _aggregate(
    policy: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    if not policy or len(policy) != len(baseline):
        raise ValueError(
            "policy and status quo episodes must be non-empty and aligned"
        )
    impacts = [value for episode in policy for value in episode["incremental"]]
    abs_h1 = [
        value
        for episode in policy
        for values in episode["abs_adjusted_ramp_h1_by_market"].values()
        for value in values
    ]
    abs_h3 = [
        value
        for episode in policy
        for values in episode["abs_adjusted_ramp_h3_by_market"].values()
        for value in values
    ]
    if not impacts or not abs_h1 or not abs_h3:
        raise ValueError(
            "evaluation episodes are missing ramp metric observations"
        )
    policy_cost = sum(episode["energy_cost"] for episode in policy)
    baseline_cost = sum(episode["energy_cost"] for episode in baseline)
    if baseline_cost == 0.0:
        raise ValueError("status quo energy cost must be non-zero")
    policy_market_values = _market_series(policy)
    status_quo_market_values = _market_series(baseline)
    if set(policy_market_values) != set(status_quo_market_values):
        raise ValueError("policy and status quo market sets differ")
    policy_market_macro = {
        key: float(mean(values))
        for key, values in policy_market_values.items()
    }
    status_quo_market_macro = {
        key: float(mean(status_quo_market_values[key]))
        for key in policy_market_values
    }
    per_market_status_quo_comparison = {
        market: {
            "policy_native_relative_incremental_ramp_impact": (
                policy_market_macro[market]
            ),
            "status_quo_native_relative_incremental_ramp_impact": (
                status_quo_market_macro[market]
            ),
            "policy_minus_status_quo_incremental_ramp_impact": (
                policy_market_macro[market]
                - status_quo_market_macro[market]
            ),
            "policy_outperforms_status_quo": (
                policy_market_macro[market]
                < status_quo_market_macro[market]
            ),
        }
        for market in policy_market_values
    }
    better_count = sum(
        row["policy_outperforms_status_quo"]
        for row in per_market_status_quo_comparison.values()
    )
    market_count = len(per_market_status_quo_comparison)
    if market_count == 0:
        raise ValueError("evaluation contains no market-level impacts")
    status_quo_impacts = [
        value
        for episode in baseline
        for value in episode["incremental"]
    ]
    semantic_adjustment_values = [
        value
        for episode in policy
        for value in episode.get(
            "semantic_adjustment_l2_values",
            [episode["semantic_adjustment_l2"]],
        )
    ]
    if not semantic_adjustment_values:
        raise ValueError("evaluation contains no decoder adjustments")
    semantic_positive_count = sum(
        value > 1e-12 for value in semantic_adjustment_values
    )
    per_evaluation_seed: dict[int, list[float]] = defaultdict(list)
    for episode in policy:
        per_evaluation_seed[int(episode["evaluation_seed"])].extend(episode["incremental"])
    summary = {
        "split": split,
        "episode_count": len(policy),
        "service_unserved": sum(episode["service_unserved"] for episode in policy),
        "batch_unfinished": sum(episode["batch_unfinished"] for episode in policy),
        "batch_expired": sum(episode["batch_expired"] for episode in policy),
        "certificate_violations": sum(episode["certificate_violations"] for episode in policy),
        "terminal_work": sum(episode["terminal_work"] for episode in policy),
        "mean_policy_native_relative_incremental_ramp_impact": float(
            mean(impacts)
        ),
        "per_market_policy_native_relative_incremental_ramp_impact": (
            policy_market_macro
        ),
        "mean_incremental_ramp_impact": float(mean(impacts)),
        "per_market_macro": policy_market_macro,
        "physical_ramp_metric_contract": {
            "input": "absolute adjusted ramp magnitude",
            "pooling": "all market-timestep values with equal weight",
            "cross_market_signed_averaging": False,
            "units": "fraction_of_market_training_q95_gross_demand_per_hour",
        },
        "abs_adjusted_ramp_h1_fraction_s_per_hour_p95": float(
            np.quantile(abs_h1, 0.95)
        ),
        "abs_adjusted_ramp_h1_fraction_s_per_hour_max": float(max(abs_h1)),
        "abs_adjusted_ramp_h3_fraction_s_per_hour_p95": float(
            np.quantile(abs_h3, 0.95)
        ),
        "abs_adjusted_ramp_h3_fraction_s_per_hour_max": float(max(abs_h3)),
        "ramp_h1_adjusted_p95": float(np.quantile(abs_h1, 0.95)),
        "ramp_h1_adjusted_max": float(max(abs_h1)),
        "ramp_h3_adjusted_p95": float(np.quantile(abs_h3, 0.95)),
        "ramp_h3_adjusted_max": float(max(abs_h3)),
        "status_quo_comparison": {
            "policy_native_relative_mean_incremental_ramp_impact": float(
                mean(impacts)
            ),
            "status_quo_native_relative_mean_incremental_ramp_impact": float(
                mean(status_quo_impacts)
            ),
            "policy_minus_status_quo_mean_incremental_ramp_impact": float(
                mean(impacts) - mean(status_quo_impacts)
            ),
            "markets_better_count": int(better_count),
            "market_count": int(market_count),
            "markets_better_share": float(better_count / market_count),
            "every_market_outperforms_status_quo": (
                better_count == market_count
            ),
            "per_market": per_market_status_quo_comparison,
        },
        "energy_cost_ratio": float(policy_cost / baseline_cost),
        "emergency_feasibility_rate": float(
            sum(episode["emergency_count"] for episode in policy)
            / sum(episode["step_count"] for episode in policy)
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
            "adjustment_rate": float(
                semantic_positive_count / len(semantic_adjustment_values)
            ),
            "emergency_fallback_rate": float(
                sum(episode["emergency_count"] for episode in policy)
                / sum(episode["step_count"] for episode in policy)
            ),
        },
        "future_leakage_detected": any(episode["future_realized_features_exposed"] for episode in policy),
        "behavior_audit": {
            "deferrable_pre_service": sum(episode["deferrable_pre_service"] for episode in policy),
            "policy_ramp_power": sum(episode["ramp_power"] for episode in policy),
            "status_quo_ramp_power": sum(episode["ramp_power"] for episode in baseline),
        },
        "bootstrap_by_month": _cluster_bootstrap_interval(
            [float(mean(episode["incremental"])) for episode in policy],
            [str(episode["month"]) for episode in policy],
            unit="month",
            draws=500,
        ),
        "bootstrap_by_day": _cluster_bootstrap_interval(
            [float(mean(episode["incremental"])) for episode in policy],
            [episode["day_group"] for episode in policy],
            unit="day",
            draws=500,
        ),
        "evaluation_seed_interval": {
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
    model = PPO.load(checkpoint_dir / "model.zip", device="cpu")
    normalization_path = checkpoint_dir / "vecnormalize.pkl"
    policy: list[dict[str, Any]] = []
    baseline: list[dict[str, Any]] = []
    for seed in seeds:
        for window in windows:
            policy.append(
                _episode(
                    factory=factory,
                    split=split,
                    seed=seed,
                    window_id=window,
                    model=model,
                    normalization_path=normalization_path,
                )
            )
            baseline.append(
                _episode(
                    factory=factory,
                    split=split,
                    seed=seed,
                    window_id=window,
                    model=None,
                    normalization_path=None,
                )
            )
    return _aggregate(policy, baseline, split)
