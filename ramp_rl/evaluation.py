"""Deterministic validation/test evaluation for policy and status quo."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ramp_rl.contract import EnvRequest, RampEnvAdapter, RampEnvironmentFactory
from ramp_rl.evidence import bootstrap_interval, evaluate_success_gate, optimizer_seed_interval


def _episode(
    *,
    factory: RampEnvironmentFactory,
    split: str,
    seed: int,
    window_id: str,
    model: PPO | SAC | None,
    normalization_path: Path | None,
) -> dict[str, Any]:
    request = EnvRequest(split=split, seed=seed, window_id=window_id, training=False)
    adapter = RampEnvAdapter(factory(request), request)
    if model is None:
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
            local_request = EnvRequest(split=split, seed=seed, window_id=window_id, training=False)
            return RampEnvAdapter(factory(local_request), local_request)

        base_vec = DummyVecEnv([make])
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
    return {
        "window_id": window_id,
        "evaluation_seed": seed,
        "month": int(reset_info["episode_context"]["month"]),
        "day_group": str(reset_info["episode_context"].get("day", window_id)),
        "source_hashes": dict(reset_info["episode_context"]["source_hashes"]),
        "future_realized_features_exposed": bool(reset_info["episode_context"]["future_realized_features_exposed"]),
        "ramp_h1": (
            [float(info["ramp_h1_adjusted"]) for info in infos]
            + [float(value) for value in infos[-1]["terminal_tail_ramp_h1_adjusted"]]
        ),
        "ramp_h3": (
            [float(info["ramp_h3_adjusted"]) for info in infos]
            + [float(value) for value in infos[-1]["terminal_tail_ramp_h3_adjusted"]]
        ),
        "incremental": (
            [float(info["incremental_ramp_impact"]) for info in infos]
            + [float(value) for value in infos[-1]["terminal_tail_incremental_ramp_impact"]]
        ),
        "energy_cost": sum(float(info["energy_cost"]) for info in infos),
        "status_quo_energy_cost": sum(float(info["status_quo_energy_cost"]) for info in infos),
        "service_unserved": sum(float(info["service_unserved"]) for info in infos),
        "batch_unfinished": float(infos[-1]["batch_unfinished"]),
        "batch_expired": sum(float(info["batch_expired"]) for info in infos),
        "certificate_violations": sum(int(info["certificate_violations"]) for info in infos),
        "terminal_work": float(infos[-1]["terminal_work"]),
        "emergency_count": sum(bool(info["emergency_feasibility"]) for info in infos),
        "step_count": len(infos),
        "semantic_adjustment_l2": sum(float(info["semantic_adjustment_l2"]) for info in infos),
        "deferrable_pre_service": sum(float(info["deferrable_pre_service"]) for info in infos),
        "ramp_power": sum(float(info["dc_power_during_realized_ramp"]) for info in infos),
    }


def _aggregate(policy: list[dict[str, Any]], baseline: list[dict[str, Any]], split: str) -> dict[str, Any]:
    impacts = [value for episode in policy for value in episode["incremental"]]
    h1 = [value for episode in policy for value in episode["ramp_h1"]]
    h3 = [value for episode in policy for value in episode["ramp_h3"]]
    policy_cost = sum(episode["energy_cost"] for episode in policy)
    baseline_cost = sum(episode["energy_cost"] for episode in baseline)
    market_values: dict[str, list[float]] = defaultdict(list)
    for episode in policy:
        market_values[f"fixture-month-{episode['month']:02d}"].extend(episode["incremental"])
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
        "mean_incremental_ramp_impact": float(mean(impacts)),
        "per_market_macro": {key: float(mean(values)) for key, values in market_values.items()},
        "ramp_h1_adjusted_p95": float(np.quantile(h1, 0.95)),
        "ramp_h1_adjusted_max": float(max(h1)),
        "ramp_h3_adjusted_p95": float(np.quantile(h3, 0.95)),
        "ramp_h3_adjusted_max": float(max(h3)),
        "energy_cost_ratio": float(policy_cost / baseline_cost),
        "emergency_feasibility_rate": float(
            sum(episode["emergency_count"] for episode in policy)
            / sum(episode["step_count"] for episode in policy)
        ),
        "semantic_adjustment_l2": sum(episode["semantic_adjustment_l2"] for episode in policy),
        "future_leakage_detected": any(episode["future_realized_features_exposed"] for episode in policy),
        "behavior_audit": {
            "deferrable_pre_service": sum(episode["deferrable_pre_service"] for episode in policy),
            "policy_ramp_power": sum(episode["ramp_power"] for episode in policy),
            "status_quo_ramp_power": sum(episode["ramp_power"] for episode in baseline),
        },
        "bootstrap_by_month": bootstrap_interval(
            [float(mean(episode["incremental"])) for episode in policy],
            [str(episode["month"]) for episode in policy],
            draws=500,
        ),
        "bootstrap_by_day": bootstrap_interval(
            [float(mean(episode["incremental"])) for episode in policy],
            [episode["day_group"] for episode in policy],
            draws=500,
        ),
        "evaluation_seed_interval": optimizer_seed_interval(
            float(mean(values)) for values in per_evaluation_seed.values()
        ),
        "policy_episodes": policy,
        "status_quo_episodes": baseline,
    }
    summary["success_gate"] = evaluate_success_gate(summary, split=split)
    return summary


def evaluate_checkpoint(
    *,
    factory: RampEnvironmentFactory,
    algorithm: str,
    checkpoint_dir: Path,
    split: str,
    seeds: list[int],
    windows: list[str],
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    model_class = PPO if algorithm == "ppo" else SAC
    model = model_class.load(checkpoint_dir / "model.zip", device="cpu")
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
