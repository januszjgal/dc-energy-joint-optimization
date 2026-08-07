"""Train one joint PPO policy with the v4 hard feasibility projector active."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.data_loader import load_scenario
from env.reward import RewardConfig
from env.safe_multi_dc_env import SafeMultiDCEnv
from env.safety_layer import SafetyConfig
from env.safety_layer import SafetyInfeasibleError
from train_v3 import (
    DEFAULT_DEADLINE_BUCKET_EDGES,
    TrainingConfig,
    linear_schedule,
)


class SafetyTrainingCallback(BaseCallback):
    """Collect projector metrics from every training transition."""

    def __init__(self) -> None:
        super().__init__()
        self.steps = 0
        self.interventions = 0
        self.projection_sum = 0.0
        self.projection_max = 0.0
        self.mandatory_steps = 0
        self.mandatory_max = 0.0
        self.minimum_deadline_slack = math.inf
        self.exact_zero_drains = 0
        self.exact_full_drains = 0
        self.negative_flush_steps = 0
        self.max_transport_error = 0.0
        self.binding_deadlines: dict[str, int] = {}

    def _on_step(self) -> bool:
        self.steps += self.training_env.num_envs
        for info in self.locals.get("infos", []):
            if not info.get("safety_enabled", False):
                continue
            intervened = bool(info["safety_intervened"])
            self.interventions += int(intervened)
            distance = float(info["safety_projection_l2"])
            self.projection_sum += distance
            self.projection_max = max(self.projection_max, distance)
            mandatory = float(info["safety_mandatory_batch"])
            self.mandatory_steps += int(mandatory > 1e-12)
            self.mandatory_max = max(self.mandatory_max, mandatory)
            slack = float(info["safety_minimum_deadline_slack"])
            if math.isfinite(slack):
                self.minimum_deadline_slack = min(
                    self.minimum_deadline_slack,
                    slack,
                )
            binding = info.get(
                "safety_binding_deadline_steps_remaining"
            )
            if binding is not None:
                key = str(int(binding))
                self.binding_deadlines[key] = (
                    self.binding_deadlines.get(key, 0) + 1
                )
            self.exact_zero_drains += int(
                info["safety_exact_zero_drain_count"]
            )
            self.exact_full_drains += int(
                info["safety_exact_full_drain_count"]
            )
            self.negative_flush_steps += int(
                info["safety_negative_flush_active"]
            )
            self.max_transport_error = max(
                self.max_transport_error,
                float(info["safety_transport_conservation_error"]),
            )
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "steps_observed": self.steps,
            "infeasibility_certificates": 0,
            "intervention_count": self.interventions,
            "intervention_rate": (
                self.interventions / self.steps if self.steps else 0.0
            ),
            "mean_projection_l2": (
                self.projection_sum / self.steps if self.steps else 0.0
            ),
            "max_projection_l2": self.projection_max,
            "mandatory_step_count": self.mandatory_steps,
            "mandatory_step_rate": (
                self.mandatory_steps / self.steps if self.steps else 0.0
            ),
            "max_mandatory_batch": self.mandatory_max,
            "minimum_deadline_slack": (
                self.minimum_deadline_slack
                if math.isfinite(self.minimum_deadline_slack)
                else None
            ),
            "binding_deadline_histogram": dict(
                sorted(
                    self.binding_deadlines.items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "exact_zero_drain_count": self.exact_zero_drains,
            "exact_full_drain_count": self.exact_full_drains,
            "negative_flush_step_count": self.negative_flush_steps,
            "max_transport_conservation_error": (
                self.max_transport_error
            ),
        }


def make_safe_env(
    scenario_path: Path,
    *,
    seed: int,
    peak_penalty_weight: float,
    reward_config: RewardConfig,
    safety_config: SafetyConfig,
    domain_randomization: bool,
    deadline_bucket_edges: tuple[int, ...] = (
        DEFAULT_DEADLINE_BUCKET_EDGES
    ),
) -> SafeMultiDCEnv:
    sites, power_model, batch_config = load_scenario(
        scenario_path,
        batch_enabled=True,
        dynamic_arrivals=True,
        seed=seed,
    )
    return SafeMultiDCEnv(
        sites=sites,
        power_model=power_model,
        batch_enabled=True,
        flexibility_factor=float(
            batch_config.get("flexibility_factor", 1.0)
        ),
        deadline_penalty_weight=float(
            batch_config.get("deadline_penalty_weight", 2.0)
        ),
        urgency_horizon_steps=int(
            batch_config.get("urgency_horizon_steps", 12)
        ),
        peak_penalty_weight=peak_penalty_weight,
        enforce_batch_completion=True,
        domain_randomization=domain_randomization,
        reward_config=reward_config,
        observe_episode_progress=True,
        deadline_bucket_edges=deadline_bucket_edges,
        safety_config=safety_config,
    )


def train_safe_candidate(
    scenario_path: Path,
    model_path: Path,
    vecnormalize_path: Path | None,
    diagnostics_path: Path,
    *,
    peak_penalty_weight: float,
    reward_config: RewardConfig,
    safety_config: SafetyConfig,
    training_config: TrainingConfig,
    deadline_bucket_edges: tuple[int, ...] = (
        DEFAULT_DEADLINE_BUCKET_EDGES
    ),
    check_environment: bool = False,
) -> None:
    if (
        training_config.timesteps <= 0
        or training_config.timesteps % training_config.n_steps != 0
    ):
        raise ValueError("timesteps must be positive and rollout-aligned")
    if training_config.n_steps % training_config.batch_size != 0:
        raise ValueError("n_steps must be divisible by batch_size")
    if not math.isclose(training_config.gamma, 1.0, abs_tol=1e-12):
        raise ValueError("hard completion training requires gamma=1")

    def factory() -> Monitor:
        return Monitor(
            make_safe_env(
                scenario_path,
                seed=training_config.seed,
                peak_penalty_weight=peak_penalty_weight,
                reward_config=reward_config,
                safety_config=safety_config,
                domain_randomization=True,
                deadline_bucket_edges=deadline_bucket_edges,
            )
        )

    if check_environment:
        raw_env = factory().unwrapped
        check_env(raw_env, warn=True)
        raw_env.close()

    vec_env = DummyVecEnv([factory])
    if training_config.normalize_observations:
        if vecnormalize_path is None:
            raise ValueError(
                "vecnormalize_path required for normalized training"
            )
        vec_env = VecNormalize(
            vec_env,
            training=True,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
    elif vecnormalize_path is not None:
        raise ValueError(
            "vecnormalize_path must be omitted without normalization"
        )

    learning_rate: float | Callable[[float], float]
    if training_config.learning_rate_schedule == "linear":
        learning_rate = linear_schedule(training_config.learning_rate)
    elif training_config.learning_rate_schedule == "fixed":
        learning_rate = training_config.learning_rate
    else:
        raise ValueError("unsupported learning-rate schedule")

    callback = SafetyTrainingCallback()
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=training_config.n_steps,
        batch_size=training_config.batch_size,
        n_epochs=training_config.n_epochs,
        gamma=training_config.gamma,
        gae_lambda=training_config.gae_lambda,
        target_kl=training_config.target_kl,
        ent_coef=0.0,
        clip_range_vf=None,
        normalize_advantage=True,
        policy_kwargs={
            "net_arch": list(training_config.net_arch)
        },
        verbose=1,
        seed=training_config.seed,
        tensorboard_log=None,
    )
    try:
        model.learn(
            total_timesteps=training_config.timesteps,
            callback=callback,
            progress_bar=False,
        )
    except SafetyInfeasibleError as exc:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            json.dumps(
                {
                    "status": "safety_infeasible",
                    "training": {
                        **asdict(training_config),
                        "net_arch": list(training_config.net_arch),
                    },
                    "reward": reward_config.as_dict(),
                    "safety": safety_config.as_dict(),
                    "scenario": str(scenario_path),
                    "certificate": exc.certificate,
                    "diagnostics": callback.summary(),
                    "num_timesteps": model.num_timesteps,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        vec_env.close()
        raise
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    if isinstance(vec_env, VecNormalize):
        assert vecnormalize_path is not None
        vecnormalize_path.parent.mkdir(parents=True, exist_ok=True)
        vec_env.save(vecnormalize_path)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(
            {
                "training": {
                    **asdict(training_config),
                    "net_arch": list(training_config.net_arch),
                },
                "reward": reward_config.as_dict(),
                "safety": safety_config.as_dict(),
                "scenario": str(scenario_path),
                "observation_shape": list(
                    vec_env.observation_space.shape
                ),
                "action_shape": list(vec_env.action_space.shape),
                "diagnostics": callback.summary(),
                "num_timesteps": model.num_timesteps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    vec_env.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train one hard-safe joint PPO candidate"
    )
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vecnormalize-path", type=Path)
    parser.add_argument("--diagnostics-path", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=["fixed", "linear"],
        default="fixed",
    )
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, required=True)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=[128, 128],
    )
    parser.add_argument("--normalize-observations", action="store_true")
    parser.add_argument("--peak-penalty-weight", type=float, default=0.015)
    parser.add_argument(
        "--service-backlog-weight",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--batch-completion-weight",
        type=float,
        required=True,
    )
    parser.add_argument("--reward-scale", type=float, default=1e-4)
    parser.add_argument("--subtract-idle-cost", action="store_true")
    parser.add_argument(
        "--urgency-potential-weight",
        type=float,
        default=0.0,
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
        "--envelope-id",
        default="ad-rounded-envelope-v1",
    )
    parser.add_argument(
        "--envelope-scope",
        default="a-d-development-only",
    )
    parser.add_argument("--negative-demand-flush", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args(argv)

    reward = RewardConfig(
        service_backlog_weight=args.service_backlog_weight,
        batch_completion_weight=args.batch_completion_weight,
        reward_scale=args.reward_scale,
        subtract_idle_cost=args.subtract_idle_cost,
        urgency_potential_weight=args.urgency_potential_weight,
    )
    safety = SafetyConfig(
        service_envelope_total=args.service_envelope_total,
        batch_arrival_envelope_total=(
            args.batch_arrival_envelope_total
        ),
        future_fleet_capacity_total=(
            args.future_fleet_capacity_total
        ),
        envelope_id=args.envelope_id,
        envelope_scope=args.envelope_scope,
        negative_demand_flush=args.negative_demand_flush,
    )
    training = TrainingConfig(
        timesteps=args.timesteps,
        learning_rate=args.lr,
        learning_rate_schedule=args.lr_schedule,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        target_kl=args.target_kl,
        normalize_observations=args.normalize_observations,
        seed=args.seed,
        net_arch=tuple(args.net_arch),
    )
    train_safe_candidate(
        args.scenario,
        args.model_path,
        args.vecnormalize_path,
        args.diagnostics_path,
        peak_penalty_weight=args.peak_penalty_weight,
        reward_config=reward,
        safety_config=safety,
        training_config=training,
        check_environment=args.check_env,
    )


if __name__ == "__main__":
    main()
