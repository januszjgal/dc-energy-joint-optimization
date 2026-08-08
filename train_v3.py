"""Train one exploratory v3 PPO recovery candidate.

This entry point is isolated from the frozen v2 campaign.  It adds only the
controls required by ``env/protocols/v3_reward_sweep.yaml`` and always retains
full-dollar costs in environment info for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.data_loader import load_scenario
from env.multi_dc_env import MultiDCEnv
from env.reward import RewardConfig


DEFAULT_DEADLINE_BUCKET_EDGES = (1, 3, 6, 12, 24)


@dataclass(frozen=True)
class TrainingConfig:
    timesteps: int
    learning_rate: float
    learning_rate_schedule: str
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    target_kl: float | None
    normalize_observations: bool
    seed: int
    net_arch: tuple[int, ...]


class TrainingDiagnosticsCallback(BaseCallback):
    """Collect compact rollout diagnostics without a TensorBoard dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.steps = 0
        self.max_service_backlog = 0.0
        self.max_batch_pool = 0.0
        self.batch_expired = 0.0
        self.action_values = 0
        self.saturated_actions = 0

    def _on_step(self) -> bool:
        self.steps += self.training_env.num_envs
        for info in self.locals.get("infos", []):
            self.max_service_backlog = max(
                self.max_service_backlog,
                float(
                    sum(
                        dc.get("backlog", 0.0)
                        for dc in info.get("per_dc", [])
                    )
                ),
            )
            self.max_batch_pool = max(
                self.max_batch_pool,
                float(info.get("total_batch_pool", 0.0)),
            )
            self.batch_expired += float(
                info.get("total_batch_expired", 0.0)
            )
        actions = self.locals.get("actions")
        if actions is not None:
            values = np.asarray(actions, dtype=np.float64)
            self.action_values += values.size
            self.saturated_actions += int(
                np.count_nonzero(np.abs(values) >= 2.95)
            )
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "steps_observed": self.steps,
            "max_service_backlog": self.max_service_backlog,
            "max_batch_pool": self.max_batch_pool,
            "batch_expired_during_training": self.batch_expired,
            "action_saturation_fraction": (
                self.saturated_actions / self.action_values
                if self.action_values
                else 0.0
            ),
        }


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linearly anneal a value to zero over training."""

    def schedule(progress_remaining: float) -> float:
        return float(progress_remaining * initial_value)

    return schedule


def make_recovery_env(
    scenario_path: Path,
    *,
    seed: int,
    peak_penalty_weight: float,
    service_backlog_weight: float,
    batch_completion_weight: float,
    reward_scale: float,
    subtract_idle_cost: bool,
    urgency_potential_weight: float,
    domain_randomization: bool,
    deadline_bucket_edges: tuple[int, ...] = DEFAULT_DEADLINE_BUCKET_EDGES,
) -> MultiDCEnv:
    """Build the opt-in v3 joint environment."""
    sites, power_model, batch_config = load_scenario(
        scenario_path,
        batch_enabled=True,
        dynamic_arrivals=True,
        seed=seed,
    )
    return MultiDCEnv(
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
        reward_config=RewardConfig(
            service_backlog_weight=service_backlog_weight,
            batch_completion_weight=batch_completion_weight,
            reward_scale=reward_scale,
            subtract_idle_cost=subtract_idle_cost,
            urgency_potential_weight=urgency_potential_weight,
        ),
        observe_episode_progress=True,
        deadline_bucket_edges=deadline_bucket_edges,
    )


def train_candidate(
    scenario_path: Path,
    model_path: Path,
    vecnormalize_path: Path | None,
    diagnostics_path: Path,
    *,
    peak_penalty_weight: float,
    reward_config: RewardConfig,
    training_config: TrainingConfig,
    deadline_bucket_edges: tuple[int, ...] = DEFAULT_DEADLINE_BUCKET_EDGES,
    check_environment: bool = False,
) -> None:
    """Train and persist one candidate plus normalization/diagnostics state."""
    if training_config.timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if training_config.timesteps % training_config.n_steps != 0:
        raise ValueError(
            "timesteps must be divisible by n_steps for exact PPO budgets"
        )
    if training_config.batch_size <= 1:
        raise ValueError("batch_size must be greater than one")
    rollout_size = training_config.n_steps
    if rollout_size % training_config.batch_size != 0:
        raise ValueError(
            "n_steps must be divisible by batch_size for one-environment runs"
        )
    if not math.isclose(training_config.gamma, 1.0, abs_tol=1e-12):
        raise ValueError("v3 completion accounting requires gamma=1")

    def factory() -> Monitor:
        env = make_recovery_env(
            scenario_path,
            seed=training_config.seed,
            peak_penalty_weight=peak_penalty_weight,
            service_backlog_weight=reward_config.service_backlog_weight,
            batch_completion_weight=reward_config.batch_completion_weight,
            reward_scale=reward_config.reward_scale,
            subtract_idle_cost=reward_config.subtract_idle_cost,
            urgency_potential_weight=(
                reward_config.urgency_potential_weight
            ),
            domain_randomization=True,
            deadline_bucket_edges=deadline_bucket_edges,
        )
        return Monitor(env)

    if check_environment:
        raw_env = factory().unwrapped
        check_env(raw_env, warn=True)
        raw_env.close()

    vec_env = DummyVecEnv([factory])
    if training_config.normalize_observations:
        vec_env = VecNormalize(
            vec_env,
            training=True,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
        if vecnormalize_path is None:
            raise ValueError(
                "vecnormalize_path is required when normalization is enabled"
            )
    elif vecnormalize_path is not None:
        raise ValueError(
            "vecnormalize_path must be omitted when normalization is disabled"
        )

    learning_rate: float | Callable[[float], float]
    if training_config.learning_rate_schedule == "linear":
        learning_rate = linear_schedule(training_config.learning_rate)
    elif training_config.learning_rate_schedule == "fixed":
        learning_rate = training_config.learning_rate
    else:
        raise ValueError(
            "learning_rate_schedule must be 'fixed' or 'linear'"
        )

    callback = TrainingDiagnosticsCallback()
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
        policy_kwargs={"net_arch": list(training_config.net_arch)},
        verbose=1,
        seed=training_config.seed,
        tensorboard_log=None,
    )
    model.learn(
        total_timesteps=training_config.timesteps,
        callback=callback,
        progress_bar=False,
    )

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
        description="Train one PPO reward-sweep v3 candidate"
    )
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vecnormalize-path", type=Path)
    parser.add_argument("--diagnostics-path", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule", choices=["fixed", "linear"], default="fixed"
    )
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--normalize-observations", action="store_true")
    parser.add_argument("--peak-penalty-weight", type=float, default=0.015)
    parser.add_argument(
        "--service-backlog-weight", type=float, required=True
    )
    parser.add_argument(
        "--batch-completion-weight", type=float, required=True
    )
    parser.add_argument("--reward-scale", type=float, default=1e-4)
    parser.add_argument("--subtract-idle-cost", action="store_true")
    parser.add_argument("--urgency-potential-weight", type=float, default=0.0)
    parser.add_argument(
        "--deadline-bucket-edges",
        type=int,
        nargs="+",
        default=list(DEFAULT_DEADLINE_BUCKET_EDGES),
    )
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args(argv)

    reward_config = RewardConfig(
        service_backlog_weight=args.service_backlog_weight,
        batch_completion_weight=args.batch_completion_weight,
        reward_scale=args.reward_scale,
        subtract_idle_cost=args.subtract_idle_cost,
        urgency_potential_weight=args.urgency_potential_weight,
    )
    training_config = TrainingConfig(
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
    train_candidate(
        args.scenario,
        args.model_path,
        args.vecnormalize_path,
        args.diagnostics_path,
        peak_penalty_weight=args.peak_penalty_weight,
        reward_config=reward_config,
        training_config=training_config,
        deadline_bucket_edges=tuple(args.deadline_bucket_edges),
        check_environment=args.check_env,
    )


if __name__ == "__main__":
    main()
