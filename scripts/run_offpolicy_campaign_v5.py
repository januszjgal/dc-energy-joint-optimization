"""Run exploratory v5 off-policy SAC/TD3 screening on the committed v4 regime."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import polyak_update, set_random_seed

ROOT = Path(__file__).resolve().parent.parent
os.sys.path.insert(0, str(ROOT))

from env.data_loader import load_scenario  # noqa: E402
from env.residual_safe_offpolicy_env import ResidualSafeOffPolicyEnv  # noqa: E402
from env.reward import RewardConfig  # noqa: E402
from env.safety_layer import SafetyConfig  # noqa: E402
from evaluate import compute_summary, run_episode  # noqa: E402

MODEL_ROOT = ROOT / "models" / "offpolicy_v5_continuous"
OUT_ROOT = ROOT / "output" / "offpolicy_v5_continuous"
SCREEN_RESULTS_PATH = OUT_ROOT / "screen_results.json"
ITERATE_RESULTS_PATH = OUT_ROOT / "iterate_results.json"
SUMMARY_PATH = OUT_ROOT / "summary.json"

for env_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

PEAK_PENALTY_WEIGHT = 0.015
STAGE_TIMESTEPS = {
    "screen": 8192,
    "iterate": 32768,
}
SCREEN_SEEDS = (301, 302, 303)
EVAL_SEED = 42
GOAL_SAVINGS = {"us": 5.0, "global": 10.0}
GOAL_INTERVENTION = 0.01
DEFAULT_WORKERS = 4

REGION_CONFIGS: dict[str, dict[str, Any]] = {
    "us": {
        "scenario": ROOT / "env" / "scenarios" / "us_model_v2_2025.yaml",
        "reward": {
            "service_backlog_weight": 700.0,
            "batch_completion_weight": 1000.0,
            "evaluation_service_backlog_weight": 1000.0,
            "evaluation_batch_completion_weight": 1000.0,
            "subtract_idle_cost": False,
            "urgency_potential_weight": 0.0,
        },
    },
    "global": {
        "scenario": ROOT / "env" / "scenarios" / "global_model_v2_2025.yaml",
        "reward": {
            "service_backlog_weight": 700.0,
            "batch_completion_weight": 1500.0,
            "evaluation_service_backlog_weight": 1000.0,
            "evaluation_batch_completion_weight": 1000.0,
            "subtract_idle_cost": False,
            "urgency_potential_weight": 250.0,
        },
    },
}

ALGO_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "screen": {
        "sac": {
            "profile": "screen-entauto025-rscale25e5-buf150k-warm4k-tf16-gs4-128x128",
            "reward_scale_by_region": {"us": 2.5e-4, "global": 2.5e-4},
            "learning_rate": 3e-4,
            "buffer_size": 150_000,
            "learning_starts": 4_096,
            "batch_size": 128,
            "tau": 0.005,
            "train_freq": 16,
            "gradient_steps": 4,
            "ent_coef": "auto_0.25",
            "net_arch": (128, 128),
        },
        "td3": {
            "profile": "screen-noise012-rscale25e5-buf150k-warm4k-tf16-gs4-128x128",
            "reward_scale_by_region": {"us": 2.5e-4, "global": 2.5e-4},
            "learning_rate": 3e-4,
            "buffer_size": 150_000,
            "learning_starts": 4_096,
            "batch_size": 128,
            "tau": 0.005,
            "train_freq": 16,
            "gradient_steps": 4,
            "action_noise_sigma": 0.12,
            "net_arch": (128, 128),
        },
    },
    "iterate": {
        "sac": {
            "profile": "iterate-entauto010-rscale5e4to1e4-buf250k-warm8k-tf16-gs8-256x256x128",
            "reward_scale_by_region": {"us": 5.0e-4, "global": 1.0e-4},
            "learning_rate": 2e-4,
            "buffer_size": 250_000,
            "learning_starts": 8_192,
            "batch_size": 256,
            "tau": 0.01,
            "train_freq": 16,
            "gradient_steps": 8,
            "ent_coef": "auto_0.10",
            "net_arch": (256, 256, 128),
        },
        "td3": {
            "profile": "iterate-noise008-rscale5e4to1e4-buf250k-warm8k-tf16-gs8-256x256x128",
            "reward_scale_by_region": {"us": 5.0e-4, "global": 1.0e-4},
            "learning_rate": 2e-4,
            "buffer_size": 250_000,
            "learning_starts": 8_192,
            "batch_size": 256,
            "tau": 0.01,
            "train_freq": 16,
            "gradient_steps": 8,
            "action_noise_sigma": 0.08,
            "net_arch": (256, 256, 128),
        },
    },
}


@dataclass(frozen=True)
class Job:
    stage: str
    algorithm: str
    region: str
    seed: int
    timesteps: int


class OffPolicyDiagnosticsCallback(BaseCallback):
    """Capture compact training diagnostics for SAC/TD3 runs."""

    def __init__(self) -> None:
        super().__init__()
        self.steps = 0
        self.interventions = 0
        self.max_projection_l2 = 0.0
        self.minimum_deadline_slack = math.inf
        self.max_batch_pool = 0.0
        self.action_values = 0
        self.saturated_actions = 0

    def _on_step(self) -> bool:
        self.steps += 1
        for info in self.locals.get("infos", []):
            if not info.get("safety_enabled", False):
                continue
            self.interventions += int(bool(info.get("safety_intervened", False)))
            self.max_projection_l2 = max(
                self.max_projection_l2,
                float(info.get("safety_projection_l2", 0.0)),
            )
            slack = float(
                info.get("safety_minimum_deadline_slack", math.inf)
            )
            if math.isfinite(slack):
                self.minimum_deadline_slack = min(
                    self.minimum_deadline_slack,
                    slack,
                )
            self.max_batch_pool = max(
                self.max_batch_pool,
                float(info.get("total_batch_pool", 0.0)),
            )
        actions = self.locals.get("actions")
        if actions is not None:
            values = np.asarray(actions, dtype=np.float64)
            self.action_values += values.size
            self.saturated_actions += int(
                np.count_nonzero(np.abs(values) >= 5.95)
            )
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "steps_observed": self.steps,
            "emergency_intervention_rate": (
                self.interventions / self.steps if self.steps else 0.0
            ),
            "max_projection_l2": self.max_projection_l2,
            "minimum_deadline_slack": (
                self.minimum_deadline_slack
                if math.isfinite(self.minimum_deadline_slack)
                else None
            ),
            "max_batch_pool": self.max_batch_pool,
            "action_saturation_fraction": (
                self.saturated_actions / self.action_values
                if self.action_values
                else 0.0
            ),
        }


class TD3BehaviorCloning(TD3):
    """TD3 actor update with TD3+BC-style Q normalization and BC regularization."""

    def __init__(
        self,
        *args: Any,
        bc_alpha: float = 1.0,
        td3bc_lambda_alpha: float = 2.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bc_alpha = float(bc_alpha)
        self.td3bc_lambda_alpha = float(td3bc_lambda_alpha)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses: list[float] = []
        actor_bc_losses: list[float] = []
        actor_q_lambdas: list[float] = []
        critic_losses: list[float] = []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(
                    0,
                    self.target_policy_noise,
                )
                noise = noise.clamp(
                    -self.target_noise_clip,
                    self.target_noise_clip,
                )
                next_actions = (
                    self.actor_target(replay_data.next_observations) + noise
                ).clamp(-1, 1)
                next_q_values = th.cat(
                    self.critic_target(
                        replay_data.next_observations,
                        next_actions,
                    ),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(
                replay_data.observations,
                replay_data.actions,
            )
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(float(critic_loss.item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                predicted_actions = self.actor(replay_data.observations)
                q_values = self.critic.q1_forward(
                    replay_data.observations,
                    predicted_actions,
                )
                q_scale = th.clamp(
                    q_values.abs().mean().detach(),
                    min=1e-6,
                )
                q_lambda = self.td3bc_lambda_alpha / q_scale
                bc_loss = F.mse_loss(predicted_actions, replay_data.actions)
                actor_loss = -q_lambda * q_values.mean() + self.bc_alpha * bc_loss
                actor_losses.append(float(actor_loss.item()))
                actor_bc_losses.append(float(bc_loss.item()))
                actor_q_lambdas.append(float(q_lambda.item()))

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.actor.parameters(),
                    self.actor_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.critic_batch_norm_stats,
                    self.critic_batch_norm_stats_target,
                    1.0,
                )
                polyak_update(
                    self.actor_batch_norm_stats,
                    self.actor_batch_norm_stats_target,
                    1.0,
                )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/actor_bc_loss", np.mean(actor_bc_losses))
            self.logger.record("train/actor_q_lambda", np.mean(actor_q_lambdas))
        self.logger.record("train/critic_loss", np.mean(critic_losses))


def reward_config(region: str, reward_scale: float) -> RewardConfig:
    reward = REGION_CONFIGS[region]["reward"]
    return RewardConfig(
        service_backlog_weight=float(reward["service_backlog_weight"]),
        batch_completion_weight=float(reward["batch_completion_weight"]),
        evaluation_service_backlog_weight=float(
            reward["evaluation_service_backlog_weight"]
        ),
        evaluation_batch_completion_weight=float(
            reward["evaluation_batch_completion_weight"]
        ),
        reward_scale=float(reward_scale),
        subtract_idle_cost=bool(reward["subtract_idle_cost"]),
        urgency_potential_weight=float(reward["urgency_potential_weight"]),
    )


def safety_config() -> SafetyConfig:
    return SafetyConfig(
        service_envelope_total=2.25,
        batch_arrival_envelope_total=1.0,
        future_fleet_capacity_total=4.0,
        envelope_id="ad-rounded-envelope-v1",
        envelope_scope="a-d-development-only",
        negative_demand_flush=False,
    )


def make_env(
    region: str,
    seed: int,
    reward_scale: float,
    *,
    domain_randomization: bool,
) -> ResidualSafeOffPolicyEnv:
    scenario_path = Path(REGION_CONFIGS[region]["scenario"])
    sites, power_model, batch_config = load_scenario(
        scenario_path,
        batch_enabled=True,
        dynamic_arrivals=True,
        seed=seed,
    )
    env = ResidualSafeOffPolicyEnv(
        sites=sites,
        power_model=power_model,
        batch_enabled=True,
        flexibility_factor=float(batch_config.get("flexibility_factor", 1.0)),
        deadline_penalty_weight=float(
            batch_config.get("deadline_penalty_weight", 2.0)
        ),
        urgency_horizon_steps=int(batch_config.get("urgency_horizon_steps", 12)),
        peak_penalty_weight=PEAK_PENALTY_WEIGHT,
        enforce_batch_completion=True,
        reward_config=reward_config(region, reward_scale),
        observe_episode_progress=True,
        deadline_bucket_edges=(1, 3, 6, 12, 24),
        domain_randomization=domain_randomization,
        safety_config=safety_config(),
    )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_model(
    algorithm: str,
    env: Monitor,
    seed: int,
    algo_config: dict[str, Any],
) -> SAC | TD3:
    policy_kwargs = {
        "net_arch": {
            "pi": list(algo_config["net_arch"]),
            "qf": list(algo_config["net_arch"]),
        }
    }
    common = dict(
        env=env,
        learning_rate=float(algo_config["learning_rate"]),
        buffer_size=int(algo_config["buffer_size"]),
        learning_starts=int(algo_config["learning_starts"]),
        batch_size=int(algo_config["batch_size"]),
        tau=float(algo_config["tau"]),
        gamma=1.0,
        train_freq=int(algo_config["train_freq"]),
        gradient_steps=int(algo_config["gradient_steps"]),
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
        device="auto",
    )
    if algorithm == "sac":
        return SAC(
            "MlpPolicy",
            ent_coef=algo_config["ent_coef"],
            **common,
        )
    if algorithm in {"td3", "td3_bc"}:
        action_dim = int(np.prod(env.action_space.shape))
        sigma = float(algo_config["action_noise_sigma"])
        action_noise = NormalActionNoise(
            mean=np.zeros(action_dim, dtype=np.float32),
            sigma=np.full(action_dim, sigma, dtype=np.float32),
        )
        target_policy_noise = float(
            algo_config.get("target_policy_noise", min(0.5, sigma * 2.0))
        )
        target_noise_clip = float(
            algo_config.get("target_noise_clip", max(0.1, sigma * 2.0))
        )
        model_cls: type[TD3] = (
            TD3BehaviorCloning
            if algorithm == "td3_bc" or "bc_alpha" in algo_config
            else TD3
        )
        model_kwargs: dict[str, Any] = {}
        if model_cls is TD3BehaviorCloning:
            model_kwargs["bc_alpha"] = float(algo_config.get("bc_alpha", 1.0))
            model_kwargs["td3bc_lambda_alpha"] = float(
                algo_config.get("td3bc_lambda_alpha", 2.5)
            )
        return model_cls(
            "MlpPolicy",
            action_noise=action_noise,
            policy_delay=int(algo_config.get("policy_delay", 2)),
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            **model_kwargs,
            **common,
        )
    raise ValueError(f"unsupported algorithm: {algorithm}")


def teacher_action(env: ResidualSafeOffPolicyEnv, teacher_name: str) -> np.ndarray:
    if teacher_name == "exact_native":
        return env.exact_native_teacher_action()
    if teacher_name == "native_marginal_cost":
        return env.marginal_cost_teacher_action()
    if teacher_name == "status_quo":
        return env.status_quo_action()
    raise ValueError(f"unsupported teacher: {teacher_name}")


def collect_teacher_rollout(
    region: str,
    reward_scale: float,
    *,
    teacher_name: str,
    seed: int,
    domain_randomization: bool,
    episodes: int,
) -> dict[str, Any]:
    env = make_env(
        region,
        seed,
        reward_scale,
        domain_randomization=domain_randomization,
    )
    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[float] = []
    infos: list[dict[str, Any]] = []
    completed = 0
    try:
        while completed < episodes:
            obs, _ = env.reset(seed=seed + completed)
            while True:
                action = teacher_action(env, teacher_name)
                next_obs, reward, terminated, truncated, info = env.step(action)
                observations.append(np.asarray(obs, dtype=np.float32))
                next_observations.append(np.asarray(next_obs, dtype=np.float32))
                actions.append(np.asarray(action, dtype=np.float32))
                rewards.append(float(reward))
                dones.append(float(terminated or truncated))
                infos.append(info)
                obs = next_obs
                if terminated or truncated:
                    completed += 1
                    break
        teacher_summary = compute_summary(infos, batch_enabled=True)
        return {
            "observations": np.asarray(observations, dtype=np.float32),
            "next_observations": np.asarray(
                next_observations,
                dtype=np.float32,
            ),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
            "infos": infos,
            "summary": teacher_summary,
        }
    finally:
        env.close()


def prefill_replay_buffer(
    model: SAC | TD3,
    dataset: dict[str, Any],
    env: Monitor,
) -> int:
    count = int(dataset["observations"].shape[0])
    normalized_actions = _normalize_to_policy_action(env, dataset["actions"])
    for index in range(count):
        model.replay_buffer.add(
            dataset["observations"][index : index + 1],
            dataset["next_observations"][index : index + 1],
            normalized_actions[index : index + 1],
            dataset["rewards"][index : index + 1],
            dataset["dones"][index : index + 1],
            [dataset["infos"][index]],
        )
    return count


def _normalize_to_policy_action(
    env: Monitor,
    actions: np.ndarray,
) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    return np.clip(
        2.0 * (actions - low) / (high - low) - 1.0,
        -1.0,
        1.0,
    ).astype(np.float32)


def behavior_clone_actor(
    model: SAC | TD3,
    env: Monitor,
    dataset: dict[str, Any],
    *,
    algorithm: str,
    steps: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    observations = dataset["observations"]
    targets = _normalize_to_policy_action(env, dataset["actions"])
    if steps <= 0 or len(observations) == 0:
        return {"steps": 0.0, "final_loss": 0.0}
    rng = np.random.default_rng(seed)
    model.actor.train()
    final_loss = 0.0
    for _ in range(steps):
        indices = rng.integers(
            0,
            len(observations),
            size=min(batch_size, len(observations)),
        )
        obs_tensor = th.as_tensor(
            observations[indices],
            dtype=th.float32,
            device=model.device,
        )
        target_tensor = th.as_tensor(
            targets[indices],
            dtype=th.float32,
            device=model.device,
        )
        if algorithm == "sac":
            predicted = model.actor(obs_tensor, deterministic=True)
        elif algorithm == "td3":
            predicted = model.actor(obs_tensor)
        else:
            raise ValueError(f"unsupported algorithm: {algorithm}")
        loss = F.mse_loss(predicted, target_tensor)
        model.actor.optimizer.zero_grad()
        loss.backward()
        model.actor.optimizer.step()
        final_loss = float(loss.detach().cpu().item())
    return {"steps": float(steps), "final_loss": final_loss}


def model_dir(job: Job, profile: str) -> Path:
    return MODEL_ROOT / job.stage / job.algorithm / job.region / profile / f"s{job.seed}"


def baseline_summary(region: str, reward_scale: float) -> dict[str, Any]:
    env = make_env(
        region,
        EVAL_SEED,
        reward_scale,
        domain_randomization=False,
    )
    try:
        total_reward, history = run_episode(
            env,
            lambda _obs, wrapped_env: wrapped_env.status_quo_action(),
            is_sb3=False,
        )
        summary = compute_summary(history, batch_enabled=True)
        summary["episode_reward"] = float(total_reward)
        return summary
    finally:
        env.close()


def evaluate_model(
    model: SAC | TD3,
    region: str,
    reward_scale: float,
    *,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    env = make_env(
        region,
        EVAL_SEED,
        reward_scale,
        domain_randomization=False,
    )
    try:
        total_reward, history = run_episode(env, model.predict, is_sb3=True)
        summary = compute_summary(history, batch_enabled=True)
        summary["episode_reward"] = float(total_reward)
        baseline_cost = float(baseline["total_cost"])
        summary["baseline_total_cost"] = baseline_cost
        summary["savings_vs_status_quo_usd"] = baseline_cost - float(
            summary["total_cost"]
        )
        summary["savings_vs_status_quo_pct"] = (
            100.0
            * (baseline_cost - float(summary["total_cost"]))
            / baseline_cost
        )
        return summary
    finally:
        env.close()


def safe_seed_passed(summary: dict[str, Any]) -> bool:
    safety = summary.get("safety", {})
    return (
        math.isclose(float(summary.get("service_served_total", 0.0)), float(summary.get("service_demand_total", 0.0)), rel_tol=0.0, abs_tol=1e-8)
        and math.isclose(float(summary.get("batch_completion_fraction", 0.0)), 1.0, rel_tol=0.0, abs_tol=1e-8)
        and float(summary.get("total_batch_expired", 0.0)) <= 1e-8
        and float(summary.get("terminal_batch_pool", 0.0)) <= 1e-8
        and float(summary.get("terminal_backlog", 0.0)) <= 1e-8
        and int(safety.get("infeasibility_certificates", 0)) == 0
        and float(safety.get("max_transport_conservation_error", 0.0)) <= 1e-8
    )


def run_job(job: Job) -> dict[str, Any]:
    stage_config = ALGO_CONFIGS[job.stage][job.algorithm]
    reward_scale = float(
        stage_config["reward_scale_by_region"][job.region]
    )
    set_random_seed(job.seed)
    train_env = Monitor(
        make_env(
            job.region,
            job.seed,
            reward_scale,
            domain_randomization=True,
        )
    )
    callback = OffPolicyDiagnosticsCallback()
    model = make_model(job.algorithm, train_env, job.seed, stage_config)
    teacher_record: dict[str, Any] | None = None
    if "teacher_name" in stage_config:
        dataset = collect_teacher_rollout(
            job.region,
            reward_scale,
            teacher_name=str(stage_config["teacher_name"]),
            seed=job.seed,
            domain_randomization=bool(
                stage_config.get("teacher_domain_randomization", True)
            ),
            episodes=int(stage_config.get("teacher_prefill_episodes", 1)),
        )
        prefill_count = prefill_replay_buffer(model, dataset, train_env)
        model.learning_starts = 0
        bc_summary = behavior_clone_actor(
            model,
            train_env,
            dataset,
            algorithm=job.algorithm,
            steps=int(stage_config.get("teacher_bc_steps", 0)),
            batch_size=int(
                stage_config.get("teacher_bc_batch_size", 256)
            ),
            seed=job.seed,
        )
        teacher_record = {
            "teacher_name": str(stage_config["teacher_name"]),
            "prefill_transitions": prefill_count,
            "rollout_summary": dataset["summary"],
            "behavior_cloning": bc_summary,
        }
    if job.timesteps > 0:
        model.learn(
            total_timesteps=job.timesteps,
            callback=callback,
            progress_bar=False,
        )

    baseline = baseline_summary(job.region, reward_scale)
    evaluation = evaluate_model(
        model,
        job.region,
        reward_scale,
        baseline=baseline,
    )
    training = callback.summary()
    profile = str(stage_config["profile"])
    output_dir = model_dir(job, profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model"
    model.save(model_path)
    record = {
        "job": asdict(job),
        "profile": profile,
        "algorithm_config": {
            key: value
            for key, value in stage_config.items()
            if key != "reward_scale_by_region"
        },
        "training_mode": (
            "teacher_bc_only"
            if teacher_record is not None and job.timesteps <= 0
            else "teacher_warmstart_rl"
            if teacher_record is not None
            else "rl_only"
        ),
        "reward_scale": reward_scale,
        "model_path": str(model_path.with_suffix(".zip").relative_to(ROOT)),
        "training": training,
        "baseline": baseline,
        "evaluation": evaluation,
        "safe_seed_passed": safe_seed_passed(evaluation),
    }
    if teacher_record is not None:
        record["teacher"] = teacher_record
    record_path = output_dir / "record.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    train_env.close()
    return record


def stage_jobs(stage: str, algorithms: tuple[str, ...]) -> list[Job]:
    return [
        Job(
            stage=stage,
            algorithm=algorithm,
            region=region,
            seed=seed,
            timesteps=int(STAGE_TIMESTEPS[stage]),
        )
        for algorithm in algorithms
        for region in ("us", "global")
        for seed in SCREEN_SEEDS
    ]


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record["job"]["stage"]),
            str(record["job"]["algorithm"]),
            str(record["job"]["region"]),
        )
        grouped.setdefault(key, []).append(record)

    aggregates: dict[str, Any] = {}
    for (stage, algorithm, region), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["job"]["seed"]))
        savings = [
            float(row["evaluation"]["savings_vs_status_quo_pct"]) for row in rows
        ]
        interventions = [
            float(row["evaluation"]["safety"]["intervention_rate"]) for row in rows
        ]
        safe_count = sum(int(bool(row["safe_seed_passed"])) for row in rows)
        aggregate = {
            "stage": stage,
            "algorithm": algorithm,
            "region": region,
            "profile": rows[0]["profile"],
            "mean_total_cost": mean(
                float(row["evaluation"]["total_cost"]) for row in rows
            ),
            "std_total_cost": pstdev(
                [float(row["evaluation"]["total_cost"]) for row in rows]
            )
            if len(rows) > 1
            else 0.0,
            "mean_savings_vs_status_quo_pct": mean(savings),
            "worst_savings_vs_status_quo_pct": min(savings),
            "mean_emergency_intervention_rate": mean(interventions),
            "max_emergency_intervention_rate": max(interventions),
            "safe_seed_count": safe_count,
            "seed_count": len(rows),
            "all_seeds_safe": safe_count == len(rows),
            "goal_savings_pct": GOAL_SAVINGS[region],
            "meets_goal": (
                safe_count == len(rows)
                and mean(interventions) < GOAL_INTERVENTION
                and mean(savings) >= GOAL_SAVINGS[region]
            ),
            "records": rows,
        }
        aggregates[f"{stage}:{algorithm}:{region}"] = aggregate
    return aggregates


def choose_best_algorithm(screen_aggregates: dict[str, Any]) -> str:
    per_algo: dict[str, list[dict[str, Any]]] = {}
    for aggregate in screen_aggregates.values():
        per_algo.setdefault(str(aggregate["algorithm"]), []).append(aggregate)
    scored: list[tuple[float, str]] = []
    for algorithm, aggregates in per_algo.items():
        if {agg["region"] for agg in aggregates} != {"us", "global"}:
            continue
        safe_penalty = min(
            1.0 if agg["all_seeds_safe"] else 0.0 for agg in aggregates
        )
        if safe_penalty <= 0.0:
            continue
        score = mean(
            float(agg["mean_savings_vs_status_quo_pct"])
            / GOAL_SAVINGS[str(agg["region"])]
            for agg in aggregates
        ) - 50.0 * mean(
            float(agg["mean_emergency_intervention_rate"]) for agg in aggregates
        )
        scored.append((score, algorithm))
    if not scored:
        return "sac"
    scored.sort(reverse=True)
    return scored[0][1]


def run_stage(
    stage: str,
    algorithms: tuple[str, ...],
    *,
    workers: int,
) -> dict[str, Any]:
    jobs = stage_jobs(stage, algorithms)
    records: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    records.sort(
        key=lambda row: (
            str(row["job"]["stage"]),
            str(row["job"]["algorithm"]),
            str(row["job"]["region"]),
            int(row["job"]["seed"]),
        )
    )
    return {
        "records": records,
        "aggregates": aggregate_records(records),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run v5 residual-safe SAC/TD3 screens"
    )
    parser.add_argument(
        "--phase",
        choices=("screen", "iterate", "all"),
        default="all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )
    args = parser.parse_args(argv)

    workers = max(1, int(args.workers))
    summary: dict[str, Any] = {
        "stage_timesteps": STAGE_TIMESTEPS,
        "seeds": list(SCREEN_SEEDS),
        "goal_savings_pct": GOAL_SAVINGS,
        "goal_emergency_intervention_rate": GOAL_INTERVENTION,
    }

    if args.phase in {"screen", "all"}:
        screen = run_stage("screen", ("sac", "td3"), workers=workers)
        write_json(SCREEN_RESULTS_PATH, screen)
        summary["screen"] = {
            "aggregates": screen["aggregates"],
            "selected_algorithm": choose_best_algorithm(screen["aggregates"]),
        }

    selected_algorithm = summary.get("screen", {}).get("selected_algorithm")
    if args.phase == "iterate" and selected_algorithm is None:
        if SCREEN_RESULTS_PATH.exists():
            existing_screen = json.loads(
                SCREEN_RESULTS_PATH.read_text(encoding="utf-8")
            )
            selected_algorithm = choose_best_algorithm(
                existing_screen["aggregates"]
            )
            summary["screen"] = {
                "aggregates": existing_screen["aggregates"],
                "selected_algorithm": selected_algorithm,
            }
        else:
            selected_algorithm = "sac"

    if args.phase in {"iterate", "all"}:
        algorithm = str(selected_algorithm or "sac")
        iterate = run_stage("iterate", (algorithm,), workers=workers)
        write_json(ITERATE_RESULTS_PATH, iterate)
        summary["iterate"] = {
            "algorithm": algorithm,
            "aggregates": iterate["aggregates"],
        }

    write_json(SUMMARY_PATH, summary)


if __name__ == "__main__":
    main()
