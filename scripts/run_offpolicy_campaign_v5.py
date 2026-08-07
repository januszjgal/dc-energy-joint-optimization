"""Run exploratory v5 off-policy SAC/TD3 screening on the committed v4 regime."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import stable_baselines3 as sb3
import torch as th
import torch.nn.functional as F
import yaml
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
EXPECTED_SB3_VERSION = "2.9.0"
PROTOCOL_PATH = ROOT / "env" / "protocols" / "v5_offpolicy_td3bc.yaml"

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
DEFAULT_CAMPAIGN_SEEDS = (301, 302, 303, 304, 305)
DESCRIPTIVE_TRANSFER_SEED = 42


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_normalized_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_frozen_protocol() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    raw = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("campaigns"), dict):
        raise ValueError(f"invalid v5 protocol: {PROTOCOL_PATH}")
    campaigns: dict[str, dict[str, Any]] = {}
    for name, config in raw["campaigns"].items():
        if not isinstance(config, dict):
            raise ValueError(f"invalid campaign {name!r} in {PROTOCOL_PATH}")
        campaign = dict(config)
        if "net_arch" in campaign:
            campaign["net_arch"] = tuple(int(value) for value in campaign["net_arch"])
        campaigns[str(name)] = campaign
    return raw, campaigns


V5_PROTOCOL, CAMPAIGNS = _load_frozen_protocol()
PROTOCOL_ID = str(V5_PROTOCOL["protocol"]["id"])
PROTOCOL_SHA256 = _sha256_normalized_text(PROTOCOL_PATH)

REGION_CONFIGS: dict[str, dict[str, Any]] = {
    "us": {
        "scenario": ROOT / "env" / "scenarios" / "us_model_v2_2025.yaml",
        "descriptive_transfer_scenario": (
            ROOT / "env" / "scenarios" / "us_model_eh_v2_2025.yaml"
        ),
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
        "descriptive_transfer_scenario": (
            ROOT / "env" / "scenarios" / "global_model_eh_v2_2025.yaml"
        ),
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
    campaign: str | None = None


class OffPolicyDiagnosticsCallback(BaseCallback):
    """Capture compact training diagnostics for SAC/TD3 runs."""

    def __init__(self) -> None:
        super().__init__()
        self.steps = 0
        self.emergency_interventions = 0
        self.decoder_adjustments = 0
        self.max_decoder_adjustment_l2 = 0.0
        self.max_emergency_adjustment_l2 = 0.0
        self.minimum_deadline_slack = math.inf
        self.max_batch_pool = 0.0
        self.action_values = 0
        self.saturated_actions = 0

    def _on_step(self) -> bool:
        self.steps += 1
        for info in self.locals.get("infos", []):
            if not info.get("safety_enabled", False):
                continue
            self.emergency_interventions += int(
                bool(
                    info.get(
                        "safety_emergency_intervened",
                        info.get("safety_intervened", False),
                    )
                )
            )
            self.decoder_adjustments += int(
                bool(info.get("safety_decoder_adjusted", False))
            )
            self.max_decoder_adjustment_l2 = max(
                self.max_decoder_adjustment_l2,
                float(info.get("safety_decoder_adjustment_l2", 0.0)),
            )
            self.max_emergency_adjustment_l2 = max(
                self.max_emergency_adjustment_l2,
                float(info.get("safety_emergency_adjustment_l2", 0.0)),
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
            "emergency_intervention_count": self.emergency_interventions,
            "emergency_intervention_rate": (
                self.emergency_interventions / self.steps if self.steps else 0.0
            ),
            "decoder_adjustment_count": self.decoder_adjustments,
            "decoder_adjustment_rate": (
                self.decoder_adjustments / self.steps if self.steps else 0.0
            ),
            "max_decoder_adjustment_l2": self.max_decoder_adjustment_l2,
            "max_emergency_adjustment_l2": self.max_emergency_adjustment_l2,
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


def _require_tested_sb3() -> None:
    if sb3.__version__ != EXPECTED_SB3_VERSION:
        raise RuntimeError(
            "Stable-Baselines3 runtime mismatch: "
            f"expected {EXPECTED_SB3_VERSION}, got {sb3.__version__}"
        )


def _replay_discounts(replay_data: Any, gamma: float) -> th.Tensor:
    discounts = getattr(replay_data, "discounts", None)
    if discounts is None:
        return th.full_like(replay_data.rewards, float(gamma))
    if isinstance(discounts, th.Tensor):
        return discounts
    return th.full_like(replay_data.rewards, float(discounts))


def _tensor_sha256(state_dict: dict[str, th.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode("utf-8"))
        value = state_dict[key].detach().cpu().numpy()
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _module_state_dict(module: th.nn.Module) -> dict[str, th.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def module_sha256(module: th.nn.Module) -> str:
    return _tensor_sha256(_module_state_dict(module))


def _parameter_delta_l2(
    before: dict[str, th.Tensor],
    after: dict[str, th.Tensor],
) -> float:
    total = 0.0
    for key in before:
        diff = after[key].to(dtype=th.float64) - before[key].to(dtype=th.float64)
        total += float(th.sum(diff * diff).item())
    return math.sqrt(total)


def _module_distance_l2(left: th.nn.Module, right: th.nn.Module) -> float:
    return _parameter_delta_l2(
        _module_state_dict(left),
        _module_state_dict(right),
    )


def actor_target_sync_summary(
    model: SAC | TD3,
    *,
    synchronize: bool,
) -> dict[str, Any]:
    if not isinstance(model, TD3):
        return {
            "supported": False,
            "synchronized": True,
            "distance_l2_before": 0.0,
            "distance_l2_after": 0.0,
        }
    distance_before = _module_distance_l2(model.actor, model.actor_target)
    if synchronize:
        model.actor_target.load_state_dict(model.actor.state_dict())
    distance_after = _module_distance_l2(model.actor, model.actor_target)
    return {
        "supported": True,
        "synchronized": distance_after <= 1e-12,
        "distance_l2_before": distance_before,
        "distance_l2_after": distance_after,
        "actor_hash": module_sha256(model.actor),
        "actor_target_hash": module_sha256(model.actor_target),
    }


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _require_tracked_source_clean() -> None:
    commands = (
        ["git", "diff", "--quiet", "--"],
        ["git", "diff", "--cached", "--quiet", "--"],
    )
    if any(subprocess.run(command, cwd=ROOT, check=False).returncode for command in commands):
        raise RuntimeError("tracked source must be committed before a frozen v5 run")


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
            discounts = _replay_discounts(replay_data, self.gamma)

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
    scenario_kind: str = "development",
) -> ResidualSafeOffPolicyEnv:
    if scenario_kind == "development":
        scenario_path = Path(REGION_CONFIGS[region]["scenario"])
    elif scenario_kind == "descriptive_transfer":
        scenario_path = Path(REGION_CONFIGS[region]["descriptive_transfer_scenario"])
    else:
        raise ValueError(f"unsupported scenario kind: {scenario_kind}")
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
        return env.bound_action_to_space(env.exact_native_teacher_action())
    if teacher_name == "native_marginal_cost":
        return env.bound_action_to_space(env.marginal_cost_teacher_action())
    if teacher_name == "status_quo":
        return env.bound_action_to_space(env.status_quo_action())
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
    preclip_action_violations = 0
    clipped_actions = 0
    executed_matches_stored = True
    try:
        while completed < episodes:
            obs, _ = env.reset(seed=seed + completed)
            while True:
                raw_action = teacher_action(env, teacher_name)
                bounded_action = env.bound_action_to_space(raw_action)
                if not np.allclose(raw_action, bounded_action, atol=1e-8):
                    clipped_actions += 1
                preclip_action_violations += int(
                    not env.action_space.contains(raw_action.astype(np.float32))
                )
                next_obs, reward, terminated, truncated, info = env.step(
                    bounded_action
                )
                executed = np.asarray(info["executed_action"], dtype=np.float32)
                executed_matches_stored = executed_matches_stored and bool(
                    np.allclose(executed, bounded_action, atol=1e-8)
                )
                observations.append(np.asarray(obs, dtype=np.float32))
                next_observations.append(np.asarray(next_obs, dtype=np.float32))
                actions.append(np.asarray(executed, dtype=np.float32))
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
            "teacher_action_audit": {
                "preclip_action_space_violations": int(preclip_action_violations),
                "clipped_action_count": int(clipped_actions),
                "stored_action_matches_executed": bool(executed_matches_stored),
            },
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
) -> dict[str, Any]:
    observations = dataset["observations"]
    targets = _normalize_to_policy_action(env, dataset["actions"])
    if steps <= 0 or len(observations) == 0:
        sync = actor_target_sync_summary(model, synchronize=False)
        return {"steps": 0.0, "final_loss": 0.0, **sync}
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
        elif algorithm in {"td3", "td3_bc"}:
            predicted = model.actor(obs_tensor)
        else:
            raise ValueError(f"unsupported algorithm: {algorithm}")
        loss = F.mse_loss(predicted, target_tensor)
        model.actor.optimizer.zero_grad()
        loss.backward()
        model.actor.optimizer.step()
        final_loss = float(loss.detach().cpu().item())
    sync = actor_target_sync_summary(model, synchronize=True)
    if not sync["synchronized"]:
        raise RuntimeError("behavior-cloned actor target did not synchronize")
    return {"steps": float(steps), "final_loss": final_loss, **sync}


def model_dir(job: Job, profile: str) -> Path:
    return MODEL_ROOT / job.stage / job.algorithm / job.region / profile / f"s{job.seed}"


def campaign_job(
    campaign_name: str,
    *,
    region: str,
    seed: int,
) -> Job:
    config = CAMPAIGNS[campaign_name]
    return Job(
        stage=str(config["stage"]),
        algorithm=str(config["algorithm"]),
        region=region,
        seed=seed,
        timesteps=int(config["timesteps"]),
        campaign=campaign_name,
    )


def campaign_source_bc_model_path(job: Job, stage_config: dict[str, Any]) -> Path | None:
    warm_start_campaign = stage_config.get("warm_start_campaign")
    if warm_start_campaign is None:
        return None
    source_job = campaign_job(
        str(warm_start_campaign),
        region=job.region,
        seed=job.seed,
    )
    source_profile = str(CAMPAIGNS[str(warm_start_campaign)]["profile"])
    return model_dir(source_job, source_profile) / "model.zip"


def _load_warm_start_parameters(
    model: SAC | TD3,
    source_model_path: Path,
    *,
    algorithm: str,
    env: Monitor,
) -> None:
    if not source_model_path.exists():
        raise FileNotFoundError(
            f"warm-start model missing: {source_model_path}"
        )
    if algorithm == "td3_bc":
        loaded = TD3BehaviorCloning.load(
            source_model_path,
            env=env,
            device="auto",
        )
    elif algorithm == "td3":
        loaded = TD3.load(source_model_path, env=env, device="auto")
    elif algorithm == "sac":
        loaded = SAC.load(source_model_path, env=env, device="auto")
    else:
        raise ValueError(f"unsupported warm-start algorithm: {algorithm}")
    model.set_parameters(loaded.get_parameters(), exact_match=True)


def baseline_summary(
    region: str,
    reward_scale: float,
    *,
    scenario_kind: str,
) -> dict[str, Any]:
    env = make_env(
        region,
        EVAL_SEED,
        reward_scale,
        domain_randomization=False,
        scenario_kind=scenario_kind,
    )
    try:
        total_reward, history = run_episode(
            env,
            lambda _obs, wrapped_env: wrapped_env.bound_action_to_space(
                wrapped_env.status_quo_action()
            ),
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
    scenario_kind: str,
    forbid_teacher_policy_calls: bool,
) -> dict[str, Any]:
    env = make_env(
        region,
        EVAL_SEED,
        reward_scale,
        domain_randomization=False,
        scenario_kind=scenario_kind,
    )
    try:
        env.set_teacher_policy_calls_allowed(not forbid_teacher_policy_calls)
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
        summary["teacher_call_guard_enabled"] = bool(
            forbid_teacher_policy_calls
        )
        summary["evaluation_scope"] = (
            "a-d-development-frozen-confirmation"
            if scenario_kind == "development"
            else "e-h-descriptive-transfer"
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


def resolve_stage_config(job: Job) -> dict[str, Any]:
    if job.campaign is not None:
        return CAMPAIGNS[job.campaign]
    return ALGO_CONFIGS[job.stage][job.algorithm]


def training_mode_label(
    job: Job,
    teacher_record: dict[str, Any] | None,
) -> str:
    if job.campaign is not None:
        return str(CAMPAIGNS[job.campaign]["expected_training_mode"])
    if teacher_record is not None and job.timesteps <= 0:
        return "teacher_bc_only"
    if teacher_record is not None:
        return "teacher_warmstart_rl"
    return "rl_only"


def run_job(job: Job) -> dict[str, Any]:
    _require_tracked_source_clean()
    _require_tested_sb3()
    stage_config = resolve_stage_config(job)
    reward_scale = float(
        stage_config["reward_scale_by_region"][job.region]
    )
    source_commit = _git_head()
    set_random_seed(job.seed)
    train_env = Monitor(
        make_env(
            job.region,
            job.seed,
            reward_scale,
            domain_randomization=True,
            scenario_kind="development",
        )
    )
    callback = OffPolicyDiagnosticsCallback()
    model = make_model(job.algorithm, train_env, job.seed, stage_config)
    source_bc_model_path = campaign_source_bc_model_path(job, stage_config)
    source_bc_model_sha256: str | None = None
    warm_start_actor_target_sync: dict[str, Any] | None = None
    if source_bc_model_path is not None:
        source_bc_model_sha256 = _sha256_file(source_bc_model_path)
        _load_warm_start_parameters(
            model,
            source_bc_model_path,
            algorithm=job.algorithm,
            env=train_env,
        )
        warm_start_actor_target_sync = actor_target_sync_summary(
            model,
            synchronize=False,
        )
        if not warm_start_actor_target_sync["synchronized"]:
            raise RuntimeError(
                "warm-start BC checkpoint has a stale actor_target"
            )
    actor_before = _module_state_dict(model.actor)
    critic_before = _module_state_dict(model.critic)
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
            "action_audit": dataset["teacher_action_audit"],
            "behavior_cloning": bc_summary,
        }
    if job.timesteps > 0:
        model.learn(
            total_timesteps=job.timesteps,
            callback=callback,
            progress_bar=False,
        )

    actor_after = _module_state_dict(model.actor)
    critic_after = _module_state_dict(model.critic)
    baseline = baseline_summary(
        job.region,
        reward_scale,
        scenario_kind="development",
    )
    evaluation = evaluate_model(
        model,
        job.region,
        reward_scale,
        baseline=baseline,
        scenario_kind="development",
        forbid_teacher_policy_calls=True,
    )
    descriptive_transfer_baseline = baseline_summary(
        job.region,
        reward_scale,
        scenario_kind="descriptive_transfer",
    )
    descriptive_transfer = evaluate_model(
        model,
        job.region,
        reward_scale,
        baseline=descriptive_transfer_baseline,
        scenario_kind="descriptive_transfer",
        forbid_teacher_policy_calls=True,
    )
    training = callback.summary()
    profile = str(stage_config["profile"])
    output_dir = model_dir(job, profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model"
    model.save(model_path)
    actor_before_sha256 = _tensor_sha256(actor_before)
    actor_after_sha256 = _tensor_sha256(actor_after)
    critic_before_sha256 = _tensor_sha256(critic_before)
    critic_after_sha256 = _tensor_sha256(critic_after)
    record = {
        "job": asdict(job),
        "campaign": job.campaign,
        "profile": profile,
        "algorithm_config": {
            key: value
            for key, value in stage_config.items()
            if key != "reward_scale_by_region"
        },
        "training_mode": training_mode_label(job, teacher_record),
        "reward_scale": reward_scale,
        "model_path": str(model_path.with_suffix(".zip").relative_to(ROOT)),
        "source_commit": source_commit,
        "protocol_id": PROTOCOL_ID,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)).replace("/", "\\"),
        "protocol_sha256": PROTOCOL_SHA256,
        "campaign_role": str(stage_config["role"]),
        "stable_baselines3_version": sb3.__version__,
        "evaluation_label": str(
            stage_config.get("evaluation_label", "a-d-development")
        ),
        "descriptive_transfer_label": str(
            stage_config.get(
                "descriptive_transfer_label",
                "e-h-descriptive-transfer",
            )
        ),
        "training": training,
        "n_updates": int(getattr(model, "_n_updates", 0)),
        "actor_hash_before": actor_before_sha256,
        "actor_hash_after": actor_after_sha256,
        "critic_hash_before": critic_before_sha256,
        "critic_hash_after": critic_after_sha256,
        "weight_update_evidence": {
            "actor_hash_changed": actor_before_sha256 != actor_after_sha256,
            "critic_hash_changed": critic_before_sha256 != critic_after_sha256,
            "actor_parameter_delta_l2": _parameter_delta_l2(
                actor_before,
                actor_after,
            ),
            "critic_parameter_delta_l2": _parameter_delta_l2(
                critic_before,
                critic_after,
            ),
        },
        "baseline": baseline,
        "evaluation": evaluation,
        "descriptive_transfer_baseline": descriptive_transfer_baseline,
        "descriptive_transfer_eh": descriptive_transfer,
        "teacher_present_during_rl": bool(
            stage_config.get("teacher_present_during_rl", False)
        ),
        "teacher_present_at_inference": bool(
            stage_config.get("teacher_present_at_inference", False)
        ),
        "safe_seed_passed": safe_seed_passed(evaluation),
    }
    if source_bc_model_path is not None:
        record["source_bc_model"] = str(source_bc_model_path.relative_to(ROOT))
        record["source_bc_model_sha256"] = source_bc_model_sha256
        record["warm_start_actor_target_sync"] = warm_start_actor_target_sync
    if teacher_record is not None:
        record["teacher"] = teacher_record
    if "bc_alpha" in stage_config or "td3bc_lambda_alpha" in stage_config:
        record["rl_hyperparameters"] = {
            "learning_rate": float(stage_config["learning_rate"]),
            "learning_starts": int(stage_config["learning_starts"]),
            "gradient_steps": int(stage_config["gradient_steps"]),
            "train_freq_steps": int(stage_config["train_freq"]),
            "action_noise_sigma": float(stage_config["action_noise_sigma"]),
            "target_policy_noise": float(stage_config["target_policy_noise"]),
            "target_noise_clip": float(stage_config["target_noise_clip"]),
            "bc_alpha": float(stage_config.get("bc_alpha", 0.0)),
            "td3bc_lambda_alpha": float(
                stage_config.get("td3bc_lambda_alpha", 0.0)
            ),
        }
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


def campaign_jobs(
    campaign_name: str,
    *,
    regions: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[Job]:
    return [
        campaign_job(campaign_name, region=region, seed=seed)
        for region in regions
        for seed in seeds
    ]


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record["job"]["stage"]),
            str(record["job"]["algorithm"]),
            str(record["job"]["region"]),
            str(record.get("profile")),
        )
        grouped.setdefault(key, []).append(record)

    aggregates: dict[str, Any] = {}
    for (stage, algorithm, region, profile), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["job"]["seed"]))
        savings = [
            float(row["evaluation"]["savings_vs_status_quo_pct"]) for row in rows
        ]
        emergency_interventions = [
            float(row["evaluation"]["safety"]["emergency_intervention_rate"])
            for row in rows
        ]
        decoder_adjustments = [
            float(row["evaluation"]["safety"]["decoder_adjustment_rate"])
            for row in rows
        ]
        safe_count = sum(int(bool(row["safe_seed_passed"])) for row in rows)
        aggregate = {
            "stage": stage,
            "algorithm": algorithm,
            "region": region,
            "profile": profile,
            "campaign": rows[0].get("campaign"),
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
            "mean_emergency_intervention_rate": mean(emergency_interventions),
            "max_emergency_intervention_rate": max(emergency_interventions),
            "mean_decoder_adjustment_rate": mean(decoder_adjustments),
            "max_decoder_adjustment_rate": max(decoder_adjustments),
            "safe_seed_count": safe_count,
            "seed_count": len(rows),
            "all_seeds_safe": safe_count == len(rows),
            "goal_savings_pct": GOAL_SAVINGS[region],
            "meets_goal": (
                safe_count == len(rows)
                and mean(emergency_interventions) < GOAL_INTERVENTION
                and mean(savings) >= GOAL_SAVINGS[region]
            ),
            "records": rows,
        }
        aggregates[f"{stage}:{algorithm}:{region}:{profile}"] = aggregate
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


def execute_jobs(
    jobs: list[Job],
    *,
    workers: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    records.sort(
        key=lambda row: (
            str(row["job"]["stage"]),
            str(row.get("campaign") or row["job"]["algorithm"]),
            str(row["job"]["region"]),
            int(row["job"]["seed"]),
        )
    )
    return {
        "records": records,
        "aggregates": aggregate_records(records),
    }


def run_stage(
    stage: str,
    algorithms: tuple[str, ...],
    *,
    workers: int,
) -> dict[str, Any]:
    return execute_jobs(stage_jobs(stage, algorithms), workers=workers)


def run_campaign(
    campaign_name: str,
    *,
    regions: tuple[str, ...],
    seeds: tuple[int, ...],
    workers: int,
) -> dict[str, Any]:
    return execute_jobs(
        campaign_jobs(campaign_name, regions=regions, seeds=seeds),
        workers=workers,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run v5 residual-safe SAC/TD3 screens"
    )
    parser.add_argument(
        "--campaign",
        choices=tuple(sorted(CAMPAIGNS)),
    )
    parser.add_argument(
        "--phase",
        choices=("screen", "iterate", "all"),
        default="all",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=("us", "global"),
        default=["us", "global"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_CAMPAIGN_SEEDS),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )
    args = parser.parse_args(argv)

    workers = max(1, int(args.workers))
    if args.campaign is not None:
        results = run_campaign(
            str(args.campaign),
            regions=tuple(str(region) for region in args.regions),
            seeds=tuple(int(seed) for seed in args.seeds),
            workers=workers,
        )
        output_path = OUT_ROOT / "campaigns" / f"{args.campaign}_results.json"
        write_json(output_path, results)
        return

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
