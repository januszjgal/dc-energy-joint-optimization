"""SB3 2.9 pure PPO/SAC runner with resumable, hashable evidence."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import random
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ramp_rl.contract import EnvRequest, RampEnvAdapter, RampEnvironmentFactory
from ramp_rl.evidence import FrozenLagrangian, sha256_file, sha256_json, verify_pure_rl_manifest
from ramp_rl.schema import EXPECTED_SB3_VERSION, FORBIDDEN_TRAINING_INPUTS


def load_factory(specification: str) -> RampEnvironmentFactory:
    module_name, separator, attribute = specification.partition(":")
    if not separator:
        raise ValueError("environment factory must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"{specification} is not callable")
    return factory


def _state_hash(state: dict[str, torch.Tensor], prefixes: tuple[str, ...] = ()) -> str:
    digest = hashlib.sha256()
    selected = 0
    for key in sorted(state):
        if prefixes and not key.startswith(prefixes):
            continue
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
        selected += 1
    if selected == 0:
        raise ValueError("state hash selected no tensors")
    return digest.hexdigest()


def model_hashes(model: PPO | SAC) -> dict[str, str]:
    if isinstance(model, PPO):
        state = model.policy.state_dict()
        return {
            "policy": _state_hash(state, ("mlp_extractor.policy_net", "action_net", "log_std")),
            "critic": _state_hash(state, ("mlp_extractor.value_net", "value_net")),
        }
    return {
        "policy": _state_hash(model.actor.state_dict()),
        "critic": _state_hash(model.critic.state_dict()),
    }


class EvidenceCallback(BaseCallback):
    def __init__(
        self,
        *,
        n_envs: int,
        epsilon_pct: float,
        dual_config: dict[str, Any],
        initial_multiplier: float,
    ) -> None:
        super().__init__()
        self.interactions = 0
        self.terminals = 0
        self.emergency = 0
        self.semantic_adjustment_sum = 0.0
        self.non_agent_actions = 0
        self.episode_records: dict[str, dict[str, Any]] = {}
        self.episode_energy = np.zeros(n_envs, dtype=np.float64)
        self.episode_status_quo_energy = np.zeros(n_envs, dtype=np.float64)
        self.last_step_all_terminal = False
        self.epsilon_pct = float(epsilon_pct)
        self.dual = FrozenLagrangian(
            multiplier=float(initial_multiplier),
            learning_rate=float(dual_config["learning_rate"]),
            maximum=float(dual_config["maximum"]),
        )
        self.dual_updates: list[dict[str, float | str]] = []

    def _on_step(self) -> bool:
        infos = list(self.locals.get("infos", []))
        self.interactions += len(infos)
        self.last_step_all_terminal = bool(infos) and all(
            bool(info.get("actual_terminal", False)) for info in infos
        )
        any_terminal = any(bool(info.get("actual_terminal", False)) for info in infos)
        if any_terminal and not self.last_step_all_terminal:
            raise RuntimeError("vector environments reached asynchronous terminal boundaries")
        for index, info in enumerate(infos):
            self.terminals += int(bool(info.get("actual_terminal", False)))
            self.emergency += int(bool(info.get("emergency_feasibility", False)))
            self.semantic_adjustment_sum += float(info.get("semantic_adjustment_l2", 0.0))
            self.non_agent_actions += int(info.get("action_provenance") != "agent_semantic")
            context = info.get("ramp_episode_context", {})
            window_id = str(context.get("window_id", info.get("window_id", "")))
            self.episode_records[window_id] = {
                "window_id": window_id,
                "split": str(context.get("split", "")),
                "source_hashes": dict(context.get("source_hashes", {})),
                "forecast_model": str(context.get("forecast_model", "")),
                "forecast_vintage": str(context.get("forecast_vintage", "")),
                "future_realized_features_exposed": bool(
                    context.get("future_realized_features_exposed", True)
                ),
            }
            self.episode_energy[index] += float(info.get("energy_cost", 0.0))
            self.episode_status_quo_energy[index] += float(
                info.get("status_quo_energy_cost", 0.0)
            )
        if self.last_step_all_terminal:
            energy_cost = float(self.episode_energy.mean())
            status_quo_energy_cost = float(self.episode_status_quo_energy.mean())
            multiplier = self.dual.update(
                energy_cost,
                status_quo_energy_cost,
                self.epsilon_pct,
                split="train",
            )
            self.training_env.env_method("set_lagrangian_multiplier", multiplier)
            self.dual_updates.append(
                {
                    "split": "train",
                    "energy_cost": energy_cost,
                    "status_quo_energy_cost": status_quo_energy_cost,
                    "multiplier": multiplier,
                }
            )
            self.episode_energy.fill(0.0)
            self.episode_status_quo_energy.fill(0.0)
        return True


def _make_base_vec_env(
    factory: RampEnvironmentFactory,
    *,
    seed: int,
    n_envs: int,
    epsilon_pct: float,
    lagrangian_multiplier: float,
) -> DummyVecEnv:
    env_fns: list[Callable[[], gym.Env]] = []
    for rank in range(n_envs):
        request = EnvRequest(
            split="train",
            seed=seed + rank,
            rank=rank,
            training=True,
            epsilon_pct=epsilon_pct,
            lagrangian_multiplier=lagrangian_multiplier,
        )

        def make(request: EnvRequest = request) -> gym.Env:
            return RampEnvAdapter(factory(request), request)

        env_fns.append(make)
    return DummyVecEnv(env_fns)


def _normalization_summary(vec: VecNormalize) -> dict[str, Any]:
    return {
        "fit_split": "train",
        "frozen_for": ["validation", "test"],
        "observation_mean": np.asarray(vec.obs_rms.mean).tolist(),
        "observation_variance": np.asarray(vec.obs_rms.var).tolist(),
        "sample_count": float(vec.obs_rms.count),
        "reward_normalization_training_only": True,
    }


def source_bundle_hash() -> str:
    root = Path(__file__).resolve().parent.parent
    paths = [
        root / "ramp_rl" / name
        for name in (
            "campaign.py",
            "contract.py",
            "evaluation.py",
            "evidence.py",
            "runner.py",
            "schema.py",
        )
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _factory_identity(factory: RampEnvironmentFactory) -> dict[str, str]:
    source = inspect.getsourcefile(factory)
    if source is None:
        raise ValueError("environment factory must have an inspectable source file")
    path = Path(source).resolve()
    root = Path(__file__).resolve().parent.parent
    try:
        display_path = str(path.relative_to(root))
    except ValueError:
        display_path = path.name
    return {
        "callable": f"{factory.__module__}:{factory.__qualname__}",
        "source_path": display_path,
        "source_sha256": sha256_file(path),
    }


def _assert_artifact_hash(path: Path, expected: dict[str, Any], label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"resume requires {label}: {path}")
    actual = sha256_file(path)
    if actual != expected.get("sha256"):
        raise RuntimeError(f"resume {label} hash mismatch")


def _make_model(
    algorithm: str,
    env: VecNormalize,
    seed: int,
    config: dict[str, Any],
) -> PPO | SAC:
    common = {
        "policy": "MlpPolicy",
        "env": env,
        "gamma": 1.0,
        "seed": seed,
        "device": "cpu",
        "verbose": 0,
        "policy_kwargs": {"net_arch": list(config.get("net_arch", [64, 64]))},
    }
    if algorithm == "ppo":
        return PPO(
            **common,
            n_steps=int(config["n_steps"]),
            batch_size=int(config["batch_size"]),
            n_epochs=int(config["n_epochs"]),
            gae_lambda=float(config["gae_lambda"]),
            learning_rate=float(config["learning_rate"]),
        )
    if algorithm == "sac":
        return SAC(
            **common,
            n_steps=int(config["n_steps"]),
            buffer_size=int(config["buffer_size"]),
            learning_starts=int(config["learning_starts"]),
            batch_size=int(config["batch_size"]),
            train_freq=int(config["train_freq"]),
            gradient_steps=int(config["gradient_steps"]),
            learning_rate=float(config["learning_rate"]),
        )
    raise ValueError(f"unsupported pure-RL algorithm: {algorithm}")


def run_training(
    *,
    factory: RampEnvironmentFactory,
    protocol: dict[str, Any],
    algorithm: str,
    seed: int,
    target_timesteps: int,
    output_dir: Path,
    n_envs: int | None = None,
    resume: bool = True,
    fixture_profile: bool = False,
    epsilon_pct: float | None = None,
) -> dict[str, Any]:
    if sb3.__version__ != EXPECTED_SB3_VERSION:
        raise RuntimeError(f"expected SB3 {EXPECTED_SB3_VERSION}, found {sb3.__version__}")
    if algorithm not in {"ppo", "sac"}:
        raise ValueError("algorithm must be ppo or sac")
    if target_timesteps <= 0:
        raise ValueError("target timesteps must be positive")
    protocol_n_envs = int(protocol["training"]["vectorized_environments"])
    if n_envs is None:
        n_envs = 2 if fixture_profile else protocol_n_envs
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if not fixture_profile and n_envs != protocol_n_envs:
        raise ValueError(
            f"integrated jobs require protocol vectorized_environments={protocol_n_envs}"
        )
    allowed_epsilons = {
        float(value) for value in protocol["multiobjective"]["epsilon_sensitivity_pct"]
    }
    if epsilon_pct is None:
        epsilon_pct = float(protocol["multiobjective"]["primary_energy_budget_pct"])
    if float(epsilon_pct) not in allowed_epsilons:
        raise ValueError(f"epsilon_pct must be one of {sorted(allowed_epsilons)}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = dict(protocol["algorithms"][algorithm])
    if fixture_profile:
        config.update(
            {
                "net_arch": [32, 32],
                "learning_rate": 5e-4,
                "batch_size": 16,
                "n_steps": 16 if algorithm == "ppo" else 36,
                "n_epochs": 2,
                "buffer_size": 2048,
                "learning_starts": 8,
                "train_freq": 1,
                "gradient_steps": 1,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.zip"
    normalization_path = output_dir / "vecnormalize.pkl"
    replay_path = output_dir / "replay_buffer.pkl"
    manifest_path = output_dir / "training_manifest.json"
    required_checkpoint_paths = [model_path, normalization_path, manifest_path]
    if algorithm == "sac":
        required_checkpoint_paths.append(replay_path)
    if resume and any(path.exists() for path in required_checkpoint_paths) and not all(
        path.exists() for path in required_checkpoint_paths
    ):
        raise RuntimeError("refusing to resume a partial checkpoint")
    resumed = bool(
        resume and all(path.exists() for path in required_checkpoint_paths)
    )
    prior_manifest: dict[str, Any] | None = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if resumed else None
    )
    dual_config = dict(protocol["multiobjective"]["lagrangian"])
    lagrangian = (
        float(prior_manifest["multiobjective"]["final_lagrangian_multiplier"])
        if prior_manifest
        else float(dual_config["initial_multiplier"])
    )
    base_vec = _make_base_vec_env(
        factory,
        seed=seed,
        n_envs=n_envs,
        epsilon_pct=float(epsilon_pct),
        lagrangian_multiplier=lagrangian,
    )
    decision_steps = int(base_vec.envs[0].contract["decision_steps"])
    if any(int(env.contract["decision_steps"]) != decision_steps for env in base_vec.envs):
        base_vec.close()
        raise RuntimeError("all vector environments must use the same decision_steps")
    factory_identity = _factory_identity(factory)
    job_identity = {
        "protocol_sha256": protocol["_sha256"],
        "algorithm": algorithm,
        "seed": int(seed),
        "n_envs": int(n_envs),
        "epsilon_pct": float(epsilon_pct),
        "config_sha256": sha256_json(config),
        "factory": factory_identity,
        "decision_steps": decision_steps,
    }
    if resumed:
        if prior_manifest.get("job_identity") != job_identity:
            base_vec.close()
            raise RuntimeError("resume job identity does not match the existing checkpoint")
        prior_artifacts = prior_manifest["artifacts"]
        _assert_artifact_hash(model_path, prior_artifacts["model"], "model")
        _assert_artifact_hash(
            normalization_path,
            prior_artifacts["normalization"],
            "normalization",
        )
        if algorithm == "sac":
            if prior_artifacts.get("replay") is None:
                base_vec.close()
                raise RuntimeError("SAC resume manifest is missing replay provenance")
            _assert_artifact_hash(replay_path, prior_artifacts["replay"], "replay buffer")
        vec_env = VecNormalize.load(normalization_path, base_vec)
        vec_env.training = True
        model_class = PPO if algorithm == "ppo" else SAC
        model = model_class.load(model_path, env=vec_env, device="cpu")
        if algorithm == "sac":
            model.load_replay_buffer(replay_path)
    else:
        vec_env = VecNormalize(base_vec, norm_obs=True, norm_reward=True, gamma=1.0)
        model = _make_model(algorithm, vec_env, seed, config)
    initial_hashes = model_hashes(model)
    if resumed and prior_manifest:
        if initial_hashes["policy"] != prior_manifest["final_policy_sha256"]:
            vec_env.close()
            raise RuntimeError("loaded policy does not match prior final policy hash")
        if initial_hashes["critic"] != prior_manifest["final_critic_sha256"]:
            vec_env.close()
            raise RuntimeError("loaded critic does not match prior final critic hash")
    start_timesteps = int(model.num_timesteps)
    episode_quantum = int(n_envs) * decision_steps
    rollout_quantum = (
        int(n_envs) * int(config["n_steps"]) if algorithm == "ppo" else episode_quantum
    )
    boundary_quantum = math.lcm(episode_quantum, rollout_quantum)
    if start_timesteps % boundary_quantum:
        vec_env.close()
        raise RuntimeError("checkpoint is not at a complete vector-episode boundary")
    effective_target = int(
        math.ceil(int(target_timesteps) / boundary_quantum) * boundary_quantum
    )
    remaining = max(effective_target - start_timesteps, 0)
    callback = EvidenceCallback(
        n_envs=int(n_envs),
        epsilon_pct=float(epsilon_pct),
        dual_config=dual_config,
        initial_multiplier=lagrangian,
    )
    if remaining:
        model.learn(
            total_timesteps=remaining,
            callback=callback,
            reset_num_timesteps=not resumed,
            progress_bar=False,
        )
    if (
        remaining
        and not callback.last_step_all_terminal
    ):
        vec_env.close()
        raise RuntimeError("refusing to save a checkpoint with incomplete trajectories")
    final_hashes = model_hashes(model)
    model.save(model_path)
    vec_env.save(normalization_path)
    if algorithm == "sac":
        model.save_replay_buffer(replay_path)
    interaction_count = int(model.num_timesteps)
    learning_starts = int(config.get("learning_starts", 0))
    warmup = min(interaction_count, learning_starts) if algorithm == "sac" else 0
    policy_rows = interaction_count - warmup
    replay_provenance = {
        "sources": (
            ["safe_random_feasible_warmup", "randomly_initialized_policy"]
            if algorithm == "sac" and warmup
            else ["randomly_initialized_policy"]
        ),
        "safe_random_feasible_warmup_rows": warmup,
        "randomly_initialized_policy_rows": policy_rows,
        "external_rows": 0,
        "teacher_rows": 0,
        "optimizer_rows": 0,
        "demonstration_rows": 0,
    }
    pure_assertions = {"random_initialization_only": True}
    pure_assertions.update({key: False for key in FORBIDDEN_TRAINING_INPUTS})
    manifest = {
        "schema_version": "ramp-pure-rl-evidence-v1",
        "protocol_id": protocol["protocol"]["id"],
        "protocol_path": protocol["_path"],
        "protocol_sha256": protocol["_sha256"],
        "source_bundle_sha256": source_bundle_hash(),
        "job_identity": job_identity,
        "algorithm": algorithm,
        "seed": seed,
        "stable_baselines3_version": sb3.__version__,
        "semantic_feasible_action": True,
        "raw_redundant_projected_logits": False,
        "gamma": 1.0,
        "credit_assignment": {
            "ppo_gae_lambda": float(config["gae_lambda"]) if algorithm == "ppo" else None,
            "sac_n_steps": int(config["n_steps"]) if algorithm == "sac" else None,
            "actual_terminals": True,
            "bootstrap_across_terminal_or_tail": False,
        },
        "initial_policy_sha256": (
            prior_manifest["initial_policy_sha256"] if prior_manifest else initial_hashes["policy"]
        ),
        "initial_critic_sha256": (
            prior_manifest["initial_critic_sha256"] if prior_manifest else initial_hashes["critic"]
        ),
        "final_policy_sha256": final_hashes["policy"],
        "final_critic_sha256": final_hashes["critic"],
        "interaction_count": interaction_count,
        "requested_target_timesteps": int(target_timesteps),
        "effective_boundary_target_timesteps": effective_target,
        "checkpoint_boundary_quantum": boundary_quantum,
        "interaction_count_this_invocation": callback.interactions,
        "update_count": int(model._n_updates),
        "actual_terminal_count_this_invocation": callback.terminals,
        "replay_provenance": replay_provenance,
        "pure_rl_assertions": pure_assertions,
        "data_split": protocol["data"]["split"],
        "training_data_provenance": {
            "episodes": (
                prior_manifest.get("training_data_provenance", {}).get("episodes", [])
                if prior_manifest
                else []
            )
            + list(callback.episode_records.values()),
            "split": "train",
            "validation_or_test_rows": 0,
            "environment_factory": factory_identity,
        },
        "normalization": _normalization_summary(vec_env),
        "forecast_identity": protocol["data"]["forecast"],
        "multiobjective": {
            "training_epsilon_pct": float(epsilon_pct),
            "primary_energy_budget_pct": float(
                protocol["multiobjective"]["primary_energy_budget_pct"]
            ),
            "epsilon_sensitivity_pct": protocol["multiobjective"]["epsilon_sensitivity_pct"],
            "lagrangian_update": protocol["multiobjective"]["lagrangian"],
            "initial_lagrangian_multiplier_this_invocation": lagrangian,
            "final_lagrangian_multiplier": callback.dual.multiplier,
            "lagrangian_updates": (
                prior_manifest.get("multiobjective", {}).get("lagrangian_updates", [])
                if prior_manifest
                else []
            )
            + callback.dual_updates,
            "selection_data": "validation",
            "test_used_for_selection": False,
        },
        "safety_telemetry": {
            "emergency_count_this_invocation": callback.emergency,
            "semantic_adjustment_l2_sum_this_invocation": callback.semantic_adjustment_sum,
            "non_agent_action_count": callback.non_agent_actions,
        },
        "resume": {
            "resumed": resumed,
            "starting_interaction_count": start_timesteps,
            "prior_final_policy_sha256": prior_manifest.get("final_policy_sha256") if prior_manifest else None,
            "loaded_policy_sha256": initial_hashes["policy"] if resumed else None,
            "loaded_critic_sha256": initial_hashes["critic"] if resumed else None,
            "isolated_job_directory": str(output_dir),
        },
        "artifacts": {
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "normalization": {"path": str(normalization_path), "sha256": sha256_file(normalization_path)},
            "replay": (
                {"path": str(replay_path), "sha256": sha256_file(replay_path)}
                if replay_path.exists()
                else None
            ),
        },
    }
    errors = verify_pure_rl_manifest(manifest)
    if callback.non_agent_actions:
        errors.append("environment reported non-agent actions during training")
    if not manifest["training_data_provenance"]["episodes"]:
        errors.append("no per-window training data provenance was recorded")
    if interaction_count % int(boundary_quantum) != 0:
        errors.append("checkpoint is not on a terminal trajectory boundary")
    manifest["pure_rl_verification"] = {"passed": not errors, "errors": errors}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    vec_env.close()
    if errors:
        raise RuntimeError("pure-RL evidence verification failed: " + "; ".join(errors))
    return manifest
