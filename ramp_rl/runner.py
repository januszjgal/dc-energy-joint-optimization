"""PPO training runner for the active four-market experiment."""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ramp_rl.contract import (
    EnvRequest, RampEnvAdapter, RampEnvironmentFactory, environment_identity,
)

ROOT = Path(__file__).resolve().parent.parent


def _repo_path(path: Path) -> str:
    """Render a path relative to the repository root, POSIX style.

    Recorded artifact paths must not embed the absolute working directory:
    runs executed from a worktree would otherwise bake that checkout's location
    into committed evidence. Falls back to the absolute path when the target
    lies outside the repository.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


EXPECTED_SB3_VERSION = "2.9.0"
LEARNING_CURVE_COLUMNS = (
    "interaction_count",
    "mean_raw_joint_reward",
    "mean_raw_ramp_reward",
    "mean_raw_peak_reward",
    "mean_raw_ramp_squared",
    "mean_raw_peak_increment",
    "mean_raw_incremental_ramp_impact",
    "episode_count",
    "mean_completed_episode_return",
    "elapsed_seconds",
)
MILESTONE_TARGETS = (110_592, 500_000, 1_000_000, 1_500_000, 2_000_000)
TRAINING_IDENTITY_FIELDS = (
    "seed",
    "requested_interactions",
    "effective_interactions",
    "n_envs",
    "ppo_config",
    "safe_quantum",
    "environment_identity",
)
DEFAULT_CHECKPOINT_ROLLOUTS = 25


def configure_single_thread_runtime() -> None:
    """Avoid CPU oversubscription when several campaign workers are active."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits configuring inter-op threads only before parallel work.
        pass


def load_factory(specification: str) -> RampEnvironmentFactory:
    module_name, separator, attribute = specification.partition(":")
    if not separator:
        raise ValueError("environment factory must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"{specification} is not callable")
    return factory


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def safe_boundary_quantum(*, n_envs: int, n_steps: int, checkpoint_rollouts: int) -> int:
    """Return the interaction count between crash-safe checkpoints.

    Episodes are continuous months of unequal length, so checkpoints are
    aligned to PPO rollout boundaries rather than to episode ends. A resumed
    run starts fresh episodes; the partial episodes in flight at the
    checkpoint were already used for learning and are simply not continued.
    """
    if n_envs <= 0 or n_steps <= 0 or checkpoint_rollouts <= 0:
        raise ValueError("n_envs, n_steps, and checkpoint_rollouts must be positive")
    return n_envs * n_steps * checkpoint_rollouts


def milestone_interactions(
    requested: tuple[int, ...], *, safe_quantum: int
) -> dict[int, int]:
    """Map requested snapshots to their first crash-safe interaction boundary."""
    if safe_quantum <= 0:
        raise ValueError("safe_quantum must be positive")
    return {
        target: math.ceil(target / safe_quantum) * safe_quantum
        for target in requested
    }


def _valid_curve_rows(path: Path, interaction_limit: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LEARNING_CURVE_COLUMNS:
            raise ValueError(f"unexpected learning curve schema: {path}")
        rows: list[dict[str, str]] = []
        previous = -1
        for row in reader:
            try:
                interaction = int(row["interaction_count"])
                if (
                    interaction <= previous
                    or interaction > interaction_limit
                    or any(row.get(column) in (None, "") for column in LEARNING_CURVE_COLUMNS)
                ):
                    break
                for column in LEARNING_CURVE_COLUMNS[1:]:
                    float(row[column])
            except (TypeError, ValueError):
                break
            rows.append({column: row[column] for column in LEARNING_CURVE_COLUMNS})
            previous = interaction
    return rows


def prepare_learning_curve(path: Path, *, interaction_limit: int) -> None:
    """Keep only durable curve rows before opening the curve for append."""
    rows = _valid_curve_rows(path, interaction_limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEARNING_CURVE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class LearningCurveWriter:
    """Append durable raw-environment metrics after every PPO rollout."""

    def __init__(self, path: Path, *, interaction_limit: int = 0):
        self.path = path
        prepare_learning_curve(path, interaction_limit=interaction_limit)
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=LEARNING_CURVE_COLUMNS)

    def write(self, row: dict[str, float | int]) -> None:
        self._writer.writerow({key: row[key] for key in LEARNING_CURVE_COLUMNS})
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()


def _checkpoint_state(checkpoint_dir: Path) -> dict[str, Any] | None:
    state_path = checkpoint_dir / "state.json"
    if not (
        state_path.exists()
        and (checkpoint_dir / "model.zip").exists()
        and (checkpoint_dir / "vecnormalize.pkl").exists()
    ):
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(state["interaction_count"]) <= 0:
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return state


def select_latest_checkpoint(output_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    """Select the newest complete latest checkpoint, including interrupted swaps."""
    candidates = (
        output_dir / "latest",
        output_dir / "latest.pending",
        output_dir / "latest.previous",
    )
    complete = [
        (path, state)
        for path in candidates
        if (state := _checkpoint_state(path)) is not None
    ]
    if not complete:
        return None
    return max(complete, key=lambda item: int(item[1]["interaction_count"]))


def _save_checkpoint(
    model: PPO, vec_env: VecNormalize, checkpoint_dir: Path, state: dict[str, Any]
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "model.zip"
    normalization_path = checkpoint_dir / "vecnormalize.pkl"
    model.save(model_path)
    vec_env.save(normalization_path)
    for path in (model_path, normalization_path):
        with path.open("rb+") as handle:
            os.fsync(handle.fileno())
    _write_json(checkpoint_dir / "state.json", state)


def _remove_directory_with_retry(path: Path, attempts: int = 20) -> None:
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25)


def _replace_directory_with_retry(
    source: Path, target: Path, attempts: int = 20
) -> None:
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25)


def save_latest_checkpoint(
    model: PPO, vec_env: VecNormalize, output_dir: Path, state: dict[str, Any]
) -> None:
    """Publish a complete checkpoint atomically enough to survive a process crash."""
    latest = output_dir / "latest"
    pending = output_dir / "latest.pending"
    previous = output_dir / "latest.previous"
    _remove_directory_with_retry(pending)
    _save_checkpoint(model, vec_env, pending, state)
    if latest.exists():
        _remove_directory_with_retry(previous)
        _replace_directory_with_retry(latest, previous)
    _replace_directory_with_retry(pending, latest)
    _remove_directory_with_retry(previous)


class TrainingCallback(BaseCallback):
    def __init__(
        self,
        *,
        curve_path: Path,
        progress_path: Path,
        target_timesteps: int,
        output_dir: Path,
        safe_quantum: int,
        n_steps: int,
        resumed_from_interactions: int,
        environment: dict[str, Any],
        ppo_config: dict[str, Any],
        seed: int,
    ) -> None:
        super().__init__()
        self.curve_writer = LearningCurveWriter(
            curve_path, interaction_limit=resumed_from_interactions
        )
        self.progress_path = progress_path
        self.target_timesteps = target_timesteps
        self.output_dir = output_dir
        self.safe_quantum = safe_quantum
        self.n_steps = n_steps
        self.resumed_from_interactions = resumed_from_interactions
        self.environment_identity = environment
        self.ppo_config = ppo_config
        self.seed = seed
        self.milestone_map = milestone_interactions(
            MILESTONE_TARGETS, safe_quantum=safe_quantum
        )
        self.milestone_dir = output_dir / "milestones"
        self.milestone_index_path = self.milestone_dir / "index.json"
        self.interactions = 0
        self.terminals = 0
        self._raw_rewards: list[float] = []
        self._ramp_rewards: list[float] = []
        self._peak_rewards: list[float] = []
        self._ramp_squared: list[float] = []
        self._peak_increments: list[float] = []
        self._raw_impacts: list[float] = []
        self._episode_returns: list[float] = []
        self._per_env_returns: list[float] = []
        self._saved_milestones: set[int] = self._existing_milestones()
        self._safe_checkpoint_pending = False
        self._started = 0.0

    def _existing_milestones(self) -> set[int]:
        try:
            index = json.loads(self.milestone_index_path.read_text(encoding="utf-8"))
            saved = index.get("milestones", {})
            return {
                int(target)
                for target, details in saved.items()
                if int(details["actual_interactions"]) == self.milestone_map[int(target)]
            }
        except (OSError, ValueError, KeyError, TypeError):
            return set()

    def _checkpoint_state(self, interaction_count: int) -> dict[str, Any]:
        return {
            "interaction_count": interaction_count,
            "requested_interactions": self.target_timesteps,
            "safe_quantum": self.safe_quantum,
            "n_envs": self.training_env.num_envs,
            "n_steps": self.n_steps,
            "environment_identity": self.environment_identity,
            "ppo_config": self.ppo_config,
            "seed": self.seed,
        }

    def _save_milestones(self, interaction_count: int) -> None:
        due = [
            target
            for target, actual in self.milestone_map.items()
            if actual <= interaction_count and target not in self._saved_milestones
        ]
        if not due:
            return
        vec_env = self.model.get_env()
        if not isinstance(vec_env, VecNormalize):
            raise RuntimeError("PPO environment must be VecNormalize")
        self.milestone_dir.mkdir(parents=True, exist_ok=True)
        try:
            index = json.loads(self.milestone_index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            index = {"milestones": {}}
        for target in due:
            actual = self.milestone_map[target]
            snapshot = self.milestone_dir / f"requested-{target}-at-{actual}"
            state = self._checkpoint_state(actual)
            state.update(
                {
                    "requested_milestone_interactions": target,
                    "actual_milestone_interactions": actual,
                }
            )
            _save_checkpoint(self.model, vec_env, snapshot, state)
            index["milestones"][str(target)] = {
                "requested_interactions": target,
                "actual_interactions": actual,
                "path": _repo_path(snapshot),
            }
            self._saved_milestones.add(target)
        _write_json(self.milestone_index_path, index)

    def save_safe_boundary(self, interaction_count: int) -> None:
        if interaction_count % self.safe_quantum:
            raise RuntimeError("checkpoint must be written at a safe boundary")
        vec_env = self.model.get_env()
        if not isinstance(vec_env, VecNormalize):
            raise RuntimeError("PPO environment must be VecNormalize")
        save_latest_checkpoint(
            self.model, vec_env, self.output_dir, self._checkpoint_state(interaction_count)
        )
        self._save_milestones(interaction_count)

    def _on_training_start(self) -> None:
        self._started = time.perf_counter()
        self._per_env_returns = [0.0] * self.training_env.num_envs
        _write_json(
            self.progress_path,
            {
                "status": "running",
                "interaction_count": int(self.num_timesteps),
                "target_interactions": self.target_timesteps,
                "resumed_from_interactions": self.resumed_from_interactions,
                "elapsed_seconds": 0.0,
            },
        )

    def _on_rollout_start(self) -> None:
        if self._safe_checkpoint_pending:
            self.save_safe_boundary(int(self.num_timesteps))
            self._safe_checkpoint_pending = False

    def _on_step(self) -> bool:
        infos = list(self.locals.get("infos", []))
        self.interactions += len(infos)
        for index, info in enumerate(infos):
            raw_reward = float(info["scalar_reward"])
            self._raw_rewards.append(raw_reward)
            self._ramp_rewards.append(float(info["ramp_reward"]))
            self._peak_rewards.append(float(info["peak_reward"]))
            self._ramp_squared.append(float(info["ramp_squared_score"]))
            self._peak_increments.append(float(info["peak_normalized_increment"]))
            self._raw_impacts.append(float(info["incremental_ramp_impact"]))
            self._per_env_returns[index] += raw_reward
            if bool(info.get("actual_terminal", False)):
                self._episode_returns.append(self._per_env_returns[index])
                self._per_env_returns[index] = 0.0
                self.terminals += 1
        return True

    def _on_rollout_end(self) -> None:
        elapsed = time.perf_counter() - self._started
        interaction_count = int(self.num_timesteps)
        row = {
            "interaction_count": interaction_count,
            "mean_raw_joint_reward": float(np.mean(self._raw_rewards)),
            "mean_raw_ramp_reward": float(np.mean(self._ramp_rewards)),
            "mean_raw_peak_reward": float(np.mean(self._peak_rewards)),
            "mean_raw_ramp_squared": float(np.mean(self._ramp_squared)),
            "mean_raw_peak_increment": float(np.mean(self._peak_increments)),
            "mean_raw_incremental_ramp_impact": float(np.mean(self._raw_impacts)),
            "episode_count": len(self._episode_returns),
            "mean_completed_episode_return": (
                float(np.mean(self._episode_returns)) if self._episode_returns else float("nan")
            ),
            "elapsed_seconds": elapsed,
        }
        self.curve_writer.write(row)
        _write_json(
            self.progress_path,
            {
                "status": "running",
                "interaction_count": interaction_count,
                "target_interactions": self.target_timesteps,
                "resumed_from_interactions": self.resumed_from_interactions,
                "episode_count": self.terminals,
                "elapsed_seconds": elapsed,
            },
        )
        self._raw_rewards.clear()
        self._ramp_rewards.clear()
        self._peak_rewards.clear()
        self._ramp_squared.clear()
        self._peak_increments.clear()
        self._raw_impacts.clear()
        self._episode_returns.clear()
        self._safe_checkpoint_pending = interaction_count % self.safe_quantum == 0

    def close(self) -> None:
        self.curve_writer.close()


def _make_vec_env(factory: RampEnvironmentFactory, *, seed: int, n_envs: int) -> DummyVecEnv:
    env_fns: list[Callable[[], gym.Env]] = []
    for rank in range(n_envs):
        request = EnvRequest(split="train", seed=seed + rank, rank=rank, training=True)

        def make(request: EnvRequest = request) -> gym.Env:
            return RampEnvAdapter(factory(request), request)

        env_fns.append(make)
    return DummyVecEnv(env_fns)


def _training_identity(
    *,
    seed: int,
    target_timesteps: int,
    effective_interactions: int,
    n_envs: int,
    ppo_config: dict[str, Any],
    safe_quantum: int,
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Return the plain fields that identify a completed training run."""
    return {
        "seed": int(seed),
        "requested_interactions": int(target_timesteps),
        "effective_interactions": int(effective_interactions),
        "n_envs": int(n_envs),
        "ppo_config": ppo_config,
        "safe_quantum": int(safe_quantum),
        "environment_identity": environment,
    }


def training_identity(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract the plain fields that identify a completed training run."""
    return {field: summary[field] for field in TRAINING_IDENTITY_FIELDS}


def validate_completed_training_summary(
    summary_path: Path, expected_identity: dict[str, Any]
) -> dict[str, Any]:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid completed training summary: {summary_path}") from error
    if not isinstance(summary, dict):
        raise RuntimeError(f"invalid completed training summary: {summary_path}")
    for field, expected in expected_identity.items():
        actual = summary.get(field)
        if isinstance(expected, int) and (isinstance(actual, bool) or not isinstance(actual, int)):
            raise RuntimeError(
                f"completed training summary has invalid {field}: {summary_path}"
            )
        if actual != expected:
            raise RuntimeError(
                f"completed training summary does not match current {field}: {summary_path}"
            )
    return summary


def run_training(
    *,
    factory: RampEnvironmentFactory,
    protocol: dict[str, Any],
    seed: int,
    target_timesteps: int,
    output_dir: Path,
    curve_path: Path | None = None,
    progress_path: Path | None = None,
    n_envs: int | None = None,
    fixture_profile: bool = False,
) -> dict[str, Any]:
    """Train one PPO policy and persist raw learning metrics at rollout boundaries."""
    configure_single_thread_runtime()
    if sb3.__version__ != EXPECTED_SB3_VERSION:
        raise RuntimeError(f"expected SB3 {EXPECTED_SB3_VERSION}, found {sb3.__version__}")
    if target_timesteps <= 0:
        raise ValueError("target timesteps must be positive")
    training_section = protocol["training"]
    configured_envs = int(training_section["vectorized_environments"])
    n_envs = (2 if fixture_profile else configured_envs) if n_envs is None else n_envs
    if n_envs <= 0 or (not fixture_profile and n_envs != configured_envs):
        raise ValueError("n_envs must match the protocol")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = dict(training_section["ppo"])
    checkpoint_rollouts = int(
        training_section.get("checkpoint_rollouts", DEFAULT_CHECKPOINT_ROLLOUTS)
    )
    if fixture_profile:
        config.update({"net_arch": [32, 32], "learning_rate": 5e-4, "batch_size": 16, "n_steps": 16, "n_epochs": 2})
        config.setdefault("gamma", 0.99)
        checkpoint_rollouts = 3
    if "gamma" not in config:
        raise ValueError("protocol ppo config must declare gamma")
    gamma = float(config["gamma"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.zip"
    normalization_path = output_dir / "vecnormalize.pkl"
    summary_path = output_dir / "training_summary.json"
    curve_path = curve_path or output_dir / "learning_curve.csv"
    progress_path = progress_path or output_dir / "progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    base_vec = _make_vec_env(factory, seed=seed, n_envs=n_envs)
    callback: TrainingCallback | None = None
    vec_env: VecNormalize | None = None
    try:
        contracts = base_vec.get_attr("contract")
        environment = environment_identity(contracts[0])
        if any(environment_identity(contract) != environment for contract in contracts[1:]):
            raise ValueError("training environments have different objective or input identities")
        safe_quantum = safe_boundary_quantum(
            n_envs=n_envs, n_steps=int(config["n_steps"]), checkpoint_rollouts=checkpoint_rollouts
        )
        effective_interactions = math.ceil(target_timesteps / safe_quantum) * safe_quantum
        identity = _training_identity(
            seed=seed,
            target_timesteps=target_timesteps,
            effective_interactions=effective_interactions,
            n_envs=n_envs,
            ppo_config=config,
            safe_quantum=safe_quantum,
            environment=environment,
        )
        completed_paths = (model_path, normalization_path, summary_path)
        if all(path.exists() for path in completed_paths):
            return validate_completed_training_summary(summary_path, identity)
        if summary_path.exists():
            raise RuntimeError(
                "completed training summary is missing required artifacts: "
                f"{summary_path}"
            )
        selected_checkpoint = select_latest_checkpoint(output_dir)
        resumed_from_interactions = 0
        if selected_checkpoint is None:
            # A crash before the first safe checkpoint leaves no resumable model state.
            # The curve is rebuilt from zero below, so the seed remains retryable.
            vec_env = VecNormalize(base_vec, norm_obs=True, norm_reward=True, gamma=gamma)
            model = PPO(
                "MlpPolicy", vec_env, gamma=gamma, seed=seed, device="cpu", verbose=0,
                policy_kwargs={"net_arch": list(config["net_arch"])},
                n_steps=int(config["n_steps"]), batch_size=int(config["batch_size"]),
                n_epochs=int(config["n_epochs"]), gae_lambda=float(config["gae_lambda"]),
                learning_rate=float(config["learning_rate"]),
            )
        else:
            checkpoint_dir, state = selected_checkpoint
            resumed_from_interactions = int(state["interaction_count"])
            expected_geometry = {
                "safe_quantum": safe_quantum,
                "n_envs": n_envs,
                "n_steps": int(config["n_steps"]),
            }
            if any(int(state.get(key, -1)) != value for key, value in expected_geometry.items()):
                raise RuntimeError(f"latest checkpoint geometry does not match current training: {checkpoint_dir}")
            for key, expected in (
                ("environment_identity", environment), ("ppo_config", config), ("seed", seed)
            ):
                if state.get(key) != expected:
                    raise RuntimeError(
                        f"latest checkpoint {key} does not match current training: {checkpoint_dir}"
                    )
            if resumed_from_interactions > effective_interactions:
                raise RuntimeError("latest checkpoint exceeds requested training horizon")
            vec_env = VecNormalize.load(checkpoint_dir / "vecnormalize.pkl", base_vec)
            model = PPO.load(checkpoint_dir / "model.zip", env=vec_env, device="cpu")
            if int(model.num_timesteps) != resumed_from_interactions:
                raise RuntimeError("latest checkpoint model and state interaction counts differ")
            # The saved observation belongs to the retired vector environments.
            # Resuming starts fresh episodes; the partial months in flight at the
            # checkpoint were already learned from and are not continued.
            model._last_obs = None
        callback = TrainingCallback(
            curve_path=curve_path,
            progress_path=progress_path,
            target_timesteps=target_timesteps,
            output_dir=output_dir,
            safe_quantum=safe_quantum,
            n_steps=int(config["n_steps"]),
            resumed_from_interactions=resumed_from_interactions,
            environment=environment,
            ppo_config=config,
            seed=seed,
        )
        remaining_interactions = effective_interactions - resumed_from_interactions
        if remaining_interactions:
            model.learn(
                total_timesteps=remaining_interactions,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            callback.save_safe_boundary(int(model.num_timesteps))
        model.save(model_path)
        vec_env.save(normalization_path)
        summary = {
            **_training_identity(
                seed=seed,
                target_timesteps=target_timesteps,
                effective_interactions=int(model.num_timesteps),
                n_envs=n_envs,
                ppo_config=config,
                safe_quantum=safe_quantum,
                environment=environment,
            ),
            "resumed_from_interactions": resumed_from_interactions,
            "safe_boundary_interactions": safe_quantum,
            "update_count": int(model._n_updates),
            "learning_curve_path": _repo_path(curve_path),
        }
        _write_json(summary_path, summary)
        _write_json(
            progress_path,
            {
                "status": "training_complete",
                "interaction_count": int(model.num_timesteps),
                "target_interactions": target_timesteps,
                "resumed_from_interactions": resumed_from_interactions,
                "episode_count": callback.terminals,
            },
        )
        return summary
    finally:
        if callback is not None:
            callback.close()
        if vec_env is not None:
            vec_env.close()
        else:
            base_vec.close()
