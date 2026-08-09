"""Immutable equal-action ensemble evaluation for frozen pure PPO members."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ramp_rl.contract import EnvRequest, RampEnvAdapter, RampEnvironmentFactory
from ramp_rl.evaluation import _aggregate, _episode
from ramp_rl.evidence import sha256_file, verify_pure_rl_manifest


EXPECTED_MEMBER_SEEDS = (2801, 2802, 2803, 2804, 2805)
EXPECTED_ACTION_SHAPE = (13,)
EQUAL_WEIGHT = 0.2


class ObservationNormalizer(Protocol):
    def normalize_obs(self, obs: np.ndarray) -> np.ndarray: ...


class DeterministicPolicy(Protocol):
    def predict(
        self, observation: np.ndarray, *, deterministic: bool
    ) -> tuple[np.ndarray, Any]: ...


class _SpaceOnlyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, observation_space: spaces.Space[Any], action_space: spaces.Space[Any]
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raise RuntimeError("space-only normalization environment cannot be stepped")


@dataclass
class EnsembleMember:
    seed: int
    model: DeterministicPolicy
    normalizer: ObservationNormalizer
    model_sha256: str
    vecnormalize_sha256: str
    training_manifest_sha256: str
    invocation_count: int = 0
    normalized_observation_digest: Any = field(default_factory=hashlib.sha256)
    environment_action_digest: Any = field(default_factory=hashlib.sha256)

    def environment_action(
        self, observation: np.ndarray, action_space: spaces.Box
    ) -> np.ndarray:
        normalized = np.asarray(
            self.normalizer.normalize_obs(np.asarray(observation, dtype=np.float32)),
            dtype=np.float32,
        )
        action, _ = self.model.predict(normalized, deterministic=True)
        environment_action = np.asarray(action, dtype=np.float32).reshape(
            action_space.shape
        )
        if not action_space.contains(environment_action):
            raise RuntimeError(
                f"ensemble member {self.seed} emitted an out-of-bounds action"
            )
        self.invocation_count += 1
        self.normalized_observation_digest.update(
            np.ascontiguousarray(normalized).tobytes()
        )
        self.environment_action_digest.update(
            np.ascontiguousarray(environment_action).tobytes()
        )
        return environment_action


class EqualActionEnsemble:
    """Mean five deterministic environment-space PPO actions with equal weights."""

    def __init__(
        self,
        *,
        members: list[EnsembleMember],
        observation_space: spaces.Space[Any],
        action_space: spaces.Box,
    ) -> None:
        if tuple(member.seed for member in members) != EXPECTED_MEMBER_SEEDS:
            raise ValueError("ensemble requires frozen members 2801-2805 in seed order")
        if action_space.shape != EXPECTED_ACTION_SHAPE:
            raise ValueError("ensemble requires the frozen 13-dimensional action")
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("ensemble requires a Box observation space")
        self.members = members
        self.observation_space = observation_space
        self.action_space = action_space
        self.invocation_count = 0
        self.action_digest = hashlib.sha256()
        self.maximum_absolute_action = 0.0

    @classmethod
    def load(
        cls,
        *,
        bindings: list[dict[str, Any]],
        root: Path,
        observation_space: spaces.Space[Any],
        action_space: spaces.Box,
    ) -> EqualActionEnsemble:
        members: list[EnsembleMember] = []
        for binding in bindings:
            seed = int(binding["seed"])
            manifest_path = root / str(binding["training_manifest_path"])
            model_path = root / str(binding["model_path"])
            normalization_path = root / str(binding["vecnormalize_path"])
            expected_hashes = {
                manifest_path: str(binding["training_manifest_sha256"]),
                model_path: str(binding["model_sha256"]),
                normalization_path: str(binding["vecnormalize_sha256"]),
            }
            for path, expected in expected_hashes.items():
                if not path.is_file():
                    raise FileNotFoundError(f"missing frozen ensemble artifact: {path}")
                if sha256_file(path) != expected:
                    raise RuntimeError(f"frozen ensemble artifact hash mismatch: {path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            errors = verify_pure_rl_manifest(manifest)
            if errors:
                raise RuntimeError(
                    f"seed {seed} pure-RL manifest failed verification: {errors}"
                )
            if int(manifest["seed"]) != seed or manifest["algorithm"] != "ppo":
                raise RuntimeError(f"seed {seed} manifest identity mismatch")
            if (
                manifest["artifacts"]["model"]["sha256"]
                != binding["model_sha256"]
                or manifest["artifacts"]["normalization"]["sha256"]
                != binding["vecnormalize_sha256"]
            ):
                raise RuntimeError(f"seed {seed} manifest artifact binding mismatch")
            model = PPO.load(model_path, device="cpu")
            holder = DummyVecEnv(
                [
                    lambda: _SpaceOnlyEnv(
                        observation_space=observation_space,
                        action_space=action_space,
                    )
                ]
            )
            normalizer = VecNormalize.load(normalization_path, holder)
            normalizer.training = False
            normalizer.norm_reward = False
            normalization = manifest["normalization"]
            if normalization["fit_split"] != "train":
                raise RuntimeError(f"seed {seed} normalization was not train-only")
            if not np.array_equal(
                np.asarray(normalizer.obs_rms.mean),
                np.asarray(normalization["observation_mean"]),
            ) or not np.array_equal(
                np.asarray(normalizer.obs_rms.var),
                np.asarray(normalization["observation_variance"]),
            ):
                raise RuntimeError(f"seed {seed} normalization state mismatch")
            if float(normalizer.obs_rms.count) != float(
                normalization["sample_count"]
            ):
                raise RuntimeError(f"seed {seed} normalization count mismatch")
            members.append(
                EnsembleMember(
                    seed=seed,
                    model=model,
                    normalizer=normalizer,
                    model_sha256=expected_hashes[model_path],
                    vecnormalize_sha256=expected_hashes[normalization_path],
                    training_manifest_sha256=expected_hashes[manifest_path],
                )
            )
        return cls(
            members=members,
            observation_space=observation_space,
            action_space=action_space,
        )

    def predict(self, observation: np.ndarray) -> np.ndarray:
        raw = np.asarray(observation, dtype=np.float32)
        if not self.observation_space.contains(raw):
            raise RuntimeError("ensemble received an invalid raw observation")
        member_actions = [
            member.environment_action(raw, self.action_space)
            for member in self.members
        ]
        mean_action = (
            sum(action.astype(np.float64) for action in member_actions)
            / float(len(member_actions))
        ).astype(np.float32)
        if not self.action_space.contains(mean_action):
            raise RuntimeError("equal environment-action mean is out of bounds")
        self.invocation_count += 1
        self.maximum_absolute_action = max(
            self.maximum_absolute_action, float(np.max(np.abs(mean_action)))
        )
        self.action_digest.update(np.ascontiguousarray(mean_action).tobytes())
        return mean_action

    def audit(self) -> dict[str, Any]:
        return {
            "controller": "deterministic-equal-weight-environment-action-mean",
            "member_seeds": list(EXPECTED_MEMBER_SEEDS),
            "weights": [EQUAL_WEIGHT] * len(EXPECTED_MEMBER_SEEDS),
            "member_count": len(self.members),
            "action_shape": list(self.action_space.shape),
            "action_low": np.asarray(self.action_space.low).tolist(),
            "action_high": np.asarray(self.action_space.high).tolist(),
            "ensemble_invocation_count": self.invocation_count,
            "all_members_invoked_once_per_decision": all(
                member.invocation_count == self.invocation_count
                for member in self.members
            ),
            "deterministic_member_actions": True,
            "environment_space_mean": True,
            "analytic_or_evaluation_actions_used_by_controller": False,
            "trainable_combiner": False,
            "member_selection_or_exclusion": False,
            "maximum_absolute_mean_action": self.maximum_absolute_action,
            "out_of_bounds_action_count": 0,
            "ensemble_action_chain_sha256": self.action_digest.hexdigest(),
            "members": [
                {
                    "seed": member.seed,
                    "weight": EQUAL_WEIGHT,
                    "invocation_count": member.invocation_count,
                    "model_sha256": member.model_sha256,
                    "vecnormalize_sha256": member.vecnormalize_sha256,
                    "training_manifest_sha256": member.training_manifest_sha256,
                    "normalized_observation_chain_sha256": (
                        member.normalized_observation_digest.hexdigest()
                    ),
                    "environment_action_chain_sha256": (
                        member.environment_action_digest.hexdigest()
                    ),
                }
                for member in self.members
            ],
        }

    def close(self) -> None:
        for member in self.members:
            close = getattr(member.normalizer, "close", None)
            if callable(close):
                close()


def _policy_episode(
    *,
    factory: RampEnvironmentFactory,
    split: str,
    seed: int,
    window_id: str,
    controller: EqualActionEnsemble,
) -> dict[str, Any]:
    request = EnvRequest(split=split, seed=seed, window_id=window_id, training=False)
    adapter = RampEnvAdapter(factory(request), request)
    tail_emitted_in_steps = bool(
        adapter.contract.get("terminal_tail_emitted_in_step_metrics", False)
    )
    observation, reset_info = adapter.reset(seed=seed)
    infos: list[dict[str, Any]] = []
    while True:
        action = controller.predict(observation)
        observation, _, terminated, _, info = adapter.step(action)
        if info["action_provenance"] != "agent_semantic":
            raise RuntimeError("ensemble policy path used a non-agent action")
        infos.append(info)
        if terminated:
            break
    adapter.close()
    tail_h1 = (
        [] if tail_emitted_in_steps else infos[-1]["terminal_tail_ramp_h1_adjusted"]
    )
    tail_h3 = (
        [] if tail_emitted_in_steps else infos[-1]["terminal_tail_ramp_h3_adjusted"]
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
    per_market_incremental: dict[str, list[float]] = defaultdict(list)
    for info in infos:
        for market, row in info.get("per_market", {}).items():
            per_market_incremental[str(market)].append(
                sum(
                    weight
                    * float(
                        row["windows"][f"{horizon}h"]["incremental_squared_impact"]
                    )
                    for horizon, weight in ((1, 0.4), (3, 0.6))
                )
            )
    return {
        "window_id": window_id,
        "evaluation_seed": seed,
        "month": int(reset_info["episode_context"]["month"]),
        "day_group": str(reset_info["episode_context"].get("day", window_id)),
        "source_hashes": dict(reset_info["episode_context"]["source_hashes"]),
        "future_realized_features_exposed": bool(
            reset_info["episode_context"]["future_realized_features_exposed"]
        ),
        "ramp_h1": [float(info["ramp_h1_adjusted"]) for info in infos]
        + [float(value) for value in tail_h1],
        "ramp_h3": [float(info["ramp_h3_adjusted"]) for info in infos]
        + [float(value) for value in tail_h3],
        "incremental": [float(info["incremental_ramp_impact"]) for info in infos]
        + [float(value) for value in tail_incremental],
        "per_market_incremental": dict(per_market_incremental),
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
        "semantic_adjustment_l2": sum(
            float(info["semantic_adjustment_l2"]) for info in infos
        )
        + sum(float(value) for value in tail_semantic_adjustment),
        "deferrable_pre_service": sum(
            float(info["deferrable_pre_service"]) for info in infos
        ),
        "ramp_power": sum(
            float(info["dc_power_during_realized_ramp"]) for info in infos
        ),
    }


def evaluate_equal_action_ensemble(
    *,
    factory: RampEnvironmentFactory,
    bindings: list[dict[str, Any]],
    root: Path,
    split: str,
    seed: int,
    windows: list[str],
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("ensemble evaluation split must be validation or test")
    if not windows:
        raise ValueError("ensemble evaluation requires at least one window")
    probe_request = EnvRequest(
        split=split, seed=seed, window_id=windows[0], training=False
    )
    probe = RampEnvAdapter(factory(probe_request), probe_request)
    probe_observation, _ = probe.reset(seed=seed)
    repeatability_controller = EqualActionEnsemble.load(
        bindings=bindings,
        root=root,
        observation_space=probe.observation_space,
        action_space=probe.action_space,
    )
    try:
        first_action = repeatability_controller.predict(probe_observation)
        second_action = repeatability_controller.predict(probe_observation)
        if not np.array_equal(first_action, second_action):
            raise RuntimeError("ensemble deterministic repeatability probe failed")
        repeatability_audit = {
            "same_raw_observation_exact_action_equal": True,
            "raw_observation_sha256": hashlib.sha256(
                np.ascontiguousarray(probe_observation).tobytes()
            ).hexdigest(),
            "environment_action_sha256": hashlib.sha256(
                np.ascontiguousarray(first_action).tobytes()
            ).hexdigest(),
            "all_five_members_invoked_twice": all(
                member.invocation_count == 2
                for member in repeatability_controller.members
            ),
        }
    finally:
        repeatability_controller.close()
    controller = EqualActionEnsemble.load(
        bindings=bindings,
        root=root,
        observation_space=probe.observation_space,
        action_space=probe.action_space,
    )
    probe.close()
    policy: list[dict[str, Any]] = []
    baseline: list[dict[str, Any]] = []
    try:
        for window in windows:
            policy.append(
                _policy_episode(
                    factory=factory,
                    split=split,
                    seed=seed,
                    window_id=window,
                    controller=controller,
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
        summary = _aggregate(policy, baseline, split)
        summary["ensemble_contract"] = {
            "protocol_id": "v6-ramp-pure-rl-equal-action-ensemble-v4",
            "pure_rl_ensemble": True,
            "member_count": 5,
            "equal_weights": [EQUAL_WEIGHT] * 5,
            "environment_space_actions": True,
            "unchanged_constraint_decoder_and_edf": True,
            "raw_evaluation_unchanged": True,
        }
        summary["controller_audit"] = controller.audit()
        summary["deterministic_repeatability_audit"] = repeatability_audit
        expected_decisions = sum(episode["step_count"] for episode in policy)
        if controller.invocation_count != expected_decisions:
            raise RuntimeError("ensemble invocation count does not match policy decisions")
        if not summary["controller_audit"]["all_members_invoked_once_per_decision"]:
            raise RuntimeError("not every ensemble member was invoked for every decision")
        return summary
    finally:
        controller.close()
