"""Recovered-policy loader and evaluator for the immutable V4R ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ramp_rl.contract import EnvRequest, RampEnvAdapter, RampEnvironmentFactory
from ramp_rl.ensemble import (
    EQUAL_WEIGHT,
    EnsembleMember,
    EqualActionEnsemble,
    _SpaceOnlyEnv,
    _policy_episode,
)
from ramp_rl.evaluation import _aggregate, _episode
from ramp_rl.evidence import sha256_file, verify_pure_rl_manifest
from ramp_rl.runner import model_hashes


PROTOCOL_ID = "v6-ramp-pure-rl-recovered-equal-action-ensemble-v4r"


def load_recovered_ensemble(
    *,
    bindings: list[dict[str, Any]],
    root: Path,
    observation_space: spaces.Space[Any],
    action_space: spaces.Box,
) -> EqualActionEnsemble:
    """Load new containers only after proving V3 weight and normalizer equivalence."""
    members: list[EnsembleMember] = []
    for binding in bindings:
        seed = int(binding["seed"])
        manifest_path = root / str(binding["original_training_manifest_path"])
        model_path = root / str(binding["recovered_model_path"])
        normalization_path = root / str(binding["recovered_vecnormalize_path"])
        expected_hashes = {
            manifest_path: str(binding["original_training_manifest_sha256"]),
            model_path: str(binding["recovered_model_sha256"]),
            normalization_path: str(binding["recovered_vecnormalize_sha256"]),
        }
        for path, expected in expected_hashes.items():
            if not path.is_file():
                raise FileNotFoundError(f"missing V4R artifact: {path}")
            if sha256_file(path) != expected:
                raise RuntimeError(f"V4R artifact hash mismatch: {path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        errors = verify_pure_rl_manifest(manifest)
        if errors:
            raise RuntimeError(f"seed {seed} pure-RL manifest failed: {errors}")
        if int(manifest["seed"]) != seed or manifest["algorithm"] != "ppo":
            raise RuntimeError(f"seed {seed} original manifest identity mismatch")
        if (
            manifest["artifacts"]["model"]["sha256"]
            != binding["original_model_sha256"]
        ):
            raise RuntimeError(f"seed {seed} original model binding mismatch")
        if (
            manifest["artifacts"]["normalization"]["sha256"]
            != binding["recovered_vecnormalize_sha256"]
        ):
            raise RuntimeError(f"seed {seed} normalization is not byte-identical to V3")
        for field in (
            "initial_policy_sha256",
            "initial_critic_sha256",
            "final_policy_sha256",
            "final_critic_sha256",
            "interaction_count",
            "update_count",
        ):
            if manifest[field] != binding[field]:
                raise RuntimeError(f"seed {seed} original {field} binding mismatch")

        model = PPO.load(model_path, device="cpu")
        loaded_hashes = model_hashes(model)
        if loaded_hashes["policy"] != binding["final_policy_sha256"]:
            raise RuntimeError(f"seed {seed} recovered policy weights differ from V3")
        if loaded_hashes["critic"] != binding["final_critic_sha256"]:
            raise RuntimeError(f"seed {seed} recovered critic weights differ from V3")

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
        if float(normalizer.obs_rms.count) != float(normalization["sample_count"]):
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
    return EqualActionEnsemble(
        members=members,
        observation_space=observation_space,
        action_space=action_space,
    )


def evaluate_recovered_equal_action_ensemble(
    *,
    factory: RampEnvironmentFactory,
    bindings: list[dict[str, Any]],
    root: Path,
    split: str,
    seed: int,
    windows: list[str],
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("V4R evaluation split must be validation or test")
    if not windows:
        raise ValueError("V4R evaluation requires at least one window")
    probe_request = EnvRequest(
        split=split, seed=seed, window_id=windows[0], training=False
    )
    probe = RampEnvAdapter(factory(probe_request), probe_request)
    probe_observation, _ = probe.reset(seed=seed)
    repeatability_controller = load_recovered_ensemble(
        bindings=bindings,
        root=root,
        observation_space=probe.observation_space,
        action_space=probe.action_space,
    )
    try:
        first_action = repeatability_controller.predict(probe_observation)
        second_action = repeatability_controller.predict(probe_observation)
        if not np.array_equal(first_action, second_action):
            raise RuntimeError("V4R deterministic repeatability probe failed")
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

    controller = load_recovered_ensemble(
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
            "protocol_id": PROTOCOL_ID,
            "recovered_repackaged_pure_rl_policies": True,
            "weight_equivalent_to_v3": True,
            "new_binary_identity": True,
            "original_v3_artifact_reuse": False,
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
            raise RuntimeError("V4R invocation count does not match policy decisions")
        if not summary["controller_audit"]["all_members_invoked_once_per_decision"]:
            raise RuntimeError("not every V4R member was invoked for every decision")
        return summary
    finally:
        controller.close()
