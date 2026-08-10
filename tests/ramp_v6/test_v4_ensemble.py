"""Frozen v4 equal-action ensemble contract tests."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from gymnasium import spaces

from ramp_rl.ensemble import (
    EXPECTED_MEMBER_SEEDS,
    EnsembleMember,
    EqualActionEnsemble,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "env" / "protocols" / "v6_pure_ramp_rl_v4.yaml"


class _Normalizer:
    def __init__(self, offset: float) -> None:
        self.offset = float(offset)
        self.inputs: list[np.ndarray] = []

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        self.inputs.append(obs.copy())
        return obs + self.offset


class _Policy:
    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.deterministic_flags: list[bool] = []
        self.observations: list[np.ndarray] = []

    def predict(
        self, observation: np.ndarray, *, deterministic: bool
    ) -> tuple[np.ndarray, Any]:
        self.deterministic_flags.append(deterministic)
        self.observations.append(observation.copy())
        return np.full(13, self.value, dtype=np.float32), None


def _controller() -> tuple[EqualActionEnsemble, list[_Normalizer], list[_Policy]]:
    normalizers = [_Normalizer(index) for index in range(5)]
    policies = [_Policy(index - 2) for index in range(5)]
    members = [
        EnsembleMember(
            seed=seed,
            model=policy,
            normalizer=normalizer,
            model_sha256=f"{index + 1:064x}",
            vecnormalize_sha256=f"{index + 11:064x}",
            training_manifest_sha256=f"{index + 21:064x}",
        )
        for index, (seed, policy, normalizer) in enumerate(
            zip(EXPECTED_MEMBER_SEEDS, policies, normalizers)
        )
    ]
    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
    )
    action_space = spaces.Box(low=-6.0, high=6.0, shape=(13,), dtype=np.float32)
    return (
        EqualActionEnsemble(
            members=members,
            observation_space=observation_space,
            action_space=action_space,
        ),
        normalizers,
        policies,
    )


class EqualActionEnsembleTests(unittest.TestCase):
    def test_every_member_uses_own_normalization_and_equal_weight(self) -> None:
        controller, normalizers, policies = _controller()
        observation = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
        action = controller.predict(observation)
        np.testing.assert_array_equal(action, np.zeros(13, dtype=np.float32))
        for index, (normalizer, policy) in enumerate(
            zip(normalizers, policies)
        ):
            self.assertEqual(len(normalizer.inputs), 1)
            np.testing.assert_array_equal(normalizer.inputs[0], observation)
            np.testing.assert_array_equal(
                policy.observations[0], observation + float(index)
            )
            self.assertEqual(policy.deterministic_flags, [True])
        audit = controller.audit()
        self.assertTrue(audit["all_members_invoked_once_per_decision"])
        self.assertEqual(audit["weights"], [0.2] * 5)
        self.assertFalse(audit["analytic_or_evaluation_actions_used_by_controller"])

    def test_repeatability_is_exact(self) -> None:
        first, _, _ = _controller()
        second, _, _ = _controller()
        observation = np.arange(20, dtype=np.float32) / 10.0
        np.testing.assert_array_equal(
            first.predict(observation), second.predict(observation)
        )
        self.assertEqual(
            first.audit()["ensemble_action_chain_sha256"],
            second.audit()["ensemble_action_chain_sha256"],
        )

    def test_member_action_bounds_fail_closed(self) -> None:
        controller, _, policies = _controller()
        policies[-1].value = 6.01
        with self.assertRaisesRegex(RuntimeError, "out-of-bounds"):
            controller.predict(np.zeros(20, dtype=np.float32))

    def test_exact_five_member_seed_order_is_required(self) -> None:
        controller, _, _ = _controller()
        with self.assertRaisesRegex(ValueError, "2801-2805"):
            EqualActionEnsemble(
                members=list(reversed(controller.members)),
                observation_space=controller.observation_space,
                action_space=controller.action_space,
            )


class FrozenV4ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_controller_is_exact_equal_environment_action_mean(self) -> None:
        controller = self.protocol["controller"]
        self.assertEqual(
            self.protocol["protocol"]["id"],
            "v6-ramp-pure-rl-equal-action-ensemble-v4",
        )
        self.assertEqual(controller["member_seeds"], list(EXPECTED_MEMBER_SEEDS))
        self.assertEqual(controller["weights"], [0.2] * 5)
        self.assertEqual(controller["action_dimensions"], 13)
        self.assertEqual(controller["action_bounds"], [-6.0, 6.0])
        self.assertEqual(controller["action_space"], "environment")
        self.assertTrue(controller["per_member_persisted_observation_normalization"])
        for forbidden in (
            "trainable_combiner",
            "member_selection",
            "member_exclusion",
            "teacher",
            "behavior_cloning",
            "mpc_actions",
            "optimizer_actions",
            "analytic_actions",
            "evaluation_actions",
        ):
            self.assertFalse(controller[forbidden])

    def test_all_five_member_artifacts_are_hash_bound(self) -> None:
        members = self.protocol["frozen_bindings"]["members"]
        self.assertEqual([row["seed"] for row in members], list(EXPECTED_MEMBER_SEEDS))
        for row in members:
            for key in (
                "training_manifest_sha256",
                "model_sha256",
                "vecnormalize_sha256",
            ):
                self.assertRegex(row[key], r"^[0-9a-f]{64}$")

    def test_gates_and_sealed_order_are_not_relaxed(self) -> None:
        gate = self.protocol["success_gate"]
        self.assertEqual(gate["macro_raw_incremental_ramp_impact_lt"], 0.0)
        self.assertEqual(
            gate["every_market_raw_incremental_ramp_impact_lt"], 0.0
        )
        self.assertEqual(gate["primary_da_cost_ratio_max"], 1.02)
        self.assertFalse(gate["post_hoc_tolerance"])
        test = self.protocol["evaluation"]["test"]
        self.assertTrue(test["sealed"])
        self.assertEqual(test["open_count_max"], 1)
        self.assertFalse(test["tune_or_select"])
        robustness = self.protocol["post_selection"]
        self.assertTrue(robustness["enabled_if_and_only_if_sealed_test_passes"])
        self.assertEqual(
            robustness["c_h_overlapping"]["hourly_trace_alignment_epoch_utc"],
            "2025-09-01T00:00:00Z",
        )
        self.assertEqual(
            len(robustness["c_h_overlapping"]["input_bindings"]), 12
        )


if __name__ == "__main__":
    unittest.main()
