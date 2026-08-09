"""Contract tests for the distinct recovered-policy V4R protocol."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from ramp_rl.ensemble import EXPECTED_MEMBER_SEEDS
from ramp_rl.recovered_ensemble import PROTOCOL_ID


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "env" / "protocols" / "v6_pure_ramp_rl_v4r.yaml"


class RecoveredV4RProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_v4r_has_new_identity_and_disclaims_original_artifacts(self) -> None:
        self.assertEqual(self.protocol["protocol"]["id"], PROTOCOL_ID)
        self.assertFalse(
            self.protocol["protocol"]["blocked_original_v4_identity_reused"]
        )
        identity = self.protocol["artifact_identity"]
        self.assertTrue(identity["weight_equivalent_to_v3"])
        self.assertTrue(identity["new_binary_identity"])
        self.assertFalse(identity["original_v3_model_container_reuse"])
        self.assertFalse(identity["original_v4_identity_claimed"])
        self.assertFalse(identity["performance_claim_before_evaluation"])

    def test_controller_is_exact_equal_environment_action_mean(self) -> None:
        controller = self.protocol["controller"]
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
            "member_weighting_from_validation",
            "teacher",
            "behavior_cloning",
            "mpc_actions",
            "optimizer_actions",
            "analytic_actions",
            "evaluation_actions",
        ):
            self.assertFalse(controller[forbidden])

    def test_all_recovered_members_bind_container_and_weight_hashes(self) -> None:
        members = self.protocol["frozen_bindings"]["members"]
        self.assertEqual([row["seed"] for row in members], list(EXPECTED_MEMBER_SEEDS))
        for row in members:
            self.assertNotEqual(
                row["recovered_model_sha256"], row["original_model_sha256"]
            )
            for key in (
                "original_training_manifest_sha256",
                "original_model_sha256",
                "recovered_training_manifest_sha256",
                "recovered_model_sha256",
                "recovered_vecnormalize_sha256",
                "initial_policy_sha256",
                "initial_critic_sha256",
                "final_policy_sha256",
                "final_critic_sha256",
            ):
                self.assertRegex(row[key], r"^[0-9a-f]{64}$")

    def test_gates_and_sealed_order_are_unchanged(self) -> None:
        gate = self.protocol["success_gate"]
        self.assertEqual(gate["macro_raw_incremental_ramp_impact_lt"], 0.0)
        self.assertEqual(
            gate["every_market_raw_incremental_ramp_impact_lt"], 0.0
        )
        self.assertEqual(gate["primary_da_cost_ratio_max"], 1.02)
        self.assertEqual(gate["emergency_feasibility_rate_lt"], 0.01)
        self.assertFalse(gate["post_hoc_tolerance"])
        test = self.protocol["evaluation"]["test"]
        self.assertTrue(test["sealed"])
        self.assertEqual(test["open_count_max"], 1)
        self.assertFalse(test["tune_or_select"])


if __name__ == "__main__":
    unittest.main()
