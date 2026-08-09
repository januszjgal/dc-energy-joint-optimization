"""Frozen v3 replication protocol checks."""

from __future__ import annotations

import unittest

from scripts.run_ramp_rl_v3 import (
    EFFECTIVE_TIMESTEPS,
    EXPECTED_SEEDS,
    NOMINAL_TIMESTEPS,
    load_v3_protocol,
)


class RampRLV3ReplicationTests(unittest.TestCase):
    def test_v3_is_exact_v1_ppo_at_100k(self) -> None:
        protocol = load_v3_protocol()
        self.assertEqual(
            protocol["protocol"]["id"],
            "v6-ramp-pure-rl-v1-replication-earlystop-v3",
        )
        self.assertEqual(protocol["campaign"]["seed_values"], EXPECTED_SEEDS)
        self.assertEqual(
            protocol["training"]["nominal_target_timesteps"],
            NOMINAL_TIMESTEPS,
        )
        self.assertEqual(
            protocol["training"]["effective_boundary_timesteps"],
            EFFECTIVE_TIMESTEPS,
        )
        self.assertEqual(
            protocol["algorithms"]["ppo"],
            {
                "pure_rl": True,
                "gamma": 1.0,
                "gae_lambda": 0.95,
                "n_steps": 512,
                "batch_size": 256,
                "n_epochs": 10,
                "learning_rate": 0.0003,
                "net_arch": [256, 256],
            },
        )

    def test_test_open_is_conditional_and_single_use(self) -> None:
        protocol = load_v3_protocol()
        test = protocol["data"]["split"]["test"]
        self.assertTrue(test["sealed"])
        self.assertEqual(test["open_count_max"], 1)
        self.assertEqual(
            test["open_condition"],
            "all-five-validation-seeds-pass-every-strict-gate",
        )
        self.assertFalse(protocol["campaign"]["automatic_follow_on_protocol"])


if __name__ == "__main__":
    unittest.main()
