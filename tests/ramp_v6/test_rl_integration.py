"""Integration checks for the ramp-core pure-RL factory boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from env.ramp_v6.factory import (
    MissingEnergyModelV3PanelError,
    make_energy_model_v3_env,
    make_fixture_env,
)
from ramp_rl.campaign import assert_long_campaign_ready
from ramp_rl.contract import (
    SEMANTIC_ACTION_ID,
    EnvRequest,
    RampContractError,
    RampEnvAdapter,
)
from ramp_rl.evaluation import _episode
from ramp_rl.fixture_env import make_fixture_env as make_bundled_tail_fixture
from ramp_rl.runner import EvidenceCallback
from ramp_rl.schema import load_protocol


class RampRLIntegrationTests(unittest.TestCase):
    def test_real_six_market_factory_satisfies_contract(self) -> None:
        request = EnvRequest(split="train", seed=2601, training=True)
        adapter = RampEnvAdapter(make_fixture_env(request), request)
        observation, reset_info = adapter.reset(seed=request.seed)
        self.assertEqual(adapter.action_space.shape, (13,))
        np.testing.assert_array_equal(adapter.action_space.low, -6.0)
        np.testing.assert_array_equal(adapter.action_space.high, 6.0)
        self.assertEqual(
            adapter.contract["semantic_action_id"], SEMANTIC_ACTION_ID
        )
        self.assertEqual(adapter.contract["interval_minutes"], 60)
        self.assertEqual(adapter.contract["decision_steps"], 9)
        self.assertEqual(adapter.contract["active_arrival_steps"], 6)
        self.assertEqual(reset_info["episode_context"]["split"], "train")
        self.assertEqual(len(reset_info["episode_context"]["source_hashes"]), 3)

        terminal = None
        for _ in range(adapter.contract["decision_steps"]):
            observation, _, terminated, truncated, info = adapter.step(
                np.zeros(adapter.action_space.shape, dtype=np.float32)
            )
            self.assertFalse(truncated)
            terminal = info
        self.assertTrue(terminated)
        self.assertTrue(terminal["tail_complete"])
        self.assertEqual(len(terminal["terminal_tail_ramp_h1_adjusted"]), 3)
        self.assertEqual(len(terminal["terminal_tail_energy_cost"]), 3)
        self.assertEqual(len(terminal["terminal_tail_batch_unfinished"]), 3)
        self.assertEqual(terminal["terminal_work"], 0.0)
        self.assertEqual(terminal["service_unserved"], 0.0)
        self.assertEqual(terminal["batch_expired"], 0.0)
        self.assertEqual(terminal["certificate_violations"], 0)
        adapter.close()

    def test_evaluation_controller_is_guarded_from_training(self) -> None:
        request = EnvRequest(split="train", seed=1, training=True)
        adapter = RampEnvAdapter(make_fixture_env(request), request)
        adapter.reset(seed=1)
        with self.assertRaises(RampContractError):
            adapter.evaluation_action("status_quo")
        adapter.close()

    def test_status_quo_cost_uses_an_independent_shadow_queue(self) -> None:
        request = EnvRequest(split="train", seed=5, training=True)
        low = make_fixture_env(request)
        high = make_fixture_env(request)
        low.reset(seed=5)
        high.reset(seed=5)
        low_action = np.full(low.action_space.shape, -6.0, dtype=np.float32)
        high_action = np.full(high.action_space.shape, 6.0, dtype=np.float32)
        for _ in range(low.action_steps):
            _, _, low_done, _, low_info = low.step(low_action)
            _, _, high_done, _, high_info = high.step(high_action)
            self.assertAlmostEqual(
                low_info["status_quo_energy_cost"],
                high_info["status_quo_energy_cost"],
            )
        self.assertTrue(low_done and high_done)
        low.close()
        high.close()

    def test_consolidated_protocol_has_only_real_panel_blocker(self) -> None:
        protocol = load_protocol()
        self.assertEqual(
            protocol["environment_protocol"]["semantic_action_id"],
            SEMANTIC_ACTION_ID,
        )
        self.assertEqual(protocol["data"]["interval_minutes"], 60)
        self.assertEqual(
            protocol["campaign"]["final_campaign_blocked_until"],
            ["energy-model-v3-ramp-panel"],
        )
        with self.assertRaisesRegex(RuntimeError, "energy_model_v3_ramp_panel"):
            assert_long_campaign_ready({})
        assert_long_campaign_ready({"energy_model_v3_ramp_panel": True})

    def test_bundled_tail_cost_is_included_once(self) -> None:
        window = "m-07-sealed-0000"
        request = EnvRequest(
            split="validation", seed=77, window_id=window, training=False
        )
        adapter = RampEnvAdapter(make_bundled_tail_fixture(request), request)
        adapter.reset(seed=77)
        infos = []
        while True:
            _, _, terminated, _, info = adapter.step(
                adapter.evaluation_action("status_quo")
            )
            infos.append(info)
            if terminated:
                break
        expected = sum(float(info["energy_cost"]) for info in infos) + sum(
            float(value)
            for value in infos[-1]["terminal_tail_energy_cost"]
        )
        adapter.close()
        episode = _episode(
            factory=make_bundled_tail_fixture,
            split="validation",
            seed=77,
            window_id=window,
            model=None,
            normalization_path=None,
        )
        self.assertAlmostEqual(episode["energy_cost"], expected)

    def test_training_dual_includes_bundled_tail_metrics(self) -> None:
        class VecStub:
            def env_method(self, name, value):
                self.update = (name, value)

        class ModelStub:
            def __init__(self):
                self.env = VecStub()

            def get_env(self):
                return self.env

        callback = EvidenceCallback(
            n_envs=1,
            epsilon_pct=2.0,
            dual_config={"learning_rate": 0.01, "maximum": 100.0},
            initial_multiplier=0.0,
        )
        callback.model = ModelStub()
        callback.locals = {
            "infos": [
                {
                    "actual_terminal": True,
                    "energy_cost": 1.0,
                    "status_quo_energy_cost": 1.0,
                    "emergency_feasibility": False,
                    "semantic_adjustment_l2": 0.0,
                    "action_provenance": "agent_semantic",
                    "ramp_terminal_tail_emitted_in_steps": False,
                    "terminal_tail_energy_cost": [2.0, 3.0],
                    "terminal_tail_status_quo_energy_cost": [1.0, 1.0],
                    "terminal_tail_emergency_feasibility": [False, True],
                    "terminal_tail_semantic_adjustment_l2": [0.25, 0.5],
                    "ramp_episode_context": {
                        "window_id": "tail-window",
                        "split": "train",
                        "source_hashes": {"fixture": "hash"},
                        "forecast_model": "fixture",
                        "forecast_vintage": "v1",
                        "future_realized_features_exposed": False,
                    },
                }
            ]
        }
        self.assertTrue(callback._on_step())
        self.assertEqual(callback.dual_updates[0]["energy_cost"], 6.0)
        self.assertEqual(callback.dual_updates[0]["status_quo_energy_cost"], 3.0)
        self.assertEqual(callback.emergency, 1)
        self.assertEqual(callback.semantic_adjustment_sum, 0.75)

    def test_real_factory_reports_missing_panel_not_interface_mismatch(self) -> None:
        request = EnvRequest(split="train", seed=1, training=True)
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "factory_manifest.json"
            with patch("env.ramp_v6.factory.ENERGY_MODEL_V3_HANDOFF", missing):
                with self.assertRaisesRegex(
                    MissingEnergyModelV3PanelError,
                    "missing energy-model-v3 ramp panel handoff",
                ):
                    make_energy_model_v3_env(request)


if __name__ == "__main__":
    unittest.main()
