"""Integration tests for the one raw-workload four-market experiment."""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from energy_model_v3.four_market_v2 import (
    CELLS, COMPUTE_CAPACITY, FACTORY_ROOT, MARKET_TO_CELL, MARKETS, make_four_market_env, month_hours,
)
from env.ramp_v6.models import HISTORY_HOURS
from ramp_rl.contract import EnvRequest, RampEnvAdapter
from scripts.build_four_market_v2_factory import (
    _load_calendar, _raw_arrivals, _site_configs, build, calibrate_objective, validate,
)
from env.ramp_v6.panel import CanonicalMarketPanel
from scripts.run_four_market_v2_campaign import preflight


ROOT = Path(__file__).resolve().parents[2]
CALENDAR_PATH = ROOT / "data" / "four_market_2025" / "calendar.json"
OBSERVATION_SIZE = 4 * 14 + 4 * 4 + 5 + 5


class FourMarketV2DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build()
        cls.factory = json.loads((FACTORY_ROOT / "factory.json").read_text())
        cls.window = sorted(cls.factory["windows"]["validation"])[0]

    def make_env(self):
        return make_four_market_env(
            EnvRequest(split="validation", seed=4101, window_id=self.window)
        )

    def test_mapping_spaces_and_observation_schema(self) -> None:
        env = self.make_env()
        try:
            observation, _ = env.reset(seed=4101)
            self.assertEqual(tuple((site.market_id, site.site_id[-1]) for site in env._current.sites), MARKET_TO_CELL)
            self.assertEqual(tuple(env._current.panel.markets), tuple(sorted(MARKETS)))
            self.assertEqual(env.action_space.shape, (9,))
            self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
            schema = env._current.observation_schema
            self.assertEqual(len(schema), OBSERVATION_SIZE)
            for market in MARKETS:
                for quantity in ("gross", "net"):
                    self.assertEqual(
                        [name for name in schema if name.startswith(f"{market}:forecast_{quantity}_t+")],
                        [f"{market}:forecast_{quantity}_t+{ahead}_z" for ahead in (0, 2, 5, 11)],
                    )
            self.assertEqual(
                [name for name in schema if name.startswith("batch_queue_deadline")],
                [
                    "batch_queue_deadline_le_1h_capacity_fraction",
                    "batch_queue_deadline_le_3h_capacity_fraction",
                    "batch_queue_deadline_le_6h_capacity_fraction",
                    "batch_queue_deadline_le_12h_capacity_fraction",
                    "batch_queue_deadline_le_24h_capacity_fraction",
                ],
            )
            self.assertFalse(any("lag0" in name or "episode_progress" in name or "terminal" in name for name in schema))
            self.assertFalse(any("forecast_vintage_age" in name or "quality_ok" in name for name in schema))
            self.assertFalse(any(name.endswith(suffix) for name in schema for suffix in (
                ":compute_capacity", ":rated_power_100mw_units", ":idle_power_fraction", ":dynamic_power_fraction",
            )))
            self.assertTrue(all(site.compute_capacity == COMPUTE_CAPACITY for site in env._current.sites))
            self.assertTrue(all(site.rated_power_mw == 500.0 for site in env._current.sites))
        finally:
            env.close()

    def test_raw_workload_deadlines_and_capacity_guarantee(self) -> None:
        env = self.make_env()
        try:
            current = env._current
            hours = month_hours(self.window)
            self.assertEqual(current.action_steps, hours)
            times = current.panel.timestamps[HISTORY_HOURS:]
            raw_service, raw_batch = _raw_arrivals(times)
            np.testing.assert_allclose(current.workload.service_arrivals, raw_service)
            np.testing.assert_allclose(current.workload.batch_arrivals, raw_batch)
            self.assertEqual(current.workload.service_arrivals.shape, (hours, 4))
            self.assertEqual(current.workload.warm_power_mw.shape, (HISTORY_HOURS, 4))
            np.testing.assert_array_equal(
                current.workload.batch_deadline_hours, np.full((hours, 4), 24)
            )
            max_arrivals = float((raw_service + raw_batch).sum(axis=1).max())
            expected = 1.0 - max_arrivals / float(current._capacity.sum())
            self.assertAlmostEqual(current._future_batch_capacity_fraction, expected)
            self.assertGreaterEqual(current._future_batch_capacity_fraction, 0.0)
        finally:
            env.close()

    def test_objective_calibration_uses_training_months_only(self) -> None:
        with patch.object(
            CanonicalMarketPanel, "from_csv", wraps=CanonicalMarketPanel.from_csv
        ) as reader:
            calibration = calibrate_objective(_load_calendar(), _site_configs())
        self.assertEqual(calibration, self.factory["objective_calibration"])
        self.assertEqual(len(reader.call_args_list), 11)
        self.assertFalse(any("2025-05" in str(call.args[0]) for call in reader.call_args_list))
        self.assertNotIn("2025-05", calibration["fit_months"])
        scores = calibration["reference_scores_by_month"].values()
        self.assertAlmostEqual(
            np.mean([row["ramp_squared_sum"] / calibration["ramp_reference"] for row in scores]),
            1.0,
        )
        self.assertAlmostEqual(
            np.mean([row["normalized_peak"] / calibration["peak_reference"] for row in scores]),
            1.0,
        )

    def test_objective_variants_share_fixed_references_and_workload(self) -> None:
        for weight in (0.0, 0.5, 1.0):
            env = make_four_market_env(
                EnvRequest(split="validation", seed=4101, window_id=self.window),
                ramp_weight=weight,
            )
            try:
                objective = env._current.protocol.objective
                self.assertEqual(objective.ramp_weight, weight)
                self.assertEqual(objective.peak_weight, 1.0 - weight)
                self.assertEqual(
                    objective.ramp_reference,
                    self.factory["objective_calibration"]["ramp_reference"],
                )
                self.assertEqual(
                    objective.peak_reference,
                    self.factory["objective_calibration"]["peak_reference"],
                )
                observation, _ = env.reset(seed=4101)
                self.assertEqual(observation.shape, (82,))
                self.assertEqual(float(observation[-1]), 1.0)
            finally:
                env.close()

    def test_factory_rejects_changed_inputs_instead_of_reusing_old_scales(self) -> None:
        with patch(
            "scripts.build_four_market_v2_factory._input_digest",
            return_value="different-inputs",
        ):
            with self.assertRaisesRegex(ValueError, "factory inputs changed"):
                validate()

    def test_only_train_validation_and_cells_a_to_d_are_accessible(self) -> None:
        calendar = json.loads(CALENDAR_PATH.read_text())
        self.assertEqual(len(calendar["train"]), 11)
        self.assertEqual([entry["window_id"] for entry in calendar["validation"]], ["2025-05"])
        self.assertEqual(calendar["history_hours"], HISTORY_HOURS)
        self.assertEqual(calendar["test"], [])
        self.assertEqual(calendar["markets"], list(MARKETS))
        rendered = json.dumps(calendar)
        for forbidden in ("cell_e", "cell_f", "cell_g", "cell_h", "e-h"):
            self.assertNotIn(forbidden, rendered)
        with self.assertRaisesRegex(ValueError, "train and validation only"):
            make_four_market_env(EnvRequest(split="test", seed=4101))

    def test_single_protocol_and_no_removed_mechanisms(self) -> None:
        protocol_files = list((ROOT / "env" / "protocols").glob("four_market_v2*.yaml"))
        self.assertEqual(protocol_files, [ROOT / "env" / "protocols" / "four_market_v2.yaml"])
        protocol = yaml.safe_load(protocol_files[0].read_text())
        rendered = yaml.safe_dump(protocol)
        for forbidden in ("envelope", "tail", "multiobjective", "lagrangian", "epsilon", "hash"):
            self.assertNotIn(forbidden, rendered)
        self.assertEqual(protocol["data"]["split"]["validation"]["months"], ["2025-05"])
        self.assertEqual(len(protocol["data"]["split"]["train"]["months"]), 11)
        self.assertEqual(protocol["training"]["ppo"]["gamma"], 0.99)
        stats = json.loads((ROOT / "data" / "four_market_2025" / "frozen_stats.json").read_text())
        self.assertFalse(any("q90" in key.lower() or "tail" in key.lower() for key in stats))
        self.assertFalse((ROOT / "ramp_rl" / "evidence.py").exists())
        self.assertFalse((ROOT / "ramp_rl" / "schema.py").exists())

    def test_factory_rebuild_uses_canonical_panels_by_reference(self) -> None:
        shutil.rmtree(FACTORY_ROOT)
        self.assertEqual(build(), validate())
        factory = json.loads((FACTORY_ROOT / "factory.json").read_text())
        self.assertEqual(factory["windows"]["test"], {})
        self.assertFalse(list(FACTORY_ROOT.rglob("canonical_panel.csv")))
        for split in ("train", "validation"):
            for record in factory["windows"][split].values():
                self.assertTrue(record["panel_path"].startswith("data/four_market_2025/months/"))
                self.assertTrue((ROOT / record["fixture_path"]).is_file())

    def test_adapter_and_status_quo_complete_the_continuous_month(self) -> None:
        request = EnvRequest(split="validation", seed=4101, window_id=self.window)
        adapter = RampEnvAdapter(make_four_market_env(request), request)
        try:
            observation, reset_info = adapter.reset(seed=4101)
            self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
            self.assertEqual(reset_info["episode_context"]["month_id"], self.window)
            steps = 0
            total_reward = 0.0
            total_ramp_impact = 0.0
            dates: list[str] = []
            while True:
                _, reward, terminated, truncated, info = adapter.step(adapter.evaluation_action("status_quo"))
                steps += 1
                dates.append(info["utc_date"])
                self.assertFalse(truncated)
                total_reward += reward
                total_ramp_impact += info["incremental_ramp_impact"]
                self.assertAlmostEqual(reward, info["ramp_reward"] + info["peak_reward"])
                self.assertAlmostEqual(
                    info["incremental_ramp_impact"],
                    sum(
                        row["windows"]["1h"]["incremental_squared_impact"]
                        for row in info["per_market"].values()
                    ),
                    places=15,
                )
                self.assertEqual(
                    {window for row in info["per_market"].values() for window in row["windows"]},
                    {"1h"},
                )
                self.assertEqual(info["batch_queue"]["queued"], 0.0)
                np.testing.assert_allclose(
                    info["service_allocation"],
                    adapter.env._current.workload.service_arrivals[steps - 1],
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    info["batch_by_destination"],
                    adapter.env._current.workload.batch_arrivals[steps - 1],
                    atol=1e-8,
                )
                if terminated:
                    self.assertTrue(info["actual_terminal"])
                    self.assertEqual(info["batch_unfinished"], 0.0)
                    self.assertEqual(info["certificate_violations"], 0)
                    objective = adapter.env._current.protocol.objective
                    self.assertAlmostEqual(
                        total_reward,
                        -objective.score(total_ramp_impact, info["running_peak_normalized_sum"]),
                        places=12,
                    )
                    break
            self.assertEqual(steps, month_hours(self.window))
            self.assertEqual(len(set(dates)), 31)
        finally:
            adapter.close()

    def test_all_window_preflight(self) -> None:
        result = preflight()
        self.assertEqual(result["train"], 11)
        self.assertEqual(result["validation"], 1)
        self.assertEqual(result["steps"], 8760)


if __name__ == "__main__":
    unittest.main()
