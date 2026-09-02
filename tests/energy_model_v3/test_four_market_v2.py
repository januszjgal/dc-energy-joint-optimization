"""Integration tests for the one raw-workload four-market experiment."""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import numpy as np
import yaml

from energy_model_v3.four_market_v2 import (
    CELLS, COMPUTE_CAPACITY, MARKET_TO_CELL, MARKETS, make_four_market_env,
)
from ramp_rl.contract import EnvRequest, RampEnvAdapter
from scripts.build_four_market_v2_factory import _raw_arrivals, build, validate
from scripts.run_four_market_v2_campaign import preflight


ROOT = Path(__file__).resolve().parents[2]
FACTORY_ROOT = ROOT / "output" / "four_market_v2" / "factory"
CALENDAR_PATH = ROOT / "data" / "four_market_2025" / "calendar.json"


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
            self.assertEqual(observation.shape, (96,))
            schema = env._current.observation_schema
            self.assertEqual(len(schema), 96)
            self.assertEqual(
                [name for name in schema if name.startswith("batch_queue_deadline")],
                [
                    "batch_queue_deadline_le_1h_capacity_fraction",
                    "batch_queue_deadline_le_3h_capacity_fraction",
                ],
            )
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
            times = current.panel.timestamps[3:-3]
            raw_service, raw_batch = _raw_arrivals(times)
            np.testing.assert_allclose(current.workload.service_arrivals[:-3], raw_service)
            np.testing.assert_allclose(current.workload.batch_arrivals[:-3], raw_batch)
            np.testing.assert_array_equal(
                current.workload.batch_deadline_hours[0], [2, 1, 2, 3]
            )
            np.testing.assert_array_equal(current.workload.service_arrivals[-3:], 0.0)
            np.testing.assert_array_equal(current.workload.batch_arrivals[-3:], 0.0)
            max_arrivals = float((raw_service + raw_batch).sum(axis=1).max())
            expected = 1.0 - max_arrivals / float(current._capacity.sum())
            self.assertAlmostEqual(current._future_batch_capacity_fraction, expected)
            self.assertGreaterEqual(current._future_batch_capacity_fraction, 0.0)
        finally:
            env.close()

    def test_only_train_validation_and_cells_a_to_d_are_accessible(self) -> None:
        calendar = json.loads(CALENDAR_PATH.read_text())
        self.assertEqual(len(calendar["train"]), 334)
        self.assertEqual(len(calendar["validation"]), 31)
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
                self.assertTrue(record["panel_path"].startswith("data/four_market_2025/windows/"))
                self.assertTrue((ROOT / record["fixture_path"]).is_file())

    def test_adapter_and_status_quo_complete(self) -> None:
        request = EnvRequest(split="validation", seed=4101, window_id=self.window)
        adapter = RampEnvAdapter(make_four_market_env(request), request)
        try:
            observation, _ = adapter.reset(seed=4101)
            self.assertEqual(observation.shape, (96,))
            while True:
                _, reward, terminated, truncated, info = adapter.step(adapter.evaluation_action("status_quo"))
                self.assertFalse(truncated)
                self.assertAlmostEqual(reward, -info["incremental_ramp_impact"])
                if terminated:
                    self.assertEqual(info["batch_unfinished"], 0.0)
                    self.assertEqual(info["certificate_violations"], 0)
                    break
        finally:
            adapter.close()

    def test_all_window_preflight(self) -> None:
        result = preflight()
        self.assertEqual(result["train"], 334)
        self.assertEqual(result["validation"], 31)
        self.assertGreater(result["steps"], 0)


if __name__ == "__main__":
    unittest.main()
