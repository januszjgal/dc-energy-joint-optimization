"""Integration tests for the four-market experiment: folds, final factory, test month."""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from energy_model_v3.four_market_v2 import (
    CELLS, MARKET_TO_CELL, MARKETS, REFERENCE_POWER_MW, factory_root, make_four_market_env,
    month_hours, site_capacities,
)
from energy_model_v3.splits import (
    CV_FOLDS, CV_SPECS, DEVELOPMENT_MONTHS, FINAL_SPEC, TEST_FLEET, TEST_MONTHS, TRAIN_FLEET,
    spec_months,
)
from env.ramp_v6 import clock
from env.ramp_v6.models import HISTORY_HOURS
from env.ramp_v6.panel import CanonicalMarketPanel
from ramp_rl.contract import EnvRequest, RampEnvAdapter
from scripts import build_causal_forecasts as forecasts
from scripts.build_data_splits import fold_data_root
from scripts.build_four_market_v2_factory import (
    _hourly_curve, _load_calendar, _raw_arrivals, _site_configs, build, build_all,
    calibrate_objective, validate,
)
from scripts.run_four_market_v2_campaign import preflight


ROOT = Path(__file__).resolve().parents[2]
CALENDAR_PATH = ROOT / "data" / "four_market_2025" / "calendar.json"
PROTOCOL_PATH = ROOT / "env" / "protocols" / "four_market_v2.yaml"
OBSERVATION_SIZE = 4 * 14 + 4 * 4 + 5 + 5
FOLD = "cv-k1"


class FourMarketV2DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build_all()
        cls.factory = json.loads((factory_root(FOLD) / "factory.json").read_text())
        cls.final = json.loads((factory_root(FINAL_SPEC) / "factory.json").read_text())
        cls.window = sorted(cls.factory["windows"]["validation"])[0]

    def make_env(self, split: str = "validation", spec: str = FOLD, window: str | None = None):
        return make_four_market_env(
            EnvRequest(split=split, seed=4101, window_id=window or self.window), spec=spec
        )

    def test_folds_partition_the_year_and_every_copy_agrees(self) -> None:
        held_out = [month for months in CV_FOLDS.values() for month in months]
        self.assertEqual(sorted(held_out), list(DEVELOPMENT_MONTHS))
        for spec in CV_SPECS:
            months = spec_months(spec)
            self.assertEqual(len(months["train"]), 9)
            self.assertEqual(len(months["validation"]), 3)
            self.assertFalse(set(months["train"]) & set(months["validation"]))
            self.assertEqual(months["test"], ())
            # Interleaved folds: every training set spans all four seasons.
            quarters = {(int(month[-2:]) - 1) // 3 for month in months["train"]}
            self.assertEqual(quarters, {0, 1, 2, 3})
        final = spec_months(FINAL_SPEC)
        self.assertEqual(final["train"], DEVELOPMENT_MONTHS)
        self.assertEqual(final["validation"], ())
        self.assertEqual(final["test"], TEST_MONTHS)
        split = yaml.safe_load(PROTOCOL_PATH.read_text())["data"]["split"]
        self.assertEqual(split["development"]["months"], list(DEVELOPMENT_MONTHS))
        self.assertEqual(split["cross_validation"]["folds"], {k: list(v) for k, v in CV_FOLDS.items()})
        self.assertEqual(split["test"]["months"], list(TEST_MONTHS))
        self.assertEqual(split["test"]["workload_cells"], [cell for _, cell in TEST_FLEET])
        calendar = _load_calendar()
        self.assertEqual(calendar["cv_folds"], {k: list(v) for k, v in CV_FOLDS.items()})

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
            self.assertFalse(any("lag0" in name or "episode_progress" in name or "terminal" in name for name in schema))
            self.assertFalse(any("forecast_vintage_age" in name or "quality_ok" in name for name in schema))
            self.assertFalse(any(name.endswith(suffix) for name in schema for suffix in (
                ":compute_capacity", ":rated_power_100mw_units", ":idle_power_fraction", ":dynamic_power_fraction",
            )))
            capacities = site_capacities()
            self.assertEqual(max(capacities), 1.0)
            np.testing.assert_allclose(
                [site.compute_capacity for site in env._current.sites], capacities, rtol=0.0, atol=1e-12
            )
        finally:
            env.close()

    def test_site_sizes_follow_measured_cell_capacity_relative_to_cell_c(self) -> None:
        cells = json.loads((ROOT / "data" / "cell_capacity.json").read_text())["cells"]
        reference = float(cells["c"]["cpu_capacity"])
        protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
        self.assertEqual(protocol["fleet"]["reference_power_mw_largest_site"], REFERENCE_POWER_MW)
        for fleet, block in ((TRAIN_FLEET, protocol["fleet"]), (TEST_FLEET, protocol["test_fleet"])):
            sites = _site_configs(fleet)
            for (market, cell), site in zip(fleet, sites):
                self.assertAlmostEqual(site.compute_capacity, float(cells[cell]["cpu_capacity"]) / reference, places=12)
                self.assertAlmostEqual(site.rated_power_mw, REFERENCE_POWER_MW * site.compute_capacity, places=9)
                self.assertAlmostEqual(block["compute_capacity"][market], site.compute_capacity, places=6)
                self.assertAlmostEqual(block["rated_power_mw"][market], site.rated_power_mw, places=3)
            self.assertAlmostEqual(block["total_rated_power_mw"], sum(s.rated_power_mw for s in sites), places=3)
        # Two test sites are larger than any training site.
        self.assertGreater(max(s.compute_capacity for s in _site_configs(TEST_FLEET)), 1.3)

    def test_arrivals_are_work_units_and_power_sees_cell_utilization(self) -> None:
        for split, spec, window, fleet in (
            ("validation", FOLD, self.window, TRAIN_FLEET),
            ("test", FINAL_SPEC, TEST_MONTHS[0], TEST_FLEET),
        ):
            env = self.make_env(split, spec, window)
            try:
                current = env._current
                times = current.panel.timestamps[HISTORY_HOURS:]
                offsets = clock.trace_hours(times)
                self.assertEqual(tuple(site.site_id[-1] for site in current.sites), tuple(c for _, c in fleet))
                for index, (site, (_, cell)) in enumerate(zip(current.sites, fleet)):
                    service, batch = _hourly_curve(cell)
                    np.testing.assert_allclose(
                        current.workload.service_arrivals[:, index] / site.compute_capacity, service[offsets]
                    )
                    np.testing.assert_allclose(
                        current.workload.batch_arrivals[:, index] / site.compute_capacity, batch[offsets]
                    )
                    utilization = service[offsets[0]] + batch[offsets[0]]
                    work = float(current.workload.service_arrivals[0, index] + current.workload.batch_arrivals[0, index])
                    self.assertAlmostEqual(
                        site.power_mw(work),
                        site.rated_power_mw * (site.idle_power_fraction + site.dynamic_power_fraction * utilization),
                    )
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
            self.assertEqual(current.workload.warm_power_mw.shape, (HISTORY_HOURS, 4))
            np.testing.assert_array_equal(current.workload.batch_deadline_hours, np.full((hours, 4), 24))
            max_arrivals = float((raw_service + raw_batch).sum(axis=1).max())
            expected = 1.0 - max_arrivals / float(current._capacity.sum())
            self.assertAlmostEqual(current._future_batch_capacity_fraction, expected)
            self.assertGreaterEqual(current._future_batch_capacity_fraction, 0.0)
        finally:
            env.close()

    def test_objective_calibration_uses_each_specs_training_months_only(self) -> None:
        for spec, factory in ((FOLD, self.factory), (FINAL_SPEC, self.final)):
            with patch.object(CanonicalMarketPanel, "from_csv", wraps=CanonicalMarketPanel.from_csv) as reader:
                calibration = calibrate_objective(spec)
            self.assertEqual(calibration, factory["objective_calibration"])
            train = spec_months(spec)["train"]
            self.assertEqual(len(reader.call_args_list), len(train))
            read = " ".join(str(call.args[0]) for call in reader.call_args_list)
            for month in spec_months(spec)["validation"] + spec_months(spec)["test"]:
                self.assertNotIn(month, read)
            if spec in CV_SPECS:
                self.assertIn(fold_data_root(spec).as_posix().split("/output/")[1], read.replace("\\", "/"))
            self.assertEqual(sorted(calibration["fit_months"]), sorted(train))
            scores = calibration["reference_scores_by_month"].values()
            self.assertAlmostEqual(
                np.mean([row["ramp_squared_sum"] / calibration["ramp_reference"] for row in scores]), 1.0
            )

    def test_fold_statistics_and_forecasts_are_fitted_on_fold_training_months(self) -> None:
        stats = json.loads((fold_data_root(FOLD) / "frozen_stats.json").read_text())
        self.assertEqual(stats["fit_months"], list(spec_months(FOLD)["train"]))
        final_stats = json.loads((ROOT / "data" / "four_market_2025" / "frozen_stats.json").read_text())
        self.assertEqual(final_stats["fit_months"], list(DEVELOPMENT_MONTHS))
        self.assertNotEqual(stats["gross_q95_mw"], final_stats["gross_q95_mw"])
        panel = pd.read_csv(ROOT / self.factory["windows"]["validation"][self.window]["panel_path"])
        self.assertTrue((panel["forecast_vintage_id"] == f"{forecasts.VINTAGE_ID}+{FOLD}").all())
        for market, scale in stats["gross_q95_mw"].items():
            self.assertTrue(np.allclose(panel.loc[panel.market_id == market, "market_scale_mw"], scale))
        # Changing a held-out month must not change any fitted weight.
        series = forecasts.load_hourly_series()
        train = spec_months(FOLD)["train"]
        models, _ = forecasts.fit_forecasts(series, train)
        changed = series.copy()
        columns = ["gross_demand_mw", "net_load_mw", "wind_mw", "solar_mw"]
        changed[columns] = changed[columns].astype(float)
        held_out = forecasts.in_months(pd.DatetimeIndex(changed["timestamp_utc"]), spec_months(FOLD)["validation"])
        changed.loc[held_out, columns] *= 1.5
        refit, _ = forecasts.fit_forecasts(changed, train)
        for key, model in models.items():
            np.testing.assert_allclose(model.coefficients, refit[key].coefficients, rtol=1e-9, atol=1e-9)
            self.assertAlmostEqual(model.intercept, refit[key].intercept, places=6)

    def test_objective_variants_share_fixed_references_and_workload(self) -> None:
        for weight in (0.0, 0.5, 1.0):
            env = make_four_market_env(
                EnvRequest(split="validation", seed=4101, window_id=self.window), spec=FOLD,
                ramp_weight=weight,
            )
            try:
                objective = env._current.protocol.objective
                self.assertEqual(objective.ramp_weight, weight)
                self.assertEqual(objective.peak_weight, 1.0 - weight)
                self.assertEqual(objective.ramp_reference, self.factory["objective_calibration"]["ramp_reference"])
                self.assertEqual(objective.peak_reference, self.factory["objective_calibration"]["peak_reference"])
                observation, _ = env.reset(seed=4101)
                self.assertEqual(observation.shape, (82,))
                self.assertEqual(float(observation[-1]), 1.0)
            finally:
                env.close()

    def test_factory_rejects_changed_inputs_instead_of_reusing_old_scales(self) -> None:
        with patch("scripts.build_four_market_v2_factory._input_digest", return_value="different-inputs"):
            with self.assertRaisesRegex(ValueError, "factory inputs changed"):
                validate(FOLD)

    def test_test_month_exists_only_in_the_final_factory(self) -> None:
        for spec in CV_SPECS:
            factory = json.loads((factory_root(spec) / "factory.json").read_text())
            self.assertEqual(factory["windows"]["test"], {})
            with self.assertRaisesRegex(ValueError, "no test windows"):
                make_four_market_env(EnvRequest(split="test", seed=4101), spec=spec)
        self.assertEqual(self.final["windows"]["validation"], {})
        self.assertEqual(sorted(self.final["windows"]["test"]), list(TEST_MONTHS))
        record = self.final["windows"]["test"][TEST_MONTHS[0]]
        self.assertEqual(record["workload_cells"], [cell for _, cell in TEST_FLEET])
        self.assertTrue(record["panel_path"].startswith("data/four_market_2026_test/months/"))
        panel = pd.read_csv(ROOT / record["panel_path"])
        stats = json.loads((ROOT / self.final["frozen_stats_path"]).read_text())
        for market, scale in stats["gross_q95_mw"].items():
            self.assertTrue(np.allclose(panel.loc[panel.market_id == market, "market_scale_mw"], scale))
        self.assertTrue((panel["forecast_vintage_id"] == forecasts.VINTAGE_ID).all())
        self.assertTrue((pd.to_datetime(panel.forecast_issue_time_utc) <= pd.to_datetime(panel.timestamp_utc)).all())
        calendar = json.loads(CALENDAR_PATH.read_text())
        self.assertEqual([row["month_id"] for row in calendar["development"]], list(DEVELOPMENT_MONTHS))
        self.assertEqual(calendar["markets"], list(MARKETS))
        self.assertEqual(calendar["history_hours"], HISTORY_HOURS)

    def test_single_protocol_and_no_removed_mechanisms(self) -> None:
        protocol_files = list((ROOT / "env" / "protocols").glob("four_market_v2*.yaml"))
        self.assertEqual(protocol_files, [PROTOCOL_PATH])
        protocol = yaml.safe_load(protocol_files[0].read_text())
        rendered = yaml.safe_dump(protocol)
        for forbidden in ("envelope", "tail", "multiobjective", "lagrangian", "epsilon", "hash"):
            self.assertNotIn(forbidden, rendered)
        self.assertEqual(protocol["data"]["forecast"]["model_id"], forecasts.VINTAGE_ID)
        self.assertEqual(protocol["training"]["ppo"]["gamma"], 0.99)
        stats = json.loads((ROOT / "data" / "four_market_2025" / "frozen_stats.json").read_text())
        self.assertFalse(any("q90" in key.lower() or "tail" in key.lower() for key in stats))
        self.assertFalse((ROOT / "ramp_rl" / "evidence.py").exists())
        self.assertFalse((ROOT / "ramp_rl" / "schema.py").exists())

    def test_final_factory_rebuild_uses_committed_panels_by_reference(self) -> None:
        shutil.rmtree(factory_root(FINAL_SPEC))
        self.assertEqual(build(FINAL_SPEC), validate(FINAL_SPEC))
        factory = json.loads((factory_root(FINAL_SPEC) / "factory.json").read_text())
        self.assertFalse(list(factory_root(FINAL_SPEC).rglob("canonical_panel.csv")))
        for record in factory["windows"]["train"].values():
            self.assertTrue(record["panel_path"].startswith("data/four_market_2025/months/"))
            self.assertTrue((ROOT / record["fixture_path"]).is_file())

    def test_adapter_and_status_quo_complete_the_continuous_month(self) -> None:
        request = EnvRequest(split="validation", seed=4101, window_id=self.window)
        adapter = RampEnvAdapter(make_four_market_env(request, spec=FOLD), request)
        try:
            observation, reset_info = adapter.reset(seed=4101)
            self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
            self.assertEqual(reset_info["episode_context"]["month_id"], self.window)
            self.assertEqual(reset_info["episode_context"]["factory_spec"], FOLD)
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
                self.assertEqual(info["batch_queue"]["queued"], 0.0)
                np.testing.assert_allclose(
                    info["service_allocation"], adapter.env._current.workload.service_arrivals[steps - 1], atol=1e-8,
                )
                if terminated:
                    self.assertTrue(info["actual_terminal"])
                    self.assertEqual(info["batch_unfinished"], 0.0)
                    self.assertEqual(info["certificate_violations"], 0)
                    objective = adapter.env._current.protocol.objective
                    self.assertAlmostEqual(
                        total_reward,
                        -objective.score(total_ramp_impact, info["running_peak_impact_normalized_sum"]),
                        places=12,
                    )
                    break
            self.assertEqual(steps, month_hours(self.window))
            # A month starts at 07:00 UTC (trace clock, UTC-7), so its hours span
            # one more UTC date than it has days.
            self.assertEqual(len(set(dates)), month_hours(self.window) // 24 + 1)
        finally:
            adapter.close()

    def test_final_factory_preflight_is_feasible_in_every_month(self) -> None:
        result = preflight()
        self.assertEqual(result["train"], 12)
        self.assertEqual(result["test"], 1)
        self.assertEqual(result["steps"], 8760 + 744)

    def test_cells_tuple_is_the_training_fleet(self) -> None:
        self.assertEqual(CELLS, tuple(cell for _, cell in TRAIN_FLEET))


if __name__ == "__main__":
    unittest.main()
