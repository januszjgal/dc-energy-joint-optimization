from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from energy_model_v3.builder import (
    MARKETS,
    build_panel,
    fixture_market_frame,
    load_scenario,
)
from energy_model_v3.calendar import make_calendar
from energy_model_v3.canonical import (
    aggregate_native_hourly,
    derive_net_load,
)
from energy_model_v3.contract import ContractError, validate_forecasts
from energy_model_v3.derivations import add_grid_features, calibrate_scale

ROOT = Path(__file__).resolve().parents[2]


class CalendarAndScenarioTests(unittest.TestCase):
    def test_locked_calendar_and_splits(self) -> None:
        calendar = make_calendar()
        self.assertEqual(calendar.rows, 5_808)
        self.assertEqual(calendar.train_end_utc, "2026-02-01T00:00:00+00:00")
        self.assertEqual(
            calendar.validation_end_utc, "2026-03-01T00:00:00+00:00"
        )
        self.assertEqual(calendar.test_end_utc, "2026-05-01T00:00:00+00:00")

    def test_all_scenarios_obey_mapping_and_penetration_rules(self) -> None:
        paths = sorted(
            (ROOT / "env" / "scenarios").glob("us_six_market_v3_*.yaml")
        )
        self.assertEqual(len(paths), 3)
        scenarios = [load_scenario(path) for path in paths]
        stress = [
            item
            for item in scenarios
            if item["study_role"] == "non_primary_price_taking_stress"
        ]
        self.assertEqual(len(stress), 1)
        self.assertTrue(stress[0]["penetration_override"])


class CanonicalTests(unittest.TestCase):
    def test_incomplete_subhourly_hour_fails_without_fill(self) -> None:
        frame = pd.DataFrame(
            {
                "interval_start_utc": pd.date_range(
                    "2025-09-01T00:00:00Z", periods=11, freq="5min"
                ),
                "value": np.arange(11),
            }
        )
        with self.assertRaisesRegex(ContractError, "incomplete hours"):
            aggregate_native_hourly(
                frame, value_columns=["value"], native_minutes=5
            )

    def test_nyiso_pal_is_not_double_subtracted(self) -> None:
        frame = pd.DataFrame(
            {
                "gross_demand_mw": [10_000.0],
                "solar_mw": [2_000.0],
            }
        )
        result = derive_net_load(frame, method="pal-reflects-btm")
        self.assertEqual(result.loc[0, "net_load_mw"], 10_000.0)
        self.assertIn("no_double_subtraction", result.loc[0, "net_load_method"])

    def test_realized_future_forecast_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "interval_start_utc": ["2025-09-01T03:00:00Z"],
                "market": ["CAISO_NP15"],
                "forecast_target": ["net_load_mw"],
                "forecast_value_mw": [1.0],
                "forecast_issue_utc": ["2025-09-01T00:00:00Z"],
                "forecast_vintage_utc": ["2025-09-01T00:00:00Z"],
                "forecast_horizon_hours": [3],
                "forecast_capability": ["realized_future"],
                "source_quality_flags": ["bad"],
            }
        )
        with self.assertRaisesRegex(ContractError, "realized future"):
            validate_forecasts(frame)


class DerivationTests(unittest.TestCase):
    def test_scale_uses_training_rows_only(self) -> None:
        calendar = make_calendar()
        frame = fixture_market_frame("CAISO_NP15", calendar=calendar)
        baseline = calibrate_scale(
            frame,
            market="CAISO_NP15",
            train_start=pd.Timestamp(calendar.train_start_utc),
            train_end=pd.Timestamp(calendar.train_end_utc),
        )
        validation = pd.to_datetime(frame["interval_start_utc"], utc=True) >= (
            pd.Timestamp(calendar.train_end_utc)
        )
        frame.loc[validation, "gross_demand_mw"] = 1e12
        frame.loc[validation, "net_load_mw"] = 1e12
        changed = calibrate_scale(
            frame,
            market="CAISO_NP15",
            train_start=pd.Timestamp(calendar.train_start_utc),
            train_end=pd.Timestamp(calendar.train_end_utc),
        )
        self.assertEqual(
            baseline.gross_demand_q95_mw, changed.gross_demand_q95_mw
        )
        self.assertEqual(
            baseline.net_load_q90_fraction, changed.net_load_q90_fraction
        )

    def test_incremental_ramp_definition(self) -> None:
        calendar = make_calendar()
        frame = fixture_market_frame("CAISO_NP15", calendar=calendar)
        scale = calibrate_scale(
            frame,
            market="CAISO_NP15",
            train_start=pd.Timestamp(calendar.train_start_utc),
            train_end=pd.Timestamp(calendar.train_end_utc),
        )
        dc = pd.Series(np.linspace(50.0, 100.0, len(frame)))
        result = add_grid_features(frame, dc, scale)
        row = 3
        net = frame["net_load_mw"]
        expected = (
            (
                (net.iloc[row] + dc.iloc[row])
                - (net.iloc[row - 1] + dc.iloc[row - 1])
            )
            / scale.gross_demand_q95_mw
        ) ** 2 - (
            (net.iloc[row] - net.iloc[row - 1])
            / scale.gross_demand_q95_mw
        ) ** 2
        self.assertAlmostEqual(result.loc[row, "incremental_ramp_1h"], expected)
        self.assertTrue(pd.isna(result.loc[0, "prior_dc_power_mw"]))


class EndToEndFixtureTests(unittest.TestCase):
    def test_six_market_fixture_build(self) -> None:
        calendar = make_calendar()
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            data_root = temporary_root / "data"
            output_root = temporary_root / "output"
            dc_power: dict[str, pd.Series] = {}
            for seed, market in enumerate(MARKETS):
                frame = fixture_market_frame(
                    market, calendar=calendar, seed=seed
                )
                market_root = data_root / "native" / market
                market_root.mkdir(parents=True)
                physical_columns = [
                    "interval_start_utc",
                    "interval_end_utc",
                    "market",
                    "gross_demand_mw",
                    "wind_mw",
                    "solar_mw",
                    "net_load_mw",
                    "net_load_method",
                    "source_quality_flags",
                ]
                price_columns = [
                    "interval_start_utc",
                    "interval_end_utc",
                    "market",
                    "price_market",
                    "price_location",
                    "da_lmp_usd_mwh",
                    "rt_lmp_usd_mwh",
                    "energy_component_usd_mwh",
                    "congestion_component_usd_mwh",
                    "loss_component_usd_mwh",
                    "source_quality_flags",
                ]
                frame[physical_columns].to_csv(
                    market_root / "physical_hourly.csv", index=False
                )
                frame[price_columns].to_csv(
                    market_root / "price_hourly.csv", index=False
                )
                dc_power[market] = pd.Series(
                    np.full(len(frame), 75.0 + seed)
                )
            manifest = build_panel(
                data_root=data_root,
                output_root=output_root,
                scenario_path=(
                    ROOT
                    / "env"
                    / "scenarios"
                    / "us_six_market_v3_primary.yaml"
                ),
                dc_power=dc_power,
                calendar=calendar,
            )
            self.assertEqual(set(manifest["scales"]), set(MARKETS))
            self.assertEqual(len(manifest["outputs"]), 6)
            self.assertTrue(manifest["training_only_calibration"])


class FrozenLegacyRegressionTests(unittest.TestCase):
    def test_branch_changes_only_additive_v3_surfaces(self) -> None:
        allowed_prefixes = (
            "energy_model_v3/",
            "tests/energy_model_v3/",
            "data/energy_model_v3/",
            "output/energy_model_v3/",
        )
        allowed_files = {
            "scripts/build_energy_model_v3.py",
            "env/protocols/independent_us_v3.yaml",
            "env/scenarios/us_six_market_v3_primary.yaml",
            "env/scenarios/us_six_market_v3_robustness.yaml",
            "env/scenarios/us_six_market_v3_stress_6gw.yaml",
        }
        result = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMRTUXB",
                "origin/master",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        changed = [line.strip().replace("\\", "/") for line in result.stdout.splitlines()]
        forbidden = [
            path
            for path in changed
            if path not in allowed_files
            and not path.startswith(allowed_prefixes)
        ]
        self.assertEqual(forbidden, [], f"legacy surfaces changed: {forbidden}")


if __name__ == "__main__":
    unittest.main()
