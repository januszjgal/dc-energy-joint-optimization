from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from energy_model_v3.builder import (
    MARKETS,
    blocked_preflight,
    build_panel,
    fixture_market_frame,
    load_scenario,
)
from energy_model_v3.calendar import make_calendar
from energy_model_v3.cli import build_measured_dc_power
from energy_model_v3.acquisition import _price_frame
from energy_model_v3.canonical import (
    aggregate_native_hourly,
    derive_net_load,
)
from energy_model_v3.contract import ContractError, validate_forecasts
from energy_model_v3.derivations import add_grid_features, calibrate_scale
from energy_model_v3.forecasts import reconstruct_forecasts

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

    def test_primary_role_cannot_redeclare_capacity(self) -> None:
        source = (
            ROOT / "env" / "scenarios" / "us_six_market_v3_primary.yaml"
        ).read_text(encoding="utf-8")
        modified = source.replace(
            "total_dc_power_mw: 600.0", "total_dc_power_mw: 1200.0"
        ).replace("rated_power_mw: 100.0", "rated_power_mw: 200.0")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.yaml"
            path.write_text(modified, encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "total must be 600"):
                load_scenario(path)

    def test_primary_role_cannot_use_robustness_mapping_or_nan_gate(self) -> None:
        source = (
            ROOT / "env" / "scenarios" / "us_six_market_v3_primary.yaml"
        ).read_text(encoding="utf-8")
        mapping = yaml.safe_load(source)
        mapping["mapping"] = "robustness_overlapping_non_oof"
        for site, cell in zip(mapping["sites"], "cdefgh", strict=True):
            site["borg_cell"] = cell
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.yaml"
            path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "mapping must be"):
                load_scenario(path)
            path.write_text(
                source.replace(
                    "max_penetration_fraction: 0.05",
                    "max_penetration_fraction: .nan",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "must be finite"):
                load_scenario(path)

    def test_measured_workload_power_matches_primary_scenario(self) -> None:
        calendar = make_calendar()
        scenario = ROOT / "env" / "scenarios" / "us_six_market_v3_primary.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            power, manifest = build_measured_dc_power(
                Path(temporary), scenario, calendar=calendar
            )
            self.assertEqual(tuple(power), MARKETS)
            self.assertEqual(manifest["mapping"], "primary_independent")
            for market, values in power.items():
                self.assertEqual(len(values), calendar.rows)
                self.assertGreater(float(values.min()), 0.0)
                self.assertLessEqual(float(values.max()), 100.0)
                self.assertGreater(float(values.std()), 0.0)
                record = manifest["markets"][market]
                self.assertEqual(record["source_5min_rows"], 8_929)
                self.assertEqual(record["aggregation_5min_rows"], 8_928)

    def test_price_frame_rejects_duplicate_utc_hours(self) -> None:
        timestamp = pd.Timestamp("2025-09-01T00:00:00Z")
        with self.assertRaisesRegex(ContractError, "duplicate UTC hours"):
            _price_frame(
                "CAISO_NP15",
                pd.Series([timestamp, timestamp]),
                pd.Series([10.0, 11.0]),
                source_flag="test",
            )


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

    def test_one_hour_forecast_is_allowed_when_issue_matches(self) -> None:
        frame = pd.DataFrame(
            {
                "interval_start_utc": ["2025-09-01T01:00:00Z"],
                "market": ["CAISO_NP15"],
                "forecast_target": ["net_load_mw"],
                "forecast_value_mw": [1.0],
                "forecast_issue_utc": ["2025-09-01T00:00:00Z"],
                "forecast_vintage_utc": ["2025-09-01T00:00:00Z"],
                "forecast_horizon_hours": [1],
                "forecast_capability": ["native_causal"],
                "source_quality_flags": ["ok"],
            }
        )
        validated = validate_forecasts(frame)
        self.assertEqual(validated.loc[0, "forecast_horizon_hours"], 1)
        frame["forecast_horizon_hours"] = 2
        with self.assertRaisesRegex(ContractError, "one or three"):
            validate_forecasts(frame)

    def test_nonfinite_and_case_variant_forbidden_forecasts_fail(self) -> None:
        frame = pd.DataFrame(
            {
                "interval_start_utc": ["2025-09-01T03:00:00Z"],
                "market": ["CAISO_NP15"],
                "forecast_target": ["net_load_mw"],
                "forecast_value_mw": [np.inf],
                "forecast_issue_utc": ["2025-09-01T00:00:00Z"],
                "forecast_vintage_utc": ["2025-09-01T00:00:00Z"],
                "forecast_horizon_hours": [3],
                "forecast_capability": ["REALIZED_FUTURE"],
                "source_quality_flags": ["bad"],
            }
        )
        with self.assertRaisesRegex(ContractError, "finite"):
            validate_forecasts(frame)
        frame["forecast_value_mw"] = 1.0
        with self.assertRaisesRegex(ContractError, "realized future"):
            validate_forecasts(frame)

    def test_misaligned_native_grid_fails(self) -> None:
        frame = pd.DataFrame(
            {
                "interval_start_utc": pd.date_range(
                    "2025-09-01T00:00:00Z", periods=12, freq="1min"
                ),
                "value": np.arange(12),
            }
        )
        with self.assertRaisesRegex(ContractError, "do not match"):
            aggregate_native_hourly(
                frame, value_columns=["value"], native_minutes=5
            )

    def test_reconstructed_forecast_models_predate_every_issue(self) -> None:
        calendar = make_calendar()
        source = fixture_market_frame(
            "CAISO_NP15", calendar=calendar, seed=909
        )
        forecast, metadata = reconstruct_forecasts(
            source, market="CAISO_NP15", calendar=calendar
        )
        model_records = {
            model["model_id"]: model
            for horizon in metadata["models"].values()
            for model in horizon["models"]
        }
        self.assertEqual(set(forecast["forecast_model_id"]), set(model_records))
        for model in model_records.values():
            self.assertLess(
                pd.Timestamp(model["latest_training_target_utc"]),
                pd.Timestamp(model["vintage_utc"]),
            )
        self.assertTrue(
            (
                forecast["forecast_vintage_utc"]
                <= forecast["forecast_issue_utc"]
            ).all()
        )
        for horizon in (1, 3):
            rows = forecast[forecast["forecast_horizon_hours"] == horizon]
            expected = pd.date_range(
                rows["interval_start_utc"].min(),
                rows["interval_start_utc"].max(),
                freq="h",
                tz="UTC",
            )
            self.assertTrue(
                pd.DatetimeIndex(rows["interval_start_utc"]).sort_values().equals(
                    expected
                )
            )


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
                fixture_mode=True,
            )
            self.assertEqual(set(manifest["scales"]), set(MARKETS))
            self.assertEqual(len(manifest["outputs"]), 6)
            self.assertTrue(manifest["training_only_calibration"])
            self.assertTrue(manifest["fixture_mode"])
            capabilities = [
                {
                    "market": market,
                    "status": "AVAILABLE",
                    "reason": "test",
                }
                for market in MARKETS
            ]
            report = blocked_preflight(
                data_root, capabilities, calendar
            )
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(set(report["invalid_primary_inputs"]), set(MARKETS))
            dc_power[MARKETS[0]] = pd.Series(
                np.full(len(frame), 101.0)
            )
            with self.assertRaisesRegex(
                ContractError, "exceeds 100.0 MW site rating"
            ):
                build_panel(
                    data_root=data_root,
                    output_root=output_root,
                    scenario_path=(
                        ROOT / "env" / "scenarios"
                        / "us_six_market_v3_primary.yaml"
                    ),
                    dc_power=dc_power,
                    calendar=calendar,
                    fixture_mode=True,
                )

    def test_live_build_rejects_synthetic_fixture(self) -> None:
        calendar = make_calendar()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = fixture_market_frame("CAISO_NP15", calendar=calendar)
            market_root = root / "native" / "CAISO_NP15"
            market_root.mkdir(parents=True)
            physical = [
                "interval_start_utc", "interval_end_utc", "market",
                "gross_demand_mw", "wind_mw", "solar_mw", "net_load_mw",
                "net_load_method", "source_quality_flags",
            ]
            price = [
                "interval_start_utc", "interval_end_utc", "market",
                "price_market", "price_location", "da_lmp_usd_mwh",
                "rt_lmp_usd_mwh", "energy_component_usd_mwh",
                "congestion_component_usd_mwh", "loss_component_usd_mwh",
                "source_quality_flags",
            ]
            frame[physical].to_csv(
                market_root / "physical_hourly.csv", index=False
            )
            frame[price].to_csv(
                market_root / "price_hourly.csv", index=False
            )
            dc = {
                market: pd.Series(np.full(len(frame), 50.0))
                for market in MARKETS
            }
            with self.assertRaisesRegex(
                ContractError, "synthetic fixture data is forbidden"
            ):
                build_panel(
                    data_root=root,
                    output_root=root / "output",
                    scenario_path=(
                        ROOT / "env" / "scenarios"
                        / "us_six_market_v3_primary.yaml"
                    ),
                    dc_power=dc,
                    calendar=calendar,
                )

    def test_preflight_validates_capability_and_content(self) -> None:
        calendar = make_calendar()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for market in MARKETS:
                market_root = root / "native" / market
                market_root.mkdir(parents=True)
                (market_root / "physical_hourly.csv").write_text(
                    "bad\n", encoding="utf-8"
                )
                (market_root / "price_hourly.csv").write_text(
                    "bad\n", encoding="utf-8"
                )
            capabilities = [
                {
                    "market": market,
                    "status": "AVAILABLE",
                    "reason": "test",
                }
                for market in MARKETS
            ]
            report = blocked_preflight(root, capabilities, calendar)
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(set(report["invalid_primary_inputs"]), set(MARKETS))
            self.assertIn(
                "coverage_assessment", report["coverage_failures"]
            )


class FrozenLegacyRegressionTests(unittest.TestCase):
    def test_branch_changes_only_additive_v3_surfaces(self) -> None:
        allowed_prefixes = (
            "energy_model_v3/",
            "tests/energy_model_v3/",
            "tests/ramp_v6/",
            "tests/fixtures/ramp_v6/",
            "data/energy_model_v3/",
            "output/energy_model_v3/",
            "output/ramp_rl_v6/",
            "output/ramp_v6_fixture/",
            "models/ramp_rl_v6/",
            "ramp_rl/",
            "env/ramp_v6/",
            "docs/figures/ramp_v6/",
        )
        allowed_files = {
            ".gitignore",
            "README.md",
            "data/README.md",
            "requirements.txt",
            "scripts/build_energy_model_v3.py",
            "scripts/build_energy_v3_ramp_factory.py",
            "scripts/build_ramp_rl_evidence_v6.py",
            "scripts/build_ramp_rl_final_evidence.py",
            "scripts/build_ramp_rl_v2_candidate_report.py",
            "scripts/bind_v4r_metric_replay_recovery.py",
            "scripts/build_final_thesis.py",
            "scripts/build_final_thesis_docx.js",
            "scripts/build_ramp_rl_thesis_results.py",
            "scripts/build_ramp_thesis_figures.py",
            "scripts/materialize_ramp_thesis.py",
            "scripts/recompute_v4r_posthoc_metrics.py",
            "scripts/run_ramp_rl_v6.py",
            "scripts/run_ramp_rl_v3.py",
            "scripts/run_ramp_rl_v4.py",
            "scripts/run_ramp_v6_fixture.py",
            "scripts/smoke_test_ramp_rl_v6.py",
            "scripts/summarize_ramp_rl_stage.py",
            "scripts/validate_ramp_thesis.py",
            "tests/__init__.py",
            "tests/test_ramp_thesis.py",
            "docs/ramp_rl_v6.md",
            "docs/ramp_v6_protocol.md",
            "docs/ramp_v6_result_handoff.md",
            "docs/v4r_adversarial_review.md",
            "env/protocols/independent_us_v3.yaml",
            "env/protocols/v6_pure_ramp_rl.schema.json",
            "env/protocols/v6_pure_ramp_rl.yaml",
            "env/protocols/v6_pure_ramp_rl_v2.yaml",
            "env/protocols/v6_pure_ramp_rl_v3.yaml",
            "env/protocols/v6_pure_ramp_rl_v4.yaml",
            "env/protocols/v6_ramp_panel.schema.json",
            "env/protocols/v6_ramp_pure_rl.yaml",
            "env/protocols/v6_ramp_pure_rl_v2.yaml",
            "env/scenarios/us_six_market_v3_primary.yaml",
            "env/scenarios/us_six_market_v3_robustness.yaml",
            "env/scenarios/us_six_market_v3_stress_6gw.yaml",
            "thesis_paper.docx",
            "thesis_paper.md",
            "thesis_ramp_v6.md",
        }
        result = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACDMRTUXB",
                "origin/master",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        changed = [
            line.strip().replace("\\", "/")
            for line in result.stdout.splitlines()
        ]
        forbidden = [
            path
            for path in changed
            if path not in allowed_files
            and not path.startswith(allowed_prefixes)
        ]
        self.assertEqual(forbidden, [], f"legacy surfaces changed: {forbidden}")


if __name__ == "__main__":
    unittest.main()
