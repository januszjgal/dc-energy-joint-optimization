"""Campaign reporting must expose component regressions behind a joint gain."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from env.ramp_v6.objective import JointObjective
from ramp_rl.runner import LEARNING_CURVE_COLUMNS
import scripts.aggregate_four_market_v2_campaign as aggregation
import scripts.run_four_market_cv as cv


ROOT = Path(__file__).resolve().parents[2]


class JointReportingTests(unittest.TestCase):
    def test_re_evaluation_preserves_the_original_training_duration(self) -> None:
        training = {"effective_interactions": 96, "seed": 1}
        held_out = ["2025-01", "2025-05", "2025-09"]
        with tempfile.TemporaryDirectory(prefix=".test-cv-cache-", dir=ROOT) as directory:
            root = Path(directory)
            output = root / "output" / cv.ARTIFACT_NAMESPACE / "cv" / "test" / "k1" / "seed-1"
            checkpoint = root / "models" / cv.ARTIFACT_NAMESPACE / "cv" / "test" / "k1" / "seed-1"
            output.mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            for name in ("training_summary.json", "model.zip", "vecnormalize.pkl"):
                (checkpoint / name).write_bytes(b"fixture")
            (output / "summary.json").write_text(
                json.dumps({"training_identity": training, "training_elapsed_seconds": 12.0}),
                encoding="utf-8",
            )
            with (
                patch.object(cv, "ROOT", root),
                patch.object(cv, "_validate_fold", return_value=held_out),
                patch.object(cv, "run_training", return_value=training),
                patch.object(cv, "evaluate_checkpoint", return_value={"fixture": True}),
                patch.object(cv, "_compact", side_effect=lambda value: value),
            ):
                result = cv.run_job("k1", 1, timesteps=96, tag="test")
            self.assertTrue(result["training_reused"])
            self.assertEqual(result["training_elapsed_seconds"], 12.0)
            self.assertEqual(result["training_interactions_per_second"], 8.0)
            # Each held-out month is evaluated as its own episode.
            self.assertEqual(sorted(result["validation_by_interactions"]["96"]), held_out)
            self.assertEqual(len(result["training_months"]), 9)
            self.assertFalse(set(result["training_months"]) & set(held_out))

    def test_cross_validation_averages_every_month_once_per_seed(self) -> None:
        results = []
        for fold, months in (("k1", ["2025-01", "2025-05", "2025-09"]),
                             ("k2", ["2025-02", "2025-06", "2025-10"])):
            for seed, offset in ((1, 0.0), (2, 0.002)):
                results.append({
                    "fold": fold, "seed": seed, "effective_interactions": 96,
                    "validation_by_interactions": {"96": {
                        month: {"improvement_J": 0.01 * (index + 1) + offset}
                        for index, month in enumerate(months)
                    }},
                })
        summary = cv.aggregate(results)
        self.assertEqual(summary["months"], 6)
        self.assertAlmostEqual(summary["mean_improvement_J_by_seed"]["1"], 0.02)
        self.assertAlmostEqual(summary["mean_improvement_J_by_seed"]["2"], 0.022)
        self.assertAlmostEqual(summary["mean_improvement_J"], 0.021)

    def test_complete_aggregation_reports_joint_gain_and_peak_regression(self) -> None:
        rows = []
        for index, seed in enumerate(aggregation.SEEDS, start=1):
            ramp_gain = 0.04 * index
            peak_gain = -0.02 * index
            improvement = 0.5 * (ramp_gain + peak_gain)
            market = {
                "policy_joint_J": 1.0 - improvement,
                "policy_ramp_impact_sum": 1.0 - ramp_gain,
                "status_quo_ramp_impact_sum": 1.0,
                "status_quo_joint_J": 1.0,
                "improvement": improvement,
                "policy_peak_mw": 400.0 * (1.0 - peak_gain),
                "status_quo_peak_mw": 400.0,
                "peak_reduction_mw": 400.0 * peak_gain,
                "policy_native_relative_incremental_ramp_impact": 0.5 - ramp_gain,
                "status_quo_native_relative_incremental_ramp_impact": 0.5,
                "policy_minus_status_quo_incremental_ramp_impact": -ramp_gain,
                "policy_outperforms_status_quo": True,
            }
            validation = {
                "objective": JointObjective().as_dict(),
                "day_count": 31,
                "mean_joint_J": 1.0 - improvement,
                "mean_monthly_ramp_impact": 1.0 - ramp_gain,
                "mean_incremental_ramp_impact": 0.5 - ramp_gain,
                "emergency_feasibility_rate": 0.0,
                **dict.fromkeys(
                    ("service_unserved", "batch_unfinished", "batch_expired",
                     "terminal_work", "certificate_violations"), 0.0
                ),
                "status_quo_comparison": {
                    "policy_J": 1.0 - improvement,
                    "status_quo_J": 1.0,
                    "policy_minus_status_quo_joint_J": -improvement,
                    "improvement": improvement,
                    "status_quo_native_relative_mean_incremental_ramp_impact": 0.5,
                    "policy_minus_status_quo_mean_incremental_ramp_impact": -ramp_gain,
                    "components": {
                        "ramp": {"improvement": ramp_gain},
                        "net_load_peak": {"improvement": peak_gain},
                    },
                    "per_market": {"M": market},
                },
            }
            rows.append({
                "seed": seed,
                "effective_interactions": 2_048_000,
                "resumed_from_interactions": 0,
                "training_elapsed_seconds": 1.0,
                "test_window_ids": ["2026-05"],
                "test": validation,
            })
        with tempfile.TemporaryDirectory(prefix=".test-joint-report-", dir=ROOT) as directory:
            output = Path(directory)
            for seed in aggregation.SEEDS:
                path = output / f"seed-{seed}" / "learning_curve.csv"
                path.parent.mkdir()
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=LEARNING_CURVE_COLUMNS)
                    writer.writeheader()
                    for count, reward in ((2048, -0.001), (4096, -0.0009)):
                        writer.writerow({
                            **dict.fromkeys(LEARNING_CURVE_COLUMNS, 0.0),
                            "interaction_count": count,
                            "mean_raw_joint_reward": reward,
                            "mean_raw_ramp_reward": 0.4 * reward,
                            "mean_raw_peak_reward": 0.6 * reward,
                        })
            with patch.object(aggregation, "_require_complete_summaries", return_value=rows):
                result = aggregation.aggregate(output)
            self.assertAlmostEqual(result["mean_improvement"], 0.055)
            self.assertEqual(result["paired_policy_minus_status_quo"]["seed_win_count"], 10)
            self.assertEqual(result["component_comparisons"]["ramp"]["seed_win_count"], 10)
            self.assertEqual(result["component_comparisons"]["net_load_peak"]["seed_loss_count"], 10)
            self.assertAlmostEqual(result["per_market"]["M"]["mean_peak_reduction_mw"], -44.0)
            self.assertTrue((output / "learning_curve_aggregate.png").is_file())
            with (output / "learning_curve_aggregate.csv").open(encoding="utf-8") as handle:
                self.assertIn("mean_raw_joint_reward", next(csv.reader(handle)))


if __name__ == "__main__":
    unittest.main()
