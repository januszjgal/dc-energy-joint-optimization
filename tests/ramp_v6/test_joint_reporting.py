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
import scripts.run_four_market_v2_pilot as pilot


ROOT = Path(__file__).resolve().parents[2]


class JointReportingTests(unittest.TestCase):
    def test_re_evaluation_preserves_the_original_training_duration(self) -> None:
        training = {"effective_interactions": 96, "seed": 1}
        with tempfile.TemporaryDirectory(prefix=".test-pilot-cache-", dir=ROOT) as directory:
            root = Path(directory)
            output = root / "output" / pilot.ARTIFACT_NAMESPACE / "pilot" / "test" / "seed-1"
            checkpoint = root / "models" / pilot.ARTIFACT_NAMESPACE / "pilot" / "test" / "seed-1"
            output.mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            for name in ("training_summary.json", "model.zip", "vecnormalize.pkl"):
                (checkpoint / name).write_bytes(b"fixture")
            (output / "summary.json").write_text(
                json.dumps({"training_identity": training, "training_elapsed_seconds": 12.0}),
                encoding="utf-8",
            )
            with (
                patch.object(pilot, "ROOT", root),
                patch.object(pilot, "_validate_campaign_geometry", return_value={"validation": ["2025-05"]}),
                patch.object(pilot, "run_training", return_value=training),
                patch.object(pilot, "evaluate_checkpoint", return_value={"fixture": True}),
                patch.object(pilot, "_compact", side_effect=lambda value: value),
            ):
                result = pilot.run_seed(1, timesteps=96, tag="test")
            self.assertTrue(result["training_reused"])
            self.assertEqual(result["training_elapsed_seconds"], 12.0)
            self.assertEqual(result["training_interactions_per_second"], 8.0)

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
                "validation_window_ids": ["2025-05"],
                "validation": validation,
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
