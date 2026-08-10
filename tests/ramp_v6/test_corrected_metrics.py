"""Regression tests for corrected V4R physical and comparator metrics."""

from __future__ import annotations

import unittest

import numpy as np

from env.ramp_v6.models import EDFQueue
from env.ramp_v6.projection import project_action
from ramp_rl.evaluation import _aggregate, _collect_per_market_metrics


def _episode(
    *,
    per_market: dict[str, list[float]],
    abs_h1: dict[str, list[float]],
    abs_h3: dict[str, list[float]],
    adjustments: list[float],
    ramp_power: float,
    energy_cost: float = 1.0,
) -> dict[str, object]:
    step_count = len(adjustments)
    incremental = [
        float(np.mean([values[index] for values in per_market.values()]))
        for index in range(len(next(iter(per_market.values()))))
    ]
    return {
        "month": 3,
        "day_group": "2026-03-01",
        "evaluation_seed": 2800,
        "incremental": incremental,
        "per_market_incremental": per_market,
        "abs_adjusted_ramp_h1_by_market": abs_h1,
        "abs_adjusted_ramp_h3_by_market": abs_h3,
        "ramp_h1": [0.0] * step_count,
        "ramp_h3": [0.0] * step_count,
        "energy_cost": energy_cost,
        "service_unserved": 0.0,
        "batch_unfinished": 0.0,
        "batch_expired": 0.0,
        "certificate_violations": 0,
        "terminal_work": 0.0,
        "emergency_count": 0,
        "step_count": step_count,
        "semantic_adjustment_l2_values": adjustments,
        "semantic_adjustment_l2": sum(adjustments),
        "future_realized_features_exposed": False,
        "deferrable_pre_service": 1.0,
        "ramp_power": ramp_power,
    }


class CorrectedPhysicalRampTests(unittest.TestCase):
    def test_opposite_market_signs_cannot_cancel_physical_metrics(self) -> None:
        infos = [
            {
                "ramp_h1_adjusted": 0.0,
                "ramp_h3_adjusted": 0.0,
                "per_market": {
                    "UP": {
                        "windows": {
                            "1h": {
                                "adjusted_fraction_s_per_hour": 0.8,
                                "incremental_squared_impact": -0.2,
                            },
                            "3h": {
                                "adjusted_fraction_s_per_hour": 0.4,
                                "incremental_squared_impact": -0.1,
                            },
                        }
                    },
                    "DOWN": {
                        "windows": {
                            "1h": {
                                "adjusted_fraction_s_per_hour": -0.8,
                                "incremental_squared_impact": -0.2,
                            },
                            "3h": {
                                "adjusted_fraction_s_per_hour": -0.4,
                                "incremental_squared_impact": -0.1,
                            },
                        }
                    },
                },
            }
        ]
        incremental, abs_h1, abs_h3 = _collect_per_market_metrics(infos)
        self.assertEqual(abs_h1, {"UP": [0.8], "DOWN": [0.8]})
        self.assertEqual(abs_h3, {"UP": [0.4], "DOWN": [0.4]})

        policy = _episode(
            per_market=incremental,
            abs_h1=abs_h1,
            abs_h3=abs_h3,
            adjustments=[0.0],
            ramp_power=1.0,
        )
        baseline = _episode(
            per_market={"UP": [-0.1], "DOWN": [-0.1]},
            abs_h1={"UP": [0.1], "DOWN": [0.1]},
            abs_h3={"UP": [0.1], "DOWN": [0.1]},
            adjustments=[0.0],
            ramp_power=2.0,
        )
        summary = _aggregate([policy], [baseline], "test")
        self.assertEqual(
            summary[
                "abs_adjusted_ramp_h1_fraction_s_per_hour_p95"
            ],
            0.8,
        )
        self.assertEqual(
            summary[
                "abs_adjusted_ramp_h1_fraction_s_per_hour_max"
            ],
            0.8,
        )
        self.assertFalse(
            summary["physical_ramp_metric_contract"][
                "cross_market_signed_averaging"
            ]
        )


class StatusQuoComparisonTests(unittest.TestCase):
    def test_native_relative_gate_is_distinct_from_status_quo_diagnostic(
        self,
    ) -> None:
        policy = _episode(
            per_market={"A": [-2.0], "B": [-1.0]},
            abs_h1={"A": [0.2], "B": [0.3]},
            abs_h3={"A": [0.1], "B": [0.2]},
            adjustments=[0.0],
            ramp_power=1.0,
        )
        baseline = _episode(
            per_market={"A": [-1.0], "B": [-2.0]},
            abs_h1={"A": [0.2], "B": [0.3]},
            abs_h3={"A": [0.1], "B": [0.2]},
            adjustments=[0.0],
            ramp_power=2.0,
        )
        summary = _aggregate([policy], [baseline], "test")
        comparison = summary["status_quo_comparison"]
        self.assertEqual(comparison["markets_better_count"], 1)
        self.assertEqual(comparison["markets_better_share"], 0.5)
        self.assertFalse(comparison["every_market_outperforms_status_quo"])
        self.assertTrue(
            summary["success_gate"]["checks"][
                "every_market_native_relative_incremental_ramp_improves"
            ]
        )
        self.assertTrue(summary["success_gate"]["passed"])
        self.assertEqual(
            summary["success_gate"]["compatibility_aliases"],
            {
                "every_market_ramp_improves": (
                    "every_market_native_relative_incremental_ramp_improves"
                )
            },
        )


class SemanticAdjustmentTests(unittest.TestCase):
    def test_unchanged_projection_has_exact_zero_adjustment(self) -> None:
        queue = EDFQueue()
        queue.add(1.0, origin=0, deadline_step=10)
        projected = project_action(
            np.zeros(5, dtype=np.float64),
            service_total=1.0,
            capacity=np.asarray([1.0, 1.0]),
            queue=queue,
            current_step=0,
            final_step=20,
            n_origins=2,
            guaranteed_future_batch_capacity_by_deadline={10: 2.0},
        )
        self.assertEqual(projected.semantic_adjustment_l2(), 0.0)

    def test_binding_projection_has_known_positive_adjustment(self) -> None:
        action = np.asarray([6.0, -6.0, -6.0, 0.0, 0.0])
        projected = project_action(
            action,
            service_total=0.5,
            capacity=np.asarray([0.2, 0.8]),
            queue=EDFQueue(),
            current_step=0,
            final_step=20,
            n_origins=2,
        )
        expected = float(
            np.linalg.norm(
                projected.projected_semantic_allocation()
                - projected.requested_semantic_allocation()
            )
        )
        self.assertGreater(expected, 0.0)
        self.assertEqual(projected.semantic_adjustment_l2(), expected)

    def test_evaluator_aggregates_real_adjustment_distribution(self) -> None:
        policy = _episode(
            per_market={"A": [-1.0, -1.0, -1.0]},
            abs_h1={"A": [0.1, 0.2, 0.3]},
            abs_h3={"A": [0.2, 0.3, 0.4]},
            adjustments=[0.0, 3.0, 4.0],
            ramp_power=1.0,
        )
        baseline = _episode(
            per_market={"A": [-0.5, -0.5, -0.5]},
            abs_h1={"A": [0.1, 0.2, 0.3]},
            abs_h3={"A": [0.2, 0.3, 0.4]},
            adjustments=[0.0, 0.0, 0.0],
            ramp_power=2.0,
        )
        summary = _aggregate([policy], [baseline], "test")
        adjustment = summary["semantic_adjustment"]
        self.assertEqual(summary["semantic_adjustment_l2"], 7.0)
        self.assertAlmostEqual(adjustment["mean_l2"], 7.0 / 3.0)
        self.assertEqual(adjustment["max_l2"], 4.0)
        self.assertEqual(adjustment["positive_adjustment_count"], 2)
        self.assertAlmostEqual(adjustment["adjustment_rate"], 2.0 / 3.0)
        self.assertEqual(adjustment["emergency_fallback_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
