"""Regression against the unchanged thesis's full-month July worked example."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from energy_model_v3.four_market_v2 import FACTORY_ROOT, MARKETS, objective_from_factory
from env.ramp_v6.objective import update_peak
from env.ramp_v6.panel import CanonicalMarketPanel
from scripts.build_four_market_v2_factory import _raw_arrivals, _site_configs, build


ROOT = Path(__file__).resolve().parents[2]


class MonthlyRampImpactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build()
        cls.factory = json.loads((FACTORY_ROOT / "factory.json").read_text())
        cls.objective = objective_from_factory(cls.factory)

    def test_references_match_positive_training_month_totals(self) -> None:
        self.assertAlmostEqual(self.objective.ramp_reference, 6.034057752312674, places=11)
        self.assertAlmostEqual(self.objective.peak_reference, 3.664598536383508, places=11)
        calibration = self.factory["objective_calibration"]
        self.assertEqual(len(calibration["fit_months"]), 11)
        self.assertNotIn("2025-05", calibration["fit_months"])
        scores = calibration["reference_scores_by_month"]
        self.assertTrue(all(row["ramp_squared_sum"] > 0.0 for row in scores.values()))
        self.assertAlmostEqual(
            self.objective.ramp_reference,
            np.mean([row["ramp_squared_sum"] for row in scores.values()]),
            places=12,
        )

    def test_july_routing_matches_the_paper_and_includes_the_rebound(self) -> None:
        panel = CanonicalMarketPanel.from_csv(
            ROOT / "data" / "four_market_2025" / "months" / "2025-07" / "canonical_panel.csv"
        )
        sites = _site_configs()
        service, batch = _raw_arrivals(panel.timestamps)
        work = service + batch
        routed = work.copy()
        index = int(panel.timestamps.get_loc(pd.Timestamp("2025-07-29T22:00:00Z")))
        isone, caiso = MARKETS.index("ISONE_NEMA"), MARKETS.index("CAISO_NP15")
        amount = service[index, isone]
        routed[index, isone] -= amount
        routed[index, caiso] += amount
        np.testing.assert_allclose(work.sum(axis=1), routed.sum(axis=1))
        self.assertTrue(np.all(routed >= 0.0))
        self.assertTrue(np.all(routed <= 1.0))
        np.testing.assert_array_equal(routed[index + 1], work[index + 1])

        net = panel.frame.pivot(
            index="timestamp_utc", columns="market_id", values="net_load_mw"
        ).loc[panel.timestamps, list(MARKETS)].to_numpy()
        stats = json.loads((ROOT / "data" / "four_market_2025" / "frozen_stats.json").read_text())
        scales = np.asarray([stats["gross_q95_mw"][market] for market in MARKETS])
        native_ramps = np.diff(net, axis=0) / scales
        results = []
        for executed in (work, routed):
            power = np.asarray([
                [site.power_mw(float(value)) for site, value in zip(sites, row)]
                for row in executed
            ])
            adjusted = net + power
            adjusted_ramps = np.diff(adjusted, axis=0) / scales
            impacts = adjusted_ramps**2 - native_ramps**2
            impact_sum = float(impacts.sum())
            peaks = adjusted[1:].max(axis=0)
            original_peaks = net[1:].max(axis=0)
            peak_impact = float(((peaks - original_peaks) / scales).sum())
            score = self.objective.score(impact_sum, peak_impact)
            previous = [None] * len(MARKETS)
            previous_original = [None] * len(MARKETS)
            raw_return = 0.0
            for slot in range(1, len(adjusted)):
                peak_increment = 0.0
                for market in range(len(MARKETS)):
                    previous[market], increase = update_peak(
                        previous[market], float(adjusted[slot, market] / scales[market])
                    )
                    previous_original[market], original_increase = update_peak(
                        previous_original[market], float(net[slot, market] / scales[market])
                    )
                    peak_increment += increase - original_increase
                raw_return += sum(self.objective.reward_components(
                    float(impacts[slot - 1].sum()), peak_increment
                ))
            self.assertAlmostEqual(raw_return, -score, places=12)
            self.assertEqual(len(impacts), 744)
            results.append((score, peaks, impact_sum))

        baseline, shifted = results
        self.assertAlmostEqual(baseline[0], 0.005620865581107415, places=10)
        self.assertAlmostEqual(shifted[0], 0.004800576997076811, places=10)
        self.assertAlmostEqual(baseline[0] - shifted[0], 0.000820288584030604, places=10)
        self.assertAlmostEqual(baseline[1][isone], 24432.61158786, places=6)
        self.assertAlmostEqual(shifted[1][isone], 24331.40723248, places=6)
        other_regions = [index for index in range(len(MARKETS)) if index != isone]
        np.testing.assert_allclose(baseline[1][other_regions], shifted[1][other_regions])
        self.assertLess(shifted[2], baseline[2])


if __name__ == "__main__":
    unittest.main()
