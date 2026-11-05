"""Regression against the thesis's full-month July worked example (trace clock, UTC-7)."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from energy_model_v3.four_market_v2 import MARKETS, factory_root, objective_from_factory
from energy_model_v3.splits import DEVELOPMENT_MONTHS, FINAL_SPEC
from env.ramp_v6.objective import update_peak
from env.ramp_v6.panel import CanonicalMarketPanel
from scripts.build_four_market_v2_factory import _raw_arrivals, _site_configs, build


ROOT = Path(__file__).resolve().parents[2]


class MonthlyRampImpactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build(FINAL_SPEC)
        cls.factory = json.loads((factory_root(FINAL_SPEC) / "factory.json").read_text())
        cls.objective = objective_from_factory(cls.factory)

    def test_references_match_positive_training_month_totals(self) -> None:
        self.assertAlmostEqual(self.objective.ramp_reference, 6.102257730896831, places=11)
        self.assertAlmostEqual(self.objective.peak_reference, 3.653738230091554, places=11)
        calibration = self.factory["objective_calibration"]
        self.assertEqual(calibration["fit_months"], list(DEVELOPMENT_MONTHS))
        scores = calibration["reference_scores_by_month"]
        self.assertTrue(all(row["ramp_squared_sum"] > 0.0 for row in scores.values()))
        self.assertAlmostEqual(
            self.objective.ramp_reference,
            np.mean([row["ramp_squared_sum"] for row in scores.values()]),
            places=12,
        )

    def test_july_example_matches_the_paper_and_includes_the_rebound(self) -> None:
        panel = CanonicalMarketPanel.from_csv(
            ROOT / "data" / "four_market_2025" / "months" / "2025-07" / "canonical_panel.csv"
        )
        sites = _site_configs()
        service, batch = _raw_arrivals(panel.timestamps)
        work = service + batch
        routed = work.copy()
        index = int(panel.timestamps.get_loc(pd.Timestamp("2025-07-29T22:00:00Z")))
        later = int(panel.timestamps.get_loc(pd.Timestamp("2025-07-30T08:00:00Z")))
        isone, miso = MARKETS.index("ISONE"), MARKETS.index("MISO")
        # ISO-NE's service runs at MISO in the same hour; its batch waits for the
        # window's lowest net load and runs at ISO-NE.
        routed[index, isone] -= service[index, isone] + batch[index, isone]
        routed[index, miso] += service[index, isone]
        routed[later, isone] += batch[index, isone]
        np.testing.assert_allclose(work.sum(), routed.sum())
        self.assertLessEqual(later - index, 23)
        self.assertTrue(np.all(routed >= -1e-12))
        self.assertTrue(np.all(routed <= np.asarray([site.compute_capacity for site in sites]) + 1e-12))
        np.testing.assert_array_equal(routed[index + 1], work[index + 1])

        net = panel.frame.pivot(
            index="timestamp_utc", columns="market_id", values="net_load_mw"
        ).loc[panel.timestamps, list(MARKETS)].to_numpy()
        stats = json.loads((ROOT / "data" / "four_market_2025" / "frozen_stats.json").read_text())
        scales = np.asarray([stats["gross_q95_mw"][market] for market in MARKETS])
        no_flex_power = np.asarray([
            [site.power_mw(float(value)) for site, value in zip(sites, row)]
            for row in work
        ])
        no_flex = net + no_flex_power
        no_flex_ramps = np.diff(no_flex, axis=0) / scales
        no_flex_peaks = no_flex[1:].max(axis=0)
        results = []
        for executed in (work, routed):
            power = np.asarray([
                [site.power_mw(float(value)) for site, value in zip(sites, row)]
                for row in executed
            ])
            adjusted = net + power
            adjusted_ramps = np.diff(adjusted, axis=0) / scales
            impacts = adjusted_ramps**2 - no_flex_ramps**2
            impact_sum = float(impacts.sum())
            peaks = adjusted[1:].max(axis=0)
            peak_impact = float(((peaks - no_flex_peaks) / scales).sum())
            score = self.objective.score(impact_sum, peak_impact)
            previous = [None] * len(MARKETS)
            previous_no_flex = [None] * len(MARKETS)
            raw_return = 0.0
            for slot in range(1, len(adjusted)):
                peak_increment = 0.0
                for market in range(len(MARKETS)):
                    previous[market], increase = update_peak(
                        previous[market], float(adjusted[slot, market] / scales[market])
                    )
                    previous_no_flex[market], no_flex_increase = update_peak(
                        previous_no_flex[market], float(no_flex[slot, market] / scales[market])
                    )
                    peak_increment += increase - no_flex_increase
                raw_return += sum(self.objective.reward_components(
                    float(impacts[slot - 1].sum()), peak_increment
                ))
            self.assertAlmostEqual(raw_return, -score, places=12)
            self.assertEqual(len(impacts), 744)
            results.append((score, peaks, impact_sum))

        baseline, shifted = results
        self.assertEqual(baseline[0], 0.0)
        self.assertAlmostEqual(shifted[0], -0.001169462869312912, places=10)
        self.assertAlmostEqual(baseline[1][isone], 24376.949089328282, places=6)
        self.assertAlmostEqual(shifted[1][isone], 24233.80698883537, places=6)
        other_regions = [index for index in range(len(MARKETS)) if index != isone]
        np.testing.assert_allclose(baseline[1][other_regions], shifted[1][other_regions])
        self.assertLess(shifted[2], baseline[2])


if __name__ == "__main__":
    unittest.main()
