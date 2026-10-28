"""The raw observation is the scaled one before the environment's own scaling."""

from __future__ import annotations

import unittest

import numpy as np

from env.ramp_v6.environment import RampAwareEnv
from tests.ramp_v6.test_continuous_month import SITE, STATS, make_panel, make_workload


class RawObservationTests(unittest.TestCase):
    def test_raw_entries_are_unscaled_versions_of_the_scaled_ones(self) -> None:
        hours = 30
        envs = {
            raw: RampAwareEnv(make_panel(hours), [SITE], make_workload(hours, {3: 0.2}, 12), STATS,
                              raw_observation=raw)
            for raw in (False, True)
        }
        try:
            obs = {raw: env.reset(seed=0)[0] for raw, env in envs.items()}
            for _ in range(5):
                obs = {raw: env.step(env.evaluation_action("status_quo"))[0] for raw, env in envs.items()}
            schema = envs[False].observation_schema
            scaled, raw = obs[False].astype(np.float64), obs[True].astype(np.float64)
            for name in ("M:net_level_z_lag1", "M:forecast_net_t+0_z"):
                i = schema.index(name)
                self.assertAlmostEqual(scaled[i], (raw[i] - STATS.net_level_mean_mw["M"]) / STATS.net_level_std_mw["M"], places=4)
            for name in ("M:running_adjusted_peak_fraction_s", "M:forecast_net_max_up_fraction_s"):
                i = schema.index(name)
                self.assertAlmostEqual(scaled[i], raw[i] / STATS.gross_q95_mw["M"], places=5)
            i = schema.index("site-M:previous_power_rated_fraction")
            self.assertAlmostEqual(scaled[i], raw[i] / SITE.rated_power_mw, places=5)
            i = schema.index("remaining_month_fraction")
            self.assertAlmostEqual(scaled[i], raw[i] / hours, places=6)
            i = schema.index("hour_sin")
            self.assertEqual(scaled[i], raw[i])
        finally:
            for env in envs.values():
                env.close()


if __name__ == "__main__":
    unittest.main()
