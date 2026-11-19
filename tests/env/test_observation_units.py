"""Observation entries stay in the simulator's own units: MW, work units, and hours."""

from __future__ import annotations

import math
import unittest

import numpy as np

from env.environment import RampAwareEnv
from env.models import FORECAST_HOURS, HISTORY_HOURS
from env.panel import CanonicalMarketPanel
from tests.env.test_continuous_month import MARKET, SITE, STATS, make_panel, make_workload


def as_float32(value: float) -> float:
    return float(np.float32(value))


class ObservationUnitTests(unittest.TestCase):
    def test_entries_are_physical_values(self) -> None:
        hours, steps = 30, 5
        frame = make_panel(hours).frame.copy()
        for quantity, column in (("gross", "gross_demand_mw"), ("net", "net_load_mw")):
            for lead in FORECAST_HOURS:
                # Forecast legs of 1, 4, 9, and 18 MW per hour, so the steepest
                # forecast climb is 18 MW/h at every issue hour.
                frame[f"forecast_{quantity}_h{lead}_mw"] = frame[column] + lead**2
        env = RampAwareEnv(
            CanonicalMarketPanel(frame), [SITE], make_workload(hours, {steps: 0.2}, 12), STATS
        )
        try:
            env.reset(seed=0)
            for _ in range(steps):
                observation, _, _, _, info = env.step(env.evaluation_action("status_quo"))
            schema = env.observation_schema

            def value(name: str) -> float:
                return float(observation[schema.index(name)])

            net = env.panel.frame["net_load_mw"].to_numpy()
            gross = env.panel.frame["gross_demand_mw"].to_numpy()
            latest = HISTORY_HOURS + steps - 1
            self.assertEqual(value(f"{MARKET}:gross_mw_lag1"), as_float32(gross[latest]))
            self.assertEqual(value(f"{MARKET}:net_mw_lag1"), as_float32(net[latest]))
            self.assertEqual(value(f"{MARKET}:forecast_net_t+0_mw"), as_float32(net[latest] + 1.0))
            for quantity in ("gross", "net"):
                self.assertAlmostEqual(
                    value(f"{MARKET}:forecast_{quantity}_max_ramp_rate_mw_per_hour"), 18.0, places=3
                )
            market = info["per_market"][MARKET]
            self.assertEqual(
                value(f"{MARKET}:running_adjusted_peak_mw"), as_float32(market["running_adjusted_peak_mw"])
            )
            self.assertEqual(
                value(f"{MARKET}:running_net_load_peak_mw"),
                as_float32(net[HISTORY_HOURS:HISTORY_HOURS + steps].max()),
            )
            self.assertEqual(
                value("site-M:previous_power_mw"), as_float32(info["per_site"]["site-M"]["power_mw"])
            )
            self.assertEqual(value("site-M:service_workload"), as_float32(0.4))
            self.assertEqual(value("site-M:batch_workload"), as_float32(0.2))
            self.assertEqual(value("remaining_hours"), float(hours - steps))
            self.assertAlmostEqual(
                value("hour_sin"),
                math.sin(2.0 * math.pi * env.current_timestamp.hour / 24.0),
                places=6,
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
