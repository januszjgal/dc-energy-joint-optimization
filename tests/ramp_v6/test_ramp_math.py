"""Mathematical identity and action-projection tests for ramp v6."""

from __future__ import annotations

import unittest

import numpy as np

from env.ramp_v6.models import EDFQueue
from env.ramp_v6.projection import (
    exact_transport,
    project_action,
    project_capped_simplex,
)
from env.ramp_v6.reward import closed_window_terms


class RampMathTests(unittest.TestCase):
    def test_constant_power_has_zero_incremental_impact(self) -> None:
        terms = closed_window_terms(110.0, 100.0, 25.0, 25.0, 200.0, 1)
        self.assertAlmostEqual(terms.incremental_squared_impact, 0.0)
        self.assertEqual(
            tuple(terms.__dict__),
            (
                "native_fraction_s_per_hour",
                "adjusted_fraction_s_per_hour",
                "incremental_squared_impact",
            ),
        )

    def test_lowering_power_on_upward_native_ramp_earns_credit(self) -> None:
        terms = closed_window_terms(110.0, 100.0, 10.0, 15.0, 200.0, 1)
        self.assertLess(terms.incremental_squared_impact, 0.0)

    def test_overshoot_and_symmetric_square_are_penalized(self) -> None:
        upward = closed_window_terms(110.0, 100.0, 40.0, 10.0, 200.0, 1)
        downward = closed_window_terms(90.0, 100.0, 10.0, 40.0, 200.0, 1)
        self.assertGreater(upward.incremental_squared_impact, 0.0)
        self.assertGreater(downward.incremental_squared_impact, 0.0)
        self.assertAlmostEqual(
            upward.incremental_squared_impact,
            downward.incremental_squared_impact,
        )

    def test_capped_simplex_and_transport_are_exact(self) -> None:
        allocation = project_capped_simplex(
            np.asarray([3.0, 1.0, 0.0]),
            2.0,
            np.asarray([0.5, 1.0, 2.0]),
        )
        self.assertAlmostEqual(float(allocation.sum()), 2.0)
        self.assertTrue(np.all(allocation >= 0.0))
        self.assertTrue(np.all(allocation <= [0.5, 1.0, 2.0]))
        flow = exact_transport(
            np.asarray([0.3, 0.7]), np.asarray([0.4, 0.6])
        )
        np.testing.assert_allclose(flow.sum(axis=1), [0.3, 0.7])
        np.testing.assert_allclose(flow.sum(axis=0), [0.4, 0.6])

    def test_edf_origin_drain_is_deterministic(self) -> None:
        queue = EDFQueue()
        queue.add(0.4, origin=1, deadline_step=3)
        queue.add(0.2, origin=0, deadline_step=2)
        queue.add(0.3, origin=0, deadline_step=3)
        drained = queue.drain(0.5, n_origins=2)
        np.testing.assert_allclose(drained, [0.5, 0.0])
        self.assertAlmostEqual(queue.conservation_error(), 0.0)

    def test_future_capacity_prevents_terminal_dumping(self) -> None:
        queue = EDFQueue()
        queue.add(3.0, origin=0, deadline_step=9)
        result = project_action(
            np.asarray([0.0, -6.0, 0.0]),
            service_total=0.0,
            capacity=np.asarray([1.0]),
            queue=queue,
            current_step=6,
            final_step=9,
            n_origins=1,
            guaranteed_future_batch_capacity_by_deadline={9: 2.0},
        )
        self.assertAlmostEqual(result.mandatory_batch, 1.0)
        self.assertAlmostEqual(float(result.batch_by_destination.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
