"""Joint-score identities, peak semantics, causality and unchanged constraints."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from env.ramp_v6.environment import RampAwareEnv
from env.ramp_v6.models import RampProtocol
from env.ramp_v6.objective import JointObjective, trajectory_scores, update_peak
from env.ramp_v6.panel import CanonicalMarketPanel
from ramp_rl.contract import environment_identity
from ramp_rl.evaluation import _aggregate, _episode, evaluate_checkpoint
from tests.ramp_v6.test_continuous_month import (
    HOLD_BATCH, SITE, STATS, make_panel, make_workload,
)


class JointObjectiveMathTests(unittest.TestCase):
    def test_equal_proportional_reductions_have_equal_weight(self) -> None:
        objective = JointObjective(ramp_reference=2.0, peak_reference=10.0)
        self.assertAlmostEqual(objective.score(2.0, 10.0), 1.0)
        self.assertAlmostEqual(objective.score(1.8, 10.0), 0.95)
        self.assertAlmostEqual(objective.score(2.0, 9.0), 0.95)
        self.assertEqual(JointObjective.from_dict(objective.as_dict()), objective)

    def test_weights_and_references_reject_invalid_values(self) -> None:
        for kwargs in (
            {"ramp_weight": -0.5, "peak_weight": 1.5},
            {"ramp_weight": 0.5, "peak_weight": 0.6},
            {"peak_weight": float("nan")},
            {"ramp_reference": 0.0},
            {"peak_reference": -1.0},
            {"peak_reference": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                JointObjective(**kwargs).validate()

    def test_peak_increments_telescope_including_negative_first_value(self) -> None:
        peak = None
        increments = []
        for current in (-3.0, -5.0, 2.0, 1.0, 4.0):
            peak, increment = update_peak(peak, current)
            increments.append(increment)
        self.assertEqual(increments, [-3.0, 0.0, 5.0, 0.0, 2.0])
        self.assertEqual(sum(increments), 4.0)

    def test_peak_is_regional_month_maximum_and_excludes_warm_hour(self) -> None:
        net = np.asarray([[999.0, 999.0], [100.0, 200.0], [200.0, 100.0]])
        ramp, peak = trajectory_scores(net, np.zeros_like(net), np.asarray([100.0, 100.0]))
        self.assertEqual(peak, 4.0)
        expected_ramp = ((-8.99)**2 + (-7.99)**2 + 1.0**2 + (-1.0)**2) / 2
        self.assertAlmostEqual(ramp, expected_ramp)


class JointEnvironmentTests(unittest.TestCase):
    @staticmethod
    def make_env(weight: float = 0.5, hours: int = 48) -> RampAwareEnv:
        return RampAwareEnv(
            make_panel(hours),
            [SITE],
            make_workload(hours, {hour: 0.3 for hour in range(hours)}, 24),
            STATS,
            RampProtocol(objective=JointObjective(
                ramp_weight=weight, peak_weight=1.0 - weight,
                ramp_reference=0.02, peak_reference=1.2,
            )),
        )

    def test_episode_reward_equals_independently_computed_joint_score(self) -> None:
        for weight in (0.0, 0.5, 1.0):
            with self.subTest(weight=weight):
                env = self.make_env(weight)
                try:
                    observation, _ = env.reset(seed=7)
                    returns = 0.0
                    power = [float(env.workload.warm_power_mw[-1, 0])]
                    for step in range(env.action_steps):
                        observation, reward, done, _, info = env.step(HOLD_BATCH)
                        returns += reward
                        power.append(info["per_site"]["site-M"]["power_mw"])
                        if not done:
                            peak_index = env.observation_schema.index("M:running_adjusted_peak_fraction_s")
                            self.assertAlmostEqual(
                                float(observation[peak_index]),
                                info["per_market"]["M"]["running_adjusted_peak_mw"] / 1200.0,
                                places=6,
                            )
                    adjusted = env.panel.frame["net_load_mw"].to_numpy() + np.asarray(power)
                    ramp = float(np.mean((np.diff(adjusted) / 1200.0)**2))
                    peak = float(np.max(adjusted[1:]) / 1200.0)
                    expected = weight * ramp / 0.02 + (1.0 - weight) * peak / 1.2
                    self.assertAlmostEqual(returns, -expected, places=12)
                    self.assertAlmostEqual(env.queue.total, 0.0)
                finally:
                    env.close()

    def test_objective_weights_do_not_change_feasibility_projection(self) -> None:
        envs = [self.make_env(weight) for weight in (0.0, 0.5, 1.0)]
        try:
            for env in envs:
                env.reset(seed=9)
            rng = np.random.default_rng(9)
            for _ in range(48):
                action = rng.uniform(-6.0, 6.0, size=3).astype(np.float32)
                infos = [env.step(action)[4] for env in envs]
                for info in infos:
                    self.assertEqual(info["service_unserved"], 0.0)
                    self.assertEqual(info["batch_expired"], 0.0)
                    self.assertAlmostEqual(info["work_conservation_error"], 0.0)
                    self.assertTrue(all(value >= -1e-9 for value in info["capacity_slack"]))
                    np.testing.assert_allclose(info["service_allocation"], infos[0]["service_allocation"])
                    np.testing.assert_allclose(info["batch_by_destination"], infos[0]["batch_by_destination"])
            self.assertTrue(all(env.queue.total < 1e-9 for env in envs))
        finally:
            for env in envs:
                env.close()

    def test_peak_state_resets_and_warm_grid_is_not_a_peak_sample(self) -> None:
        frame = make_panel(4).frame.copy()
        frame.loc[0, "net_load_mw"] = 1e6
        env = RampAwareEnv(
            CanonicalMarketPanel(frame), [SITE], make_workload(4, {}, 24), STATS
        )
        try:
            first, _ = env.reset()
            peak_index = env.observation_schema.index("M:running_adjusted_peak_fraction_s")
            self.assertEqual(first[peak_index], 0.0)
            _, _, _, _, info = env.step(env.evaluation_action("status_quo"))
            self.assertLess(info["per_market"]["M"]["running_adjusted_peak_mw"], 2000.0)
            reset, _ = env.reset()
            np.testing.assert_array_equal(reset, first)
            self.assertEqual(reset[-1], 1.0)
        finally:
            env.close()

    def test_evaluation_keeps_joint_ramp_and_peak_scores_distinct(self) -> None:
        episode = _episode(
            factory=lambda request: self.make_env(hours=48),
            split="validation", seed=1, window_id="test",
            model=None, normalization_path=None,
        )
        summary = _aggregate([episode], [episode], "validation")
        comparison = summary["status_quo_comparison"]
        self.assertEqual(comparison["improvement_J"], 0.0)
        self.assertEqual(comparison["improvement"], 0.0)
        self.assertEqual(comparison["components"]["net_load_peak"]["improvement"], 0.0)
        self.assertEqual(comparison["per_market"]["M"]["peak_reduction_mw"], 0.0)
        self.assertNotEqual(episode["J"], episode["ramp_impact_J"])

    def test_evaluation_rejects_mismatched_objective_before_loading_model(self) -> None:
        env = self.make_env(weight=1.0)
        try:
            identity = environment_identity(env.ramp_rl_contract())
        finally:
            env.close()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "training_summary.json").write_text(
                json.dumps({"environment_identity": identity}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "checkpoint objective"):
                evaluate_checkpoint(
                    factory=lambda request: self.make_env(weight=0.5),
                    checkpoint_dir=checkpoint, split="validation", seeds=[1], windows=["test"],
                )


if __name__ == "__main__":
    unittest.main()
