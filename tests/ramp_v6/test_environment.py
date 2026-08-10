"""Causality, anti-gaming, and scale tests for the v6 environment."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from env.ramp_v6.environment import RampAwareEnv
from env.ramp_v6.fixture import load_fixture
from env.ramp_v6.models import FrozenRampStats, SiteConfig, WorkloadTrace
from env.ramp_v6.panel import CanonicalMarketPanel
from tests.ramp_v6.common import FIXTURE_ROOT, make_env


class EnvironmentTests(unittest.TestCase):
    def test_state_contract_has_required_causal_features(self) -> None:
        env = make_env()
        observation, _ = env.reset()
        schema = env.observation_schema
        required_fragments = (
            "gross_level_z_lag3",
            "net_level_z_lag3",
            "native_ramp_1h",
            "native_ramp_3h",
            "forecast_net_h3_z",
            "forecast_net_max_up_3h",
            "previous_power_rated_fraction",
            "queued_batch_capacity_fraction",
            "compute_capacity",
            "da_lmp",
            "hour_sin",
        )
        for fragment in required_fragments:
            self.assertTrue(any(fragment in name for name in schema), fragment)
        self.assertFalse(any("future_realized" in name for name in schema))
        self.assertFalse(any("rt_price" in name for name in schema))
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(len(env.action_schema), 2 * env.n_sites + 1)
        self.assertFalse(any("origin_preference" in name for name in env.action_schema))

    def test_no_wrap_and_real_three_hour_warm_history(self) -> None:
        env = make_env()
        observation, _ = env.reset()
        index = env.observation_schema.index(
            "A:native_ramp_3h_fraction_s_per_hour"
        )
        expected = (830.0 - 800.0) / (1200.0 * 3.0)
        self.assertAlmostEqual(float(observation[index]), expected, places=7)

    def test_future_realizations_do_not_change_current_state_or_reward(self) -> None:
        panel, sites, workload, stats, protocol = load_fixture(FIXTURE_ROOT)
        mutated = panel.frame.copy()
        current_timestamp = panel.timestamps[protocol.history_hours]
        future_mask = mutated["timestamp_utc"] > current_timestamp
        mutated.loc[future_mask, "gross_demand_mw"] += 9999.0
        mutated.loc[future_mask, "net_load_mw"] -= 7777.0
        env_a = RampAwareEnv(panel, sites, workload, stats, protocol)
        env_b = RampAwareEnv(
            CanonicalMarketPanel(mutated), sites, workload, stats, protocol
        )
        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()
        np.testing.assert_array_equal(obs_a, obs_b)
        action = np.zeros(env_a.action_space.shape, dtype=np.float64)
        _, reward_a, _, _, info_a = env_a.step(action)
        _, reward_b, _, _, info_b = env_b.step(action)
        self.assertAlmostEqual(reward_a, reward_b)
        self.assertEqual(
            info_a["weighted_incremental_ramp_impact"],
            info_b["weighted_incremental_ramp_impact"],
        )

    def test_future_issued_forecast_is_rejected(self) -> None:
        panel, _, _, _, _ = load_fixture(FIXTURE_ROOT)
        mutated = panel.frame.copy()
        mutated.loc[0, "forecast_issue_time_utc"] = (
            mutated.loc[0, "timestamp_utc"] + np.timedelta64(1, "h")
        )
        with self.assertRaisesRegex(ValueError, "issue time"):
            CanonicalMarketPanel(mutated)

    def test_tail_has_no_arrivals_and_all_work_completes(self) -> None:
        env = make_env()
        observation, _ = env.reset()
        history = []
        while True:
            action = np.concatenate(
                [
                    np.zeros(env.n_sites),
                    np.asarray([-6.0]),
                    np.zeros(env.n_sites),
                ]
            )
            observation, _, terminated, _, info = env.step(action)
            history.append(info)
            if terminated:
                break
        self.assertEqual(history[-1]["batch_queue"]["queued"], 0.0)
        self.assertAlmostEqual(
            history[-1]["batch_queue"]["conservation_error"], 0.0
        )
        self.assertTrue(
            all(
                item["new_arrivals"] == 0.0
                for item in history
                if item["terminal_tail_active"]
            )
        )
        self.assertTrue(
            all(min(item["capacity_slack"]) >= -1e-9 for item in history)
        )
        self.assertTrue(
            all(item["deadline_missed_work"] == 0.0 for item in history)
        )

    def test_physical_scale_multiplies_work_and_power_consistently(self) -> None:
        base = make_env(1.0)
        stress = make_env(10.0)
        obs_base, _ = base.reset()
        obs_stress, _ = stress.reset()
        action = np.zeros(base.action_space.shape, dtype=np.float64)
        _, _, _, _, info_base = base.step(action)
        _, _, _, _, info_stress = stress.step(action)
        for site_id in ("site-a", "site-b"):
            self.assertAlmostEqual(
                info_stress["per_site"][site_id]["work"],
                10.0 * info_base["per_site"][site_id]["work"],
            )
            self.assertAlmostEqual(
                info_stress["per_site"][site_id]["power_mw"],
                10.0 * info_base["per_site"][site_id]["power_mw"],
            )
        self.assertAlmostEqual(
            info_stress["per_market"]["A"]["power_over_frozen_s"],
            10.0 * info_base["per_market"]["A"]["power_over_frozen_s"],
        )
        self.assertAlmostEqual(
            info_stress["per_market"]["A"]["power_over_gross_demand"],
            10.0 * info_base["per_market"]["A"]["power_over_gross_demand"],
        )

    def test_market_power_is_aggregated_once(self) -> None:
        env = make_env()
        observation, _ = env.reset()
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
        for market in ("A", "B"):
            site_total = sum(
                row["power_mw"]
                for row in info["per_site"].values()
                if row["market_id"] == market
            )
            self.assertAlmostEqual(
                info["per_market"][market]["power_mw"], site_total
            )

    def test_six_markets_and_one_gw_total_scale(self) -> None:
        panel, _, workload, stats, protocol = load_fixture(FIXTURE_ROOT)
        source = panel.frame[panel.frame["market_id"] == "A"]
        frames = []
        markets = [f"M{index}" for index in range(6)]
        for market in markets:
            rows = source.copy()
            rows["market_id"] = market
            rows["forecast_vintage_id"] = market + "-" + rows[
                "forecast_vintage_id"
            ]
            frames.append(rows)
        six_panel = CanonicalMarketPanel(pd.concat(frames, ignore_index=True))
        multiplier = 1000.0 / 600.0
        sites = [
            SiteConfig(site_id=f"site-{index}", market_id=market).scaled(
                multiplier
            )
            for index, market in enumerate(markets)
        ]
        six_workload = WorkloadTrace(
            service_arrivals=np.repeat(
                workload.service_arrivals[:, :1], 6, axis=1
            ),
            batch_arrivals=np.repeat(
                workload.batch_arrivals[:, :1], 6, axis=1
            ),
            batch_deadline_hours=np.repeat(
                workload.batch_deadline_hours[:, :1], 6, axis=1
            ),
            warm_power_mw=np.repeat(workload.warm_power_mw[:, :1], 6, axis=1),
        ).scaled(multiplier)
        six_stats = FrozenRampStats(
            fit_start_utc=stats.fit_start_utc,
            fit_end_utc=stats.fit_end_utc,
            gross_q95_mw={market: stats.gross_q95_mw["A"] for market in markets},
            gross_level_mean_mw={
                market: stats.gross_level_mean_mw["A"] for market in markets
            },
            gross_level_std_mw={
                market: stats.gross_level_std_mw["A"] for market in markets
            },
            net_level_mean_mw={
                market: stats.net_level_mean_mw["A"] for market in markets
            },
            net_level_std_mw={
                market: stats.net_level_std_mw["A"] for market in markets
            },
            native_abs_ramp_q90_fraction_s_per_hour={
                market: stats.native_abs_ramp_q90_fraction_s_per_hour["A"]
                for market in markets
            },
            fit_months=stats.fit_months,
        )
        env = RampAwareEnv(
            six_panel, sites, six_workload, six_stats, protocol
        )
        observation, _ = env.reset()
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
        self.assertEqual(env.n_sites, 6)
        self.assertEqual(env.action_space.shape, (13,))
        self.assertAlmostEqual(sum(site.rated_power_mw for site in sites), 1000.0)
        self.assertEqual(set(info["per_market"]), set(markets))

    def test_out_of_bounds_action_is_rejected(self) -> None:
        env = make_env()
        env.reset()
        action = np.zeros(env.action_space.shape)
        action[0] = 6.01
        with self.assertRaisesRegex(ValueError, "outside"):
            env.step(action)


if __name__ == "__main__":
    unittest.main()
