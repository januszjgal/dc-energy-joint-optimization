"""Continuous-episode, causal-timeline, and deadline tests on a synthetic panel.

These tests use a one-market, one-site panel built in memory so that every
claim in the problem statement about the event order inside an hour, the
carry-over of state across midnight, and the month-end cut can be checked
without the four-market data.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from env.ramp_v6.environment import RampAwareEnv
from env.ramp_v6.models import (
    FORECAST_HOURS,
    HISTORY_HOURS,
    FrozenRampStats,
    RampProtocol,
    SiteConfig,
    WorkloadTrace,
)
from env.ramp_v6.panel import CanonicalMarketPanel
from ramp_rl.contract import EnvRequest, RampEnvAdapter


MARKET = "M"
START = pd.Timestamp("2025-03-01T00:00:00Z")


def make_panel(hours: int, *, perturb_from_row: int | None = None, delta: float = 0.0) -> CanonicalMarketPanel:
    """One-market panel with warm rows, persistence forecasts, and an optional perturbation.

    ``perturb_from_row`` counts panel rows from the first warm row, so row
    ``HISTORY_HOURS + k`` is decision hour ``k``.
    """
    n = HISTORY_HOURS + hours
    stamps = pd.date_range(START - pd.Timedelta(hours=HISTORY_HOURS), periods=n, freq="h", tz="UTC")
    k = np.arange(n, dtype=np.float64)
    gross = 1000.0 + 200.0 * np.sin(2.0 * np.pi * k / 24.0) + 3.0 * k
    wind = np.full(n, 50.0)
    solar = np.clip(100.0 * np.sin(2.0 * np.pi * (k - 6.0) / 24.0), 0.0, None)
    net = gross - wind - solar
    if perturb_from_row is not None:
        mask = np.arange(n) >= perturb_from_row
        gross = gross + delta * mask
        net = net + delta * mask
    frame = pd.DataFrame({
        "timestamp_utc": stamps,
        "market_id": MARKET,
        "gross_demand_mw": gross,
        "net_load_mw": net,
        "wind_mw": wind,
        "solar_mw": solar,
        "market_scale_mw": 1200.0,
        "forecast_issue_time_utc": stamps,
        "forecast_vintage_id": "persistence",
        "quality_ok": True,
    })
    for hour in FORECAST_HOURS:
        frame[f"forecast_gross_h{hour}_mw"] = gross
        frame[f"forecast_net_h{hour}_mw"] = net
    return CanonicalMarketPanel(frame)


STATS = FrozenRampStats(
    fit_start_utc="2025-02-01T00:00:00+00:00",
    fit_end_utc="2025-02-28T23:00:00+00:00",
    gross_q95_mw={MARKET: 1200.0},
    gross_level_mean_mw={MARKET: 1000.0},
    gross_level_std_mw={MARKET: 150.0},
    net_level_mean_mw={MARKET: 900.0},
    net_level_std_mw={MARKET: 150.0},
    fit_months=("2025-02",),
)
SITE = SiteConfig(
    site_id="site-M", market_id=MARKET, rated_power_mw=100.0, compute_capacity=1.0,
    idle_power_fraction=0.5, dynamic_power_fraction=0.5,
)


def make_workload(hours: int, batch: dict[int, float], window: int, service: float = 0.4) -> WorkloadTrace:
    service_arrivals = np.full((hours, 1), service)
    batch_arrivals = np.zeros((hours, 1))
    for hour, amount in batch.items():
        batch_arrivals[hour, 0] = amount
    return WorkloadTrace(
        service_arrivals=service_arrivals,
        batch_arrivals=batch_arrivals,
        batch_deadline_hours=np.full((hours, 1), window, dtype=np.int64),
        warm_power_mw=np.full((HISTORY_HOURS, 1), 70.0),
    )


def make_env(hours: int, batch: dict[int, float], window: int, **panel_kwargs) -> RampAwareEnv:
    return RampAwareEnv(make_panel(hours, **panel_kwargs), [SITE], make_workload(hours, batch, window), STATS)


HOLD_BATCH = np.asarray([0.0, -6.0, 0.0], dtype=np.float32)  # request zero optional batch
RELEASE_BATCH = np.asarray([0.0, 6.0, 0.0], dtype=np.float32)  # request all available batch


def run(env: RampAwareEnv, action_fn, hours: int) -> list[dict]:
    env.reset(seed=0)
    infos = []
    for _ in range(hours):
        _, _, terminated, _, info = env.step(action_fn(env))
        infos.append(info)
        if terminated:
            break
    return infos


class CausalTimelineTests(unittest.TestCase):
    def test_hour_t_observation_ignores_rows_at_or_after_hour_t(self) -> None:
        hours = 30
        for k in (0, 5, 20):
            clean = make_env(hours, {10: 0.3}, 3)
            leaky = make_env(hours, {10: 0.3}, 3, perturb_from_row=HISTORY_HOURS + k, delta=500.0)
            obs_clean, _ = clean.reset(seed=0)
            obs_leaky, _ = leaky.reset(seed=0)
            np.testing.assert_array_equal(obs_clean, obs_leaky)
            for step in range(hours):
                action = clean.evaluation_action("status_quo")
                obs_clean, reward_clean, done, _, info_clean = clean.step(action)
                obs_leaky, reward_leaky, _, _, info_leaky = leaky.step(action)
                ramp_clean = info_clean["per_market"][MARKET]["windows"]["1h"]["native_ramp_mw"]
                ramp_leaky = info_leaky["per_market"][MARKET]["windows"]["1h"]["native_ramp_mw"]
                if step < k:
                    # Rows at or after hour k are still in the future: nothing may differ.
                    self.assertEqual(reward_clean, reward_leaky, msg=f"k={k} step={step}")
                    self.assertEqual(ramp_clean, ramp_leaky, msg=f"k={k} step={step}")
                    np.testing.assert_array_equal(obs_clean, obs_leaky, err_msg=f"k={k} step={step}")
                elif step == k:
                    # Hour k's realized value is revealed only after the action, then
                    # scored: the realized ramp differs, and the next observation sees it.
                    self.assertAlmostEqual(ramp_leaky - ramp_clean, 500.0, msg=f"k={k}")
                    if not done:
                        self.assertFalse(np.array_equal(obs_clean, obs_leaky), msg=f"k={k}")
                    break

    def test_observation_schema_uses_completed_hours_and_forecasts_only(self) -> None:
        env = make_env(12, {}, 2)
        schema = env.observation_schema
        # Per market: t-1 gross and net levels, four forecasts and one max-up
        # rate per quantity. Nothing older than t-1 is observed.
        self.assertEqual(len(schema), 12 + 4 + 5 + 4)
        self.assertEqual(env.observation_space.shape, (len(schema),))
        self.assertFalse(any("lag0" in name or "lag2" in name for name in schema))
        self.assertFalse(any("native_ramp" in name for name in schema))
        self.assertTrue(all(f"{MARKET}:{q}_level_z_lag1" in schema for q in ("gross", "net")))
        self.assertIn(f"{MARKET}:forecast_net_t+0_z", schema)
        self.assertIn(f"{MARKET}:forecast_net_t+2_z", schema)
        self.assertFalse(any("episode_progress" in name or "terminal_tail" in name for name in schema))
        contract = env.ramp_rl_contract()
        self.assertEqual(contract["grid_observation_lag_hours"], 1)
        self.assertEqual(contract["terminal_tail_hours"], 0)
        self.assertEqual(contract["history_hours"], HISTORY_HOURS)
        RampEnvAdapter(env, EnvRequest(split="validation", seed=0))

    def test_forecast_names_and_values_use_lead_minus_issue_lag(self) -> None:
        frame = make_panel(26).frame.copy()
        row_offsets = np.arange(len(frame), dtype=float) / 100.0
        leads_and_offsets = ((1, 0), (3, 2), (6, 5), (12, 11))
        for quantity, mean, quantity_offset in (("gross", 1000.0, 0.0), ("net", 900.0, 0.3)):
            for lead, _ in leads_and_offsets:
                # Distinct leads and issue rows catch both column and timing mistakes.
                frame[f"forecast_{quantity}_h{lead}_mw"] = (
                    mean + 150.0 * (lead / 10.0 + row_offsets + quantity_offset)
                )
        env = RampAwareEnv(
            CanonicalMarketPanel(frame), [SITE], make_workload(26, {}, 24), STATS
        )
        try:
            observation, _ = env.reset(seed=0)
            schema = env.observation_schema
            for quantity in ("gross", "net"):
                self.assertEqual(
                    [name for name in schema if name.startswith(f"{MARKET}:forecast_{quantity}_t+")],
                    [f"{MARKET}:forecast_{quantity}_t+{ahead}_z" for _, ahead in leads_and_offsets],
                )
            for step in (0, 1):
                issue_row = HISTORY_HOURS + step - 1
                for quantity, quantity_offset in (("gross", 0.0), ("net", 0.3)):
                    for lead, ahead in leads_and_offsets:
                        name = f"{MARKET}:forecast_{quantity}_t+{ahead}_z"
                        expected = lead / 10.0 + issue_row / 100.0 + quantity_offset
                        self.assertAlmostEqual(float(observation[schema.index(name)]), expected, places=6)
                if step == 0:
                    observation, _, _, _, _ = env.step(env.evaluation_action("status_quo"))
        finally:
            env.close()


class ContinuityTests(unittest.TestCase):
    def test_batch_and_power_history_carry_across_midnight(self) -> None:
        hours = 30
        env = make_env(hours, {23: 0.3}, 3)
        infos = run(env, lambda _: HOLD_BATCH, hours)
        self.assertEqual(len(infos), hours)
        self.assertEqual(infos[23]["timestamp_utc"], "2025-03-01T23:00:00+00:00")
        self.assertEqual(infos[24]["timestamp_utc"], "2025-03-02T00:00:00+00:00")
        # Held through midnight, executed in the last permitted slot 23 + 3 - 1 = 25.
        self.assertAlmostEqual(infos[23]["batch_completed"], 0.0)
        self.assertAlmostEqual(infos[23]["batch_queue"]["queued"], 0.3)
        self.assertAlmostEqual(infos[24]["batch_completed"], 0.0)
        self.assertAlmostEqual(infos[24]["batch_queue"]["queued"], 0.3)
        self.assertAlmostEqual(infos[25]["batch_completed"], 0.3)
        self.assertAlmostEqual(infos[25]["batch_queue"]["queued"], 0.0)
        # The first ramp of the new day is measured against the previous day's last hour.
        net = env.panel.frame.set_index("timestamp_utc")["net_load_mw"]
        p23 = infos[23]["per_site"]["site-M"]["power_mw"]
        p24 = infos[24]["per_site"]["site-M"]["power_mw"]
        n23 = float(net.loc[pd.Timestamp("2025-03-01T23:00:00Z")])
        n24 = float(net.loc[pd.Timestamp("2025-03-02T00:00:00Z")])
        self.assertAlmostEqual(
            infos[24]["per_market"][MARKET]["windows"]["1h"]["adjusted_ramp_mw"],
            (n24 + p24) - (n23 + p23),
        )
        self.assertTrue(infos[-1]["actual_terminal"])

    def test_month_end_cuts_windows_so_queues_close_empty(self) -> None:
        hours = 10
        env = make_env(hours, {8: 0.2, 9: 0.1}, 3)
        infos = run(env, lambda _: HOLD_BATCH, hours)
        self.assertEqual(len(infos), hours)
        # Hour 8 may still use the final slot 9; hour 9 has no slack at all.
        self.assertAlmostEqual(infos[8]["batch_completed"], 0.0)
        self.assertAlmostEqual(infos[9]["batch_completed"], 0.3)
        self.assertAlmostEqual(infos[9]["batch_unfinished"], 0.0)
        self.assertEqual(infos[9]["batch_queue"]["queued"], 0.0)
        self.assertTrue(infos[9]["actual_terminal"])
        self.assertEqual(env.action_steps, hours)


class DeadlineSemanticsTests(unittest.TestCase):
    def test_24_hour_window_allows_immediate_or_optional_early_execution(self) -> None:
        for release_slot in (0, 10):
            with self.subTest(release_slot=release_slot):
                env = make_env(30, {0: 0.2}, 24)
                try:
                    actions = iter(
                        RELEASE_BATCH if hour == release_slot else HOLD_BATCH
                        for hour in range(30)
                    )
                    infos = run(env, lambda _: next(actions), 30)
                    expected = np.zeros(30)
                    expected[release_slot] = 0.2
                    np.testing.assert_allclose(
                        [info["batch_completed"] for info in infos], expected, atol=1e-12
                    )
                    self.assertAlmostEqual(infos[-1]["batch_queue"]["queued"], 0.0)
                finally:
                    env.close()

    def test_24_hour_window_forces_execution_in_slot_23(self) -> None:
        env = make_env(30, {0: 0.2}, 24)
        try:
            infos = run(env, lambda _: HOLD_BATCH, 30)
            expected = np.zeros(30)
            expected[23] = 0.2
            np.testing.assert_allclose(
                [info["batch_completed"] for info in infos], expected, atol=1e-12
            )
            self.assertAlmostEqual(infos[22]["batch_queue"]["queued"], 0.2)
            self.assertAlmostEqual(infos[23]["batch_queue"]["queued"], 0.0)
        finally:
            env.close()

    def test_24_hour_backlog_stays_feasible_and_clears_at_terminal(self) -> None:
        hours = 48
        env = make_env(hours, {hour: 0.3 for hour in range(hours)}, 24)
        try:
            infos = run(env, lambda _: HOLD_BATCH, hours)
            self.assertEqual(len(infos), hours)
            completed = 0.0
            for hour, info in enumerate(infos):
                completed += info["batch_completed"]
                due_count = sum(min(arrival + 23, hours - 1) <= hour for arrival in range(hours))
                self.assertGreaterEqual(completed + 1e-10, 0.3 * due_count)
                self.assertAlmostEqual(info["service_completed"], 0.4)
                self.assertAlmostEqual(info["batch_expired"], 0.0)
                self.assertEqual(info["certificate_violations"], 0)
                self.assertTrue(all(slack >= -1e-10 for slack in info["capacity_slack"]))
                self.assertAlmostEqual(info["work_conservation_error"], 0.0)
            self.assertGreater(max(info["batch_queue"]["queued"] for info in infos), 1.0)
            self.assertAlmostEqual(completed, 14.4)
            self.assertAlmostEqual(infos[-1]["batch_unfinished"], 0.0)
            self.assertAlmostEqual(infos[-1]["batch_queue"]["queued"], 0.0)
            self.assertTrue(infos[-1]["actual_terminal"])
        finally:
            env.close()

    def test_window_length_counts_the_arrival_slot(self) -> None:
        # H = 1 runs in its arrival slot; H = 2 may wait one slot, never two.
        for window, expected_slot in ((1, 5), (2, 6), (3, 7)):
            env = make_env(12, {5: 0.2}, window)
            infos = run(env, lambda _: HOLD_BATCH, 12)
            completed = [round(info["batch_completed"], 9) for info in infos]
            self.assertEqual(completed.index(0.2), expected_slot, msg=f"H={window}")
            self.assertEqual(sum(completed), 0.2)

    def test_future_capacity_guard_excludes_the_late_slot(self) -> None:
        env = make_env(12, {2: 0.2}, 3)
        env.reset(seed=0)
        for _ in range(2):
            env.step(HOLD_BATCH)
        guard = env._guaranteed_future_batch_capacity_by_deadline()
        # Arrival in slot 2 with H = 3: usable future slots are 3 and 4 only.
        self.assertEqual(list(guard), [5])
        self.assertAlmostEqual(guard[5], 2.0 * env._future_batch_capacity_fraction * 1.0)

    def test_edf_drain_order_is_deadline_then_origin_then_arrival(self) -> None:
        from env.ramp_v6.models import EDFQueue

        queue = EDFQueue()
        queue.add(0.1, origin=1, deadline_step=4)   # sequence 0
        queue.add(0.1, origin=0, deadline_step=4)   # sequence 1: same deadline, lower origin first
        queue.add(0.1, origin=1, deadline_step=3)   # sequence 2: earliest deadline first
        queue.add(0.1, origin=0, deadline_step=5)   # sequence 3
        drained = queue.drain(0.25, n_origins=2)
        # 0.1 (origin 1, d=3) + 0.1 (origin 0, d=4) + 0.05 (origin 1, d=4)
        np.testing.assert_allclose(drained, [0.1, 0.15])
        self.assertEqual([entry.sequence for entry in queue.entries], [0, 3])


class StatusQuoTests(unittest.TestCase):
    def test_status_quo_runs_every_arrival_in_place_in_its_hour(self) -> None:
        hours = 24
        rng = np.random.default_rng(3)
        batch = {hour: float(rng.uniform(0.05, 0.3)) for hour in range(hours)}
        env = make_env(hours, batch, 3)
        infos = run(env, lambda e: e.evaluation_action("status_quo"), hours)
        for hour, info in enumerate(infos):
            self.assertAlmostEqual(info["service_allocation"][0], 0.4)
            self.assertAlmostEqual(info["batch_by_destination"][0], batch[hour])
            self.assertAlmostEqual(info["batch_queue"]["queued"], 0.0)


if __name__ == "__main__":
    unittest.main()
