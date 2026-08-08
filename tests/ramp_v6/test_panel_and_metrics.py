"""Train-only split/statistics and evaluation metric tests."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from env.ramp_v6.evaluation import (
    CausalPolicyContext,
    compare_to_status_quo,
    flat_preference_policy,
    no_proxy_summary,
    run_episode,
    status_quo_policy,
    summarize,
)
from env.ramp_v6.panel import CanonicalMarketPanel
from tests.ramp_v6.common import make_env


def monthly_panel() -> CanonicalMarketPanel:
    timestamps = pd.date_range(
        "2025-01-01", "2025-03-31 23:00:00", freq="h", tz="UTC"
    )
    hours = np.arange(len(timestamps), dtype=np.float64)
    gross = 1000.0 + 100.0 * np.sin(2.0 * np.pi * hours / 24.0)
    net = gross - 150.0
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "market_id": "M",
            "gross_demand_mw": gross,
            "net_load_mw": net,
            "wind_mw": 100.0,
            "solar_mw": 50.0,
            "market_scale_mw": 1200.0,
            "da_lmp_usd_per_mwh": 30.0,
            "forecast_issue_time_utc": timestamps,
            "forecast_vintage_id": [f"v{index}" for index in range(len(hours))],
            "quality_ok": True,
        }
    )
    for quantity, values in (("gross", gross), ("net", net)):
        for horizon in (1, 2, 3):
            frame[f"forecast_{quantity}_h{horizon}_mw"] = np.roll(
                values, -horizon
            )
    return CanonicalMarketPanel(frame)


class PanelAndMetricsTests(unittest.TestCase):
    def test_complete_month_split_and_train_only_fit(self) -> None:
        panel = monthly_panel()
        train, validation, test = panel.split_complete_months(
            ["2025-01"], ["2025-02"], ["2025-03"]
        )
        self.assertEqual(len(train), 31 * 24)
        self.assertEqual(len(validation), 28 * 24)
        self.assertEqual(len(test), 31 * 24)
        stats = panel.fit_stats(["2025-01"])
        mutated = panel.frame.copy()
        mutated.loc[
            mutated["timestamp_utc"].dt.strftime("%Y-%m") == "2025-03",
            "gross_demand_mw",
        ] += 1_000_000.0
        mutated_stats = CanonicalMarketPanel(mutated).fit_stats(["2025-01"])
        self.assertEqual(stats.gross_q95_mw, mutated_stats.gross_q95_mw)
        self.assertEqual(
            stats.native_abs_ramp_q90_fraction_s_per_hour,
            mutated_stats.native_abs_ramp_q90_fraction_s_per_hour,
        )

    def test_partial_month_is_rejected(self) -> None:
        panel = monthly_panel()
        partial = CanonicalMarketPanel(
            panel.frame[panel.frame["timestamp_utc"] < "2025-03-31"].copy()
        )
        with self.assertRaisesRegex(ValueError, "complete chronological month"):
            partial.split_complete_months(
                ["2025-01"], ["2025-02"], ["2025-03"]
            )

    def test_absent_month_is_rejected(self) -> None:
        panel = monthly_panel()
        with self.assertRaisesRegex(ValueError, "complete chronological month"):
            panel.split_complete_months(
                ["2025-01"], ["2025-02"], ["2025-04"]
            )

    def test_metrics_include_comparators_cost_budget_and_strata(self) -> None:
        candidate_env = make_env()
        candidate_reward, candidate_history = run_episode(
            candidate_env, flat_preference_policy
        )
        candidate = summarize(candidate_history, total_reward=candidate_reward)
        status_env = make_env()
        status_reward, status_history = run_episode(
            status_env, status_quo_policy
        )
        status = summarize(status_history, total_reward=status_reward)
        comparison = compare_to_status_quo(
            candidate, status, candidate_env.protocol.cost_budget_fraction
        )
        no_proxy = no_proxy_summary(candidate)
        self.assertIn("A", candidate["per_market"])
        self.assertIn("1h", candidate["per_market"]["A"])
        self.assertIn("forecast_error_strata", candidate)
        self.assertIn("cost_budget_compliant", comparison)
        self.assertEqual(no_proxy["macro_incremental_ramp_impact_mean"], 0.0)
        self.assertEqual(
            candidate["workload_safety"]["terminal_batch_queue"], 0.0
        )

    def test_negative_price_cost_budget_uses_absolute_baseline(self) -> None:
        candidate = {
            "da_energy_cost_usd": -100.0,
            "macro_incremental_ramp_impact_mean": 0.0,
        }
        comparison = compare_to_status_quo(candidate, candidate, 0.05)
        self.assertTrue(comparison["cost_budget_compliant"])
        self.assertEqual(comparison["cost_budget_limit_usd"], -95.0)

    def test_policy_callback_receives_only_causal_context(self) -> None:
        seen: list[CausalPolicyContext] = []

        def inspect_policy(observation, context):
            del observation
            seen.append(context)
            self.assertFalse(hasattr(context, "panel"))
            return np.zeros(2 * context.n_sites + 1)

        env = make_env()
        run_episode(env, inspect_policy)
        self.assertEqual(len(seen), env.action_steps)

    def test_invalid_quality_and_vintage_are_rejected(self) -> None:
        panel = monthly_panel()
        bad_quality = panel.frame.copy()
        bad_quality["quality_ok"] = "False"
        with self.assertRaisesRegex(ValueError, "quality_ok"):
            CanonicalMarketPanel(bad_quality)
        bad_vintage = panel.frame.copy()
        bad_vintage.loc[0, "forecast_vintage_id"] = None
        with self.assertRaisesRegex(ValueError, "forecast_vintage_id"):
            CanonicalMarketPanel(bad_vintage)


if __name__ == "__main__":
    unittest.main()
