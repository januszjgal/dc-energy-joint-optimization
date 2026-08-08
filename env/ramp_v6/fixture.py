"""Load deterministic v6 fixtures without coupling to downloader artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from env.ramp_v6.models import FrozenRampStats, RampProtocol, SiteConfig, WorkloadTrace
from env.ramp_v6.panel import CanonicalMarketPanel
from env.ramp_v6.protocol import load_ramp_protocol


def load_fixture(
    fixture_root: Path,
    *,
    scale_multiplier: float = 1.0,
) -> tuple[
    CanonicalMarketPanel,
    list[SiteConfig],
    WorkloadTrace,
    FrozenRampStats,
    RampProtocol,
]:
    panel = CanonicalMarketPanel.from_csv(fixture_root / "canonical_panel.csv")
    payload = json.loads((fixture_root / "fixture.json").read_text(encoding="utf-8"))
    sites = [SiteConfig(**item).scaled(scale_multiplier) for item in payload["sites"]]
    workload_payload = payload["workload"]
    workload = WorkloadTrace(
        service_arrivals=np.asarray(
            workload_payload["service_arrivals"], dtype=np.float64
        ),
        batch_arrivals=np.asarray(
            workload_payload["batch_arrivals"], dtype=np.float64
        ),
        batch_deadline_hours=np.asarray(
            workload_payload["batch_deadline_hours"], dtype=np.int64
        ),
        warm_power_mw=np.asarray(
            workload_payload["warm_power_mw"], dtype=np.float64
        ),
    ).scaled(scale_multiplier)
    stats_payload = payload["frozen_stats"]
    stats_payload["native_abs_ramp_q90_fraction_s_per_hour"] = {
        market: {int(horizon): value for horizon, value in thresholds.items()}
        for market, thresholds in stats_payload[
            "native_abs_ramp_q90_fraction_s_per_hour"
        ].items()
    }
    stats_payload["fit_months"] = tuple(stats_payload["fit_months"])
    stats = FrozenRampStats(**stats_payload)
    protocol = load_ramp_protocol()
    return panel, sites, workload, stats, protocol


def load_six_market_fixture(
    fixture_root: Path,
    *,
    total_rated_power_mw: float = 1000.0,
) -> tuple[
    CanonicalMarketPanel,
    list[SiteConfig],
    WorkloadTrace,
    FrozenRampStats,
    RampProtocol,
]:
    """Expand the deterministic ramp-core fixture to the frozen six-market shape."""
    panel, _, workload, stats, protocol = load_fixture(fixture_root)
    source = panel.frame[panel.frame["market_id"] == "A"]
    markets = tuple(f"M{index}" for index in range(6))
    frames = []
    for market in markets:
        rows = source.copy()
        rows["market_id"] = market
        rows["forecast_vintage_id"] = (
            market + "-" + rows["forecast_vintage_id"].astype(str)
        )
        frames.append(rows)
    six_panel = CanonicalMarketPanel(pd.concat(frames, ignore_index=True))
    scale = float(total_rated_power_mw) / (6.0 * 100.0)
    sites = [
        SiteConfig(site_id=f"site-{index}", market_id=market).scaled(scale)
        for index, market in enumerate(markets)
    ]
    six_workload = WorkloadTrace(
        service_arrivals=np.repeat(workload.service_arrivals[:, :1], 6, axis=1),
        batch_arrivals=np.repeat(workload.batch_arrivals[:, :1], 6, axis=1),
        batch_deadline_hours=np.repeat(
            workload.batch_deadline_hours[:, :1], 6, axis=1
        ),
        warm_power_mw=np.repeat(workload.warm_power_mw[:, :1], 6, axis=1),
    ).scaled(scale)
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
            market: dict(
                stats.native_abs_ramp_q90_fraction_s_per_hour["A"]
            )
            for market in markets
        },
        fit_months=stats.fit_months,
        stats_id="fixture-six-market-train-only-stats-v1",
    )
    return six_panel, sites, six_workload, six_stats, protocol
