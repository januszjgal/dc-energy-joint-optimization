"""Load deterministic v6 fixtures without coupling to downloader artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
