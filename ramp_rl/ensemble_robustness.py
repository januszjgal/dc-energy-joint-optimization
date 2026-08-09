"""Predeclared post-selection robustness factories for pure-RL ensemble v4."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from env.ramp_v6.environment import RampAwareEnv
from env.ramp_v6.factory import (
    ENERGY_MODEL_V3_HANDOFF,
    EnergyModelV3WindowEnv,
    _build_energy_model_v3_window,
)
from env.ramp_v6.models import RampProtocol, WorkloadTrace
from ramp_rl.contract import EnvRequest


ROOT = Path(__file__).resolve().parent.parent
PRIMARY_TOTAL_RATED_POWER_MW = 600.0
ROBUSTNESS_TOTAL_RATED_POWER_MW = 1000.0
SCALE_MULTIPLIER = (
    ROBUSTNESS_TOTAL_RATED_POWER_MW / PRIMARY_TOTAL_RATED_POWER_MW
)
MARKETS = (
    "CAISO_NP15",
    "ERCOT_LZ_NORTH",
    "NYISO_NYC_J",
    "MISO_MINN_HUB",
    "SPP_NORTH_HUB",
    "ISONE_NEMA",
)
OVERLAPPING_CELLS = ("c", "d", "e", "f", "g", "h")
TRACE_ALIGNMENT_EPOCH = pd.Timestamp("2025-09-01T00:00:00Z")
TRACE_ALIGNMENT_END = pd.Timestamp("2026-05-01T00:00:00Z")
OVERLAPPING_INPUT_SHA256 = {
    "cell_c_tiers": "b85ce22da8a70f34cf741ea985cbd8d8e93ce80c29be1d67318e95524c2fd652",
    "cell_d_tiers": "4edc94c9c57aefe5753751a05647c3823bec69ed395219f1066afc53a354a795",
    "cell_e_tiers": "86381fbc93aa9aed5c2e3fbc5a3eb915c10238a39a4119bc619eae4d6ae4a85e",
    "cell_f_tiers": "f3cfdd137188300b54a1aa23ae7545ec84194821cee6244c3ff816800076a514",
    "cell_g_tiers": "90d4778cae8810fd4fa5c41f3eb9e18fb3fd8069d8ccb381e58585a062722a7a",
    "cell_h_tiers": "dafb1d5007eddcfb27183c343ee0586c64ac6e2128516e44c8e8b291c859a759",
    "batch_distributions_c": "b1027f86519bc1ab5755e0a9ce56079c648496ea1ee2923821366553fa745358",
    "batch_distributions_d": "6ed6389c7da970ff60b30537c44ff04e565e9c904811813ce8a2d774e879f008",
    "batch_distributions_e": "3afd71cdbdb7c44ec2f10f04e46704884201a0d15dcb3acff41a08faf7312e4c",
    "batch_distributions_f": "4dd0a0403694d89ac0794d84a802f5e7fecaef8a849b9422d9fec6616f417715",
    "batch_distributions_g": "e32ddfabc4bb4ac01f526be3306968910a5c3b1716ee08a806729a332687dad2",
    "batch_distributions_h": "b6efc9a58fac9bc1d366ff76a6b8bb8a6314f3cee827d0385f9f5d840541b5e6",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _overlapping_workload() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int],
    dict[str, str],
]:
    timestamps = pd.date_range(
        TRACE_ALIGNMENT_EPOCH,
        TRACE_ALIGNMENT_END,
        freq="h",
        inclusive="left",
    )
    service: dict[str, np.ndarray] = {}
    batch: dict[str, np.ndarray] = {}
    deadlines: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for market, cell in zip(MARKETS, OVERLAPPING_CELLS):
        tier_path = ROOT / "data" / "cells" / f"cell_{cell}_tiers.csv"
        if _sha256(tier_path) != OVERLAPPING_INPUT_SHA256[f"cell_{cell}_tiers"]:
            raise ValueError(f"cell {cell} tier input hash mismatch")
        tier = pd.read_csv(tier_path)
        if len(tier) % 12 == 1:
            tier = tier.iloc[:-1]
        if len(tier) % 12:
            raise ValueError(f"tier trace cannot be aggregated to hours: cell {cell}")
        hourly = tier.groupby(np.arange(len(tier)) // 12)[
            ["service_demand_norm", "batch_demand_norm"]
        ].mean()
        repetitions = math.ceil(len(timestamps) / len(hourly))
        service[market] = np.tile(
            hourly["service_demand_norm"].to_numpy(), repetitions
        )[: len(timestamps)]
        batch[market] = np.tile(
            hourly["batch_demand_norm"].to_numpy(), repetitions
        )[: len(timestamps)]
        distribution_path = (
            ROOT / "data" / "jobs" / f"batch_distributions_{cell}.json"
        )
        if (
            _sha256(distribution_path)
            != OVERLAPPING_INPUT_SHA256[f"batch_distributions_{cell}"]
        ):
            raise ValueError(f"cell {cell} deadline input hash mismatch")
        distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
        deadlines[market] = max(
            1,
            math.ceil(
                2.0 * float(distribution["duration"]["mean"]) / 3600.0
            ),
        )
        hashes[f"tier_workload:cell_{cell}"] = _sha256(tier_path)
        hashes[f"deadline_model:cell_{cell}"] = _sha256(distribution_path)
    service_frame = pd.DataFrame(service, index=timestamps)
    batch_frame = pd.DataFrame(batch, index=timestamps)
    batch_limit = (
        RampProtocol().batch_arrival_envelope_fraction_of_fleet * len(MARKETS)
    )
    totals = batch_frame.sum(axis=1)
    scale = np.minimum(1.0, batch_limit / totals.where(totals > 0.0, 1.0))
    admitted_batch = batch_frame.mul(scale, axis=0)
    service_frame += batch_frame - admitted_batch
    if float(service_frame.sum(axis=1).max()) > 0.75 * len(MARKETS) + 1e-12:
        raise ValueError("overlapping service workload exceeds the fleet envelope")
    if float(admitted_batch.sum(axis=1).max()) > batch_limit + 1e-12:
        raise ValueError("overlapping batch workload exceeds the fleet envelope")
    return service_frame, admitted_batch, deadlines, hashes


def _scaled_environment(base: RampAwareEnv, variant: str) -> RampAwareEnv:
    context = dict(base._episode_context)
    context["robustness_variant"] = variant
    context["primary_total_rated_power_mw"] = PRIMARY_TOTAL_RATED_POWER_MW
    context["total_rated_power_mw"] = ROBUSTNESS_TOTAL_RATED_POWER_MW
    context["scale_multiplier"] = SCALE_MULTIPLIER
    sites = [site.scaled(SCALE_MULTIPLIER) for site in base.sites]
    if variant == "one-gw-total":
        workload = base.workload.scaled(SCALE_MULTIPLIER)
        context["workload_mapping"] = ["a", "b", "c", "d", "e", "f"]
    elif variant == "c-h-overlapping":
        service, batch, deadlines, hashes = _overlapping_workload()
        day = pd.Timestamp(str(context["day"]), tz="UTC")
        active_times = pd.date_range(day, periods=24, freq="h", tz="UTC")
        tail = np.zeros((3, len(MARKETS)), dtype=np.float64)
        warm_times = pd.date_range(
            day - pd.Timedelta(hours=3), periods=3, freq="h", tz="UTC"
        )
        warm_work = (
            service.loc[warm_times, list(MARKETS)].to_numpy()
            + batch.loc[warm_times, list(MARKETS)].to_numpy()
        )
        warm_power = np.asarray(
            [
                [
                    site.power_mw(float(warm_work[row, column]))
                    for column, site in enumerate(base.sites)
                ]
                for row in range(len(warm_times))
            ],
            dtype=np.float64,
        )
        workload = WorkloadTrace(
            service_arrivals=np.vstack(
                [service.loc[active_times, list(MARKETS)].to_numpy(), tail]
            )
            * SCALE_MULTIPLIER,
            batch_arrivals=np.vstack(
                [batch.loc[active_times, list(MARKETS)].to_numpy(), tail]
            )
            * SCALE_MULTIPLIER,
            batch_deadline_hours=np.tile(
                np.asarray([deadlines[market] for market in MARKETS], dtype=np.int64),
                (27, 1),
            ),
            warm_power_mw=warm_power * SCALE_MULTIPLIER,
        )
        context["workload_mapping"] = list(OVERLAPPING_CELLS)
        context["source_hashes"] = {
            **dict(context["source_hashes"]),
            **hashes,
        }
    else:
        raise ValueError(f"unknown robustness variant: {variant}")
    return RampAwareEnv(
        base.panel,
        sites,
        workload,
        base.stats,
        base.protocol,
        episode_context=context,
        epsilon_pct=base._epsilon_pct,
        lagrangian_multiplier=base._lagrangian_multiplier,
    )


class RobustnessEnergyModelV3WindowEnv(EnergyModelV3WindowEnv):
    def __init__(
        self, payload: dict[str, Any], request: EnvRequest, variant: str
    ) -> None:
        self.variant = variant
        super().__init__(payload, request)

    def _new_window(self, window_id: str) -> RampAwareEnv:
        base = _build_energy_model_v3_window(self.payload, self.request, window_id)
        return _scaled_environment(base, self.variant)


def _make(request: EnvRequest, variant: str) -> RobustnessEnergyModelV3WindowEnv:
    if request.split != "test" or request.training:
        raise ValueError("v4 robustness factories are post-selection test-only")
    payload = json.loads(ENERGY_MODEL_V3_HANDOFF.read_text(encoding="utf-8"))
    return RobustnessEnergyModelV3WindowEnv(payload, request, variant)


def make_one_gw_total_env(
    request: EnvRequest,
) -> RobustnessEnergyModelV3WindowEnv:
    return _make(request, "one-gw-total")


def make_c_h_overlapping_env(
    request: EnvRequest,
) -> RobustnessEnergyModelV3WindowEnv:
    return _make(request, "c-h-overlapping")
