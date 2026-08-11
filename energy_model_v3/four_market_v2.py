"""Self-contained four-market factory for the paired admission study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from env.ramp_v6.environment import RampAwareEnv
from env.ramp_v6.models import FrozenRampStats, RampProtocol, SiteConfig, WorkloadTrace
from env.ramp_v6.panel import CanonicalMarketPanel
from ramp_rl.contract import EnvRequest


ROOT = Path(__file__).resolve().parent.parent
FACTORY_ROOT = ROOT / "output" / "four_market_v2" / "factory"
MARKET_TO_CELL = (
    ("CAISO_NP15", "a"),
    ("MISO_MINN_HUB", "b"),
    ("SPP_NORTH_HUB", "c"),
    ("ISONE_NEMA", "d"),
)
MARKETS = tuple(market for market, _ in MARKET_TO_CELL)
CELLS = tuple(cell for _, cell in MARKET_TO_CELL)
RATED_POWER_MW = 500.0
COMPUTE_CAPACITY = 1.0
TOTAL_RATED_POWER_MW = 2_000.0
VARIANTS = ("envelope_on", "envelope_off")


def _canonical_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _protocol(variant: str) -> RampProtocol:
    return RampProtocol(
        protocol_id=f"four-market-v2-{variant}",
        cost_budget_fraction=0.02,
        admission_envelope_enabled=variant == "envelope_on",
    )


def _load_verified_window(
    artifact_root: Path, canonical_panel_sha256: str, fixture_sha256: str
) -> tuple[
    CanonicalMarketPanel,
    list[SiteConfig],
    WorkloadTrace,
    FrozenRampStats,
]:
    panel_path = artifact_root / "canonical_panel.csv"
    fixture_path = artifact_root / "fixture.json"
    if not panel_path.is_file() or not fixture_path.is_file():
        raise FileNotFoundError(f"missing v2 window artifact: {artifact_root}")
    if _canonical_hash(panel_path) != canonical_panel_sha256:
        raise ValueError("v2 canonical panel hash mismatch")
    if _canonical_hash(fixture_path) != fixture_sha256:
        raise ValueError("v2 fixture hash mismatch")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    workload_payload = payload["workload"]
    stats_payload = payload["frozen_stats"]
    stats_payload["native_abs_ramp_q90_fraction_s_per_hour"] = {
        market: {int(horizon): value for horizon, value in thresholds.items()}
        for market, thresholds in stats_payload[
            "native_abs_ramp_q90_fraction_s_per_hour"
        ].items()
    }
    stats_payload["fit_months"] = tuple(stats_payload["fit_months"])
    return (
        CanonicalMarketPanel.from_csv(panel_path),
        [SiteConfig(**site) for site in payload["sites"]],
        WorkloadTrace(
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
        ),
        FrozenRampStats(**stats_payload),
    )


class FourMarketV2WindowEnv(gym.Env):
    """Window wrapper whose only runtime input is the v2 artifact root."""

    metadata = {"render_modes": []}

    def __init__(
        self, payload: dict[str, Any], request: EnvRequest, variant: str
    ) -> None:
        super().__init__()
        self.variant = variant
        self.payload = payload
        self.request = request
        self._multiplier = float(request.lagrangian_multiplier)
        split_windows = payload["windows"][request.split]
        self._window_ids = tuple(sorted(split_windows))
        if request.window_id is not None and request.window_id not in split_windows:
            raise ValueError(
                f"v2 factory has no {request.window_id!r} in {request.split}"
            )
        self._rng = np.random.default_rng(request.seed + 1009 * request.rank)
        self._order: list[str] = []
        self._current = self._new_window(self._initial_window_id())
        self.action_space = self._current.action_space
        self.observation_space = self._current.observation_space

    def _initial_window_id(self) -> str:
        if self.request.window_id is not None:
            return self.request.window_id
        return self._window_ids[
            (self.request.seed + self.request.rank) % len(self._window_ids)
        ]

    def _next_window_id(self, options: dict[str, Any]) -> str:
        requested = options.get("window_id", self.request.window_id)
        if requested is not None:
            if requested not in self._window_ids:
                raise ValueError(
                    f"window {requested!r} is outside the {self.request.split} split"
                )
            return str(requested)
        if self.request.split != "train":
            return self._initial_window_id()
        if not self._order:
            self._order = [
                self._window_ids[index]
                for index in self._rng.permutation(len(self._window_ids))
            ]
        return self._order.pop()

    def _new_window(self, window_id: str) -> RampAwareEnv:
        record = self.payload["windows"][self.request.split][window_id]
        artifact_root = ROOT / record["artifact_root"]
        panel, sites, workload, stats = _load_verified_window(
            artifact_root.resolve(),
            record["source_hashes"]["canonical_panel"],
            record["source_hashes"]["fixture"],
        )
        if tuple(site.market_id for site in sites) != MARKETS:
            raise ValueError("v2 site mapping is invalid")
        if len(sites) != 4 or any(
            site.compute_capacity != COMPUTE_CAPACITY
            or site.rated_power_mw != RATED_POWER_MW
            for site in sites
        ):
            raise ValueError("v2 sites must be four direct 500 MW / K=1 sites")
        expected_stats = self.payload["frozen_stats_sha256"]
        actual_stats = hashlib.sha256(
            json.dumps(asdict(stats), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if actual_stats != expected_stats:
            raise ValueError("v2 frozen statistics mismatch")
        if tuple(workload.batch_deadline_hours[0]) != (2, 1, 2, 3):
            raise ValueError("v2 batch deadline semantics are invalid")
        active = panel.timestamps[3:-3]
        if set(active.strftime("%Y-%m-%d")) != {str(record["day"])}:
            raise ValueError("v2 window day is invalid")
        context = {
            "split": self.request.split,
            "window_id": window_id,
            "month": int(record["month"]),
            "day": str(record["day"]),
            "chronological": True,
            "forecast_model": self.payload["forecast_model"],
            "forecast_vintage": str(record["forecast_vintage"]),
            "source_hashes": dict(record["source_hashes"]),
            "future_realized_features_exposed": False,
            "study_role": "four_market_v2_validation_only",
            "variant": self.variant,
            "admission_envelope_enabled": self.variant == "envelope_on",
            "markets": list(MARKETS),
            "workload_cells": list(CELLS),
            "total_rated_power_mw": TOTAL_RATED_POWER_MW,
        }
        return RampAwareEnv(
            panel,
            sites,
            workload,
            stats,
            _protocol(self.variant),
            episode_context=context,
            epsilon_pct=self.request.epsilon_pct,
            lagrangian_multiplier=self._multiplier,
        )

    def ramp_rl_contract(self) -> dict[str, Any]:
        return self._current.ramp_rl_contract()

    def set_lagrangian_multiplier(self, value: float) -> None:
        self._multiplier = float(value)
        self._current.set_lagrangian_multiplier(value)

    def evaluation_action(self, name: str) -> np.ndarray:
        return self._current.evaluation_action(name)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed + 1009 * self.request.rank)
            self._order = []
        requested = dict(options or {})
        window_id = self._next_window_id(requested)
        self._current.close()
        self._current = self._new_window(window_id)
        if (
            self._current.action_space != self.action_space
            or self._current.observation_space != self.observation_space
        ):
            raise ValueError("v2 windows have inconsistent spaces")
        requested["split"] = self.request.split
        requested["window_id"] = window_id
        return self._current.reset(seed=seed, options=requested)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self._current.step(action)

    def close(self) -> None:
        self._current.close()


def make_four_market_v2_env(
    request: EnvRequest, *, variant: str = "envelope_on"
) -> FourMarketV2WindowEnv:
    """Build a validation-only v2 factory variant without legacy dependencies."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown v2 variant: {variant}")
    if request.split not in {"train", "validation"}:
        raise ValueError("four-market-v2 permits train and validation only")
    manifest_path = FACTORY_ROOT / variant / "factory_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload["variant"]["id"] != variant:
        raise ValueError("variant manifest identity mismatch")
    if payload["split_periods"]["test"]:
        raise ValueError("v2 factory must not expose a test split")
    return FourMarketV2WindowEnv(payload, request, variant)


def make_envelope_on_env(request: EnvRequest) -> FourMarketV2WindowEnv:
    return make_four_market_v2_env(request, variant="envelope_on")


def make_envelope_off_env(request: EnvRequest) -> FourMarketV2WindowEnv:
    return make_four_market_v2_env(request, variant="envelope_off")


def factory_identity_paths(factory: Any) -> tuple[Path, ...]:
    """Return manifests that must remain fixed when resuming a training job."""
    variants = {
        "make_envelope_on_env": "envelope_on",
        "make_envelope_off_env": "envelope_off",
    }
    try:
        variant = variants[factory.__name__]
    except (AttributeError, KeyError) as error:
        raise ValueError("unknown four-market v2 factory identity") from error
    return (
        ROOT / "data" / "four_market_v2" / "source_manifest.json",
        FACTORY_ROOT / variant / "factory_manifest.json",
    )
