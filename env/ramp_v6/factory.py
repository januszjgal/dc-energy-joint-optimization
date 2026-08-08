"""Factories joining the ramp-core environment to the pure-RL harness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from env.ramp_v6.environment import RampAwareEnv
from env.ramp_v6.fixture import load_fixture, load_six_market_fixture
from ramp_rl.contract import EnvRequest
from ramp_rl.schema import load_protocol

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "ramp_v6"
ENERGY_MODEL_V3_PANEL_ROOT = ROOT / "output" / "energy_model_v3" / "ramp_v6"
ENERGY_MODEL_V3_HANDOFF = ENERGY_MODEL_V3_PANEL_ROOT / "factory_manifest.json"
SPLIT_MONTH = {"train": 1, "validation": 7, "test": 9}


class MissingEnergyModelV3PanelError(FileNotFoundError):
    """Raised when the sole remaining long-campaign input is unavailable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_context(request: EnvRequest) -> dict[str, Any]:
    month = SPLIT_MONTH[request.split]
    window_id = request.window_id or (
        f"ramp-core-m-{month:02d}-rank-{request.rank}-seed-{request.seed}"
    )
    transform = {
        "id": "ramp-core-six-market-expansion-v1",
        "source_market": "A",
        "markets": [f"M{index}" for index in range(6)],
        "total_rated_power_mw": 1000.0,
    }
    return {
        "split": request.split,
        "window_id": window_id,
        "month": month,
        "day": window_id,
        "chronological": True,
        "forecast_model": "ramp-core-deterministic-causal-forecast-v1",
        "forecast_vintage": "fixture-six-market-2026-08-08",
        "source_hashes": {
            "canonical_panel": _sha256(FIXTURE_ROOT / "canonical_panel.csv"),
            "fixture": _sha256(FIXTURE_ROOT / "fixture.json"),
            "six_market_transform": hashlib.sha256(
                json.dumps(transform, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        "future_realized_features_exposed": False,
    }


def make_fixture_env(request: EnvRequest) -> RampAwareEnv:
    """Create the actual ramp-core six-market environment for smoke evidence."""
    if request.split not in SPLIT_MONTH:
        raise ValueError("split must be train, validation, or test")
    panel, sites, workload, stats, protocol = load_six_market_fixture(FIXTURE_ROOT)
    return RampAwareEnv(
        panel,
        sites,
        workload,
        stats,
        protocol,
        episode_context=_fixture_context(request),
        epsilon_pct=request.epsilon_pct,
        lagrangian_multiplier=request.lagrangian_multiplier,
    )


def _build_energy_model_v3_window(
    payload: dict[str, Any],
    request: EnvRequest,
    window_id: str,
) -> RampAwareEnv:
    split_windows = payload["windows"][request.split]
    window = split_windows[window_id]
    artifact_root = Path(str(window["artifact_root"]))
    if not artifact_root.is_absolute():
        artifact_root = ROOT / artifact_root
    panel_path = artifact_root / "canonical_panel.csv"
    fixture_path = artifact_root / "fixture.json"
    required = {"canonical_panel": panel_path, "fixture": fixture_path}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise MissingEnergyModelV3PanelError(
            "missing energy-model-v3 ramp panel artifacts: " + ", ".join(missing)
        )
    declared_hashes = window.get("source_hashes")
    if not isinstance(declared_hashes, dict):
        raise ValueError("energy-model-v3 window requires source_hashes")
    actual_hashes = {
        name: _sha256(path) for name, path in required.items()
    }
    for name, actual in actual_hashes.items():
        if declared_hashes.get(name) != actual:
            raise ValueError(
                f"energy-model-v3 {name} SHA-256 does not match the handoff"
            )
    panel, sites, workload, stats, protocol = load_fixture(artifact_root)
    if len(panel.markets) != 6 or len(sites) != 6:
        raise ValueError("energy-model-v3 primary handoff must contain six markets/sites")
    campaign = load_protocol()
    allowed_months = {
        int(value)
        for value in campaign["data"]["split"][request.split]["months"]
    }
    declared_month = int(window["month"])
    if declared_month not in allowed_months:
        raise ValueError(
            f"energy-model-v3 window month {declared_month} is outside "
            f"the frozen {request.split} split"
        )
    panel_months = set(panel.timestamps.month.astype(int))
    if panel_months != {declared_month}:
        raise ValueError(
            "energy-model-v3 panel timestamps do not match the declared split month"
        )
    training_months = {
        int(value) for value in campaign["data"]["split"]["train"]["months"]
    }
    fit_months = {int(str(value)[-2:]) for value in stats.fit_months}
    if not fit_months or not fit_months.issubset(training_months):
        raise ValueError(
            "frozen ramp statistics were not fit exclusively on training months"
        )
    stats_sha256 = hashlib.sha256(
        json.dumps(asdict(stats), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if stats_sha256 != payload["frozen_stats_sha256"]:
        raise ValueError(
            "energy-model-v3 window does not use the manifest's frozen statistics"
        )
    context = {
        "split": request.split,
        "window_id": window_id,
        "month": declared_month,
        "day": str(window.get("day", window_id)),
        "chronological": True,
        "forecast_model": str(payload["forecast_model"]),
        "forecast_vintage": str(window["forecast_vintage"]),
        "source_hashes": dict(declared_hashes),
        "future_realized_features_exposed": False,
    }
    return RampAwareEnv(
        panel,
        sites,
        workload,
        stats,
        protocol,
        episode_context=context,
        epsilon_pct=request.epsilon_pct,
        lagrangian_multiplier=request.lagrangian_multiplier,
    )


class EnergyModelV3WindowEnv(gym.Env):
    """Resample train windows while keeping validation and test deterministic."""

    metadata = {"render_modes": []}

    def __init__(self, payload: dict[str, Any], request: EnvRequest):
        super().__init__()
        self.payload = payload
        self.request = request
        self._multiplier = float(request.lagrangian_multiplier)
        split_windows = payload["windows"][request.split]
        self._window_ids = tuple(sorted(split_windows))
        if request.window_id is not None and request.window_id not in split_windows:
            raise ValueError(
                f"energy-model-v3 handoff does not contain window "
                f"{request.window_id!r} in {request.split}"
            )
        self._rng = np.random.default_rng(request.seed + 1009 * request.rank)
        self._order: list[str] = []
        self._current = self._new_window(self._initial_window_id())
        self.action_space = self._current.action_space
        self.observation_space = self._current.observation_space

    def _initial_window_id(self) -> str:
        if self.request.window_id is not None:
            return self.request.window_id
        return self._window_ids[(self.request.seed + self.request.rank) % len(self._window_ids)]

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
        current = _build_energy_model_v3_window(
            self.payload, self.request, window_id
        )
        current.set_lagrangian_multiplier(self._multiplier)
        return current

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
            raise ValueError("energy-model-v3 windows have inconsistent spaces")
        requested["split"] = self.request.split
        requested["window_id"] = window_id
        return self._current.reset(seed=seed, options=requested)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self._current.step(action)

    def close(self) -> None:
        self._current.close()


def make_energy_model_v3_env(request: EnvRequest) -> EnergyModelV3WindowEnv:
    """Load a resampling real-panel environment from the frozen handoff."""
    if not ENERGY_MODEL_V3_HANDOFF.is_file():
        raise MissingEnergyModelV3PanelError(
            "missing energy-model-v3 ramp panel handoff: "
            f"{ENERGY_MODEL_V3_HANDOFF}"
        )
    payload = json.loads(ENERGY_MODEL_V3_HANDOFF.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "energy-model-v3-ramp-v6-factory-v1":
        raise ValueError("unsupported energy-model-v3 ramp factory manifest")
    if not str(payload.get("forecast_model", "")).strip():
        raise ValueError("energy-model-v3 handoff requires forecast_model")
    frozen_stats_sha256 = payload.get("frozen_stats_sha256")
    if not (
        isinstance(frozen_stats_sha256, str)
        and len(frozen_stats_sha256) == 64
        and all(character in "0123456789abcdef" for character in frozen_stats_sha256)
    ):
        raise ValueError(
            "energy-model-v3 handoff requires frozen_stats_sha256"
        )
    windows = payload.get("windows")
    if not isinstance(windows, dict) or any(
        not isinstance(windows.get(split), dict) or not windows[split]
        for split in ("train", "validation", "test")
    ):
        raise ValueError(
            "energy-model-v3 handoff requires non-empty train, validation, "
            "and test window maps"
        )
    split_windows = windows.get(request.split)
    if not isinstance(split_windows, dict) or not split_windows:
        raise ValueError(
            f"energy-model-v3 handoff has no {request.split} windows"
        )
    return EnergyModelV3WindowEnv(payload, request)
