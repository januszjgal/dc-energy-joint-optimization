"""Build and preflight the exact six-market hourly panel."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .calendar import StudyCalendar, make_calendar
from .contract import (
    PHYSICAL_COLUMNS,
    PRICE_COLUMNS,
    ContractError,
    validate_hourly_index,
    validate_native_table,
)
from .derivations import (
    add_grid_features,
    calibrate_scale,
    diagnostic_summary,
)

MARKETS = (
    "PJM_DOM",
    "NYISO_NYC_J",
    "CAISO_NP15",
    "ERCOT_LZ_NORTH",
    "MISO_MINN_HUB",
    "SPP_NORTH_HUB",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scenario(path: Path) -> dict[str, Any]:
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    if scenario.get("schema_version") != "energy-model-v3":
        raise ContractError(f"{path} is not an energy-model-v3 scenario")
    sites = scenario.get("sites", [])
    if len(sites) != 6:
        raise ContractError("v3 scenarios require exactly six sites")
    markets = [site["market"] for site in sites]
    if tuple(markets) != MARKETS:
        raise ContractError(
            f"scenario market order must be {MARKETS}, got {tuple(markets)}"
        )
    cells = [site["borg_cell"] for site in sites]
    expected_cells = (
        list("cdefgh")
        if scenario.get("mapping") == "robustness_overlapping_non_oof"
        else list("abcdef")
    )
    if cells != expected_cells:
        raise ContractError(
            f"scenario cells must be {expected_cells}, got {cells}"
        )
    powers = np.array([float(site["rated_power_mw"]) for site in sites])
    total = float(scenario["total_dc_power_mw"])
    if not np.isclose(powers.sum(), total, rtol=0, atol=0.1):
        raise ContractError("site powers do not sum to total_dc_power_mw")
    if scenario.get("penetration_override", False):
        if scenario.get("study_role") != "non_primary_price_taking_stress":
            raise ContractError("penetration override is stress-only")
    elif float(scenario["max_penetration_fraction"]) > 0.05:
        raise ContractError("non-stress penetration gate cannot exceed 5%")
    return scenario


def _market_input(root: Path, market: str, table: str) -> Path:
    return root / "native" / market / f"{table}.csv"


def _load_market(
    root: Path,
    market: str,
    calendar: StudyCalendar,
) -> pd.DataFrame:
    start = pd.Timestamp(calendar.start_utc)
    end = pd.Timestamp(calendar.end_utc)
    physical_path = _market_input(root, market, "physical_hourly")
    price_path = _market_input(root, market, "price_hourly")
    if not physical_path.exists() or not price_path.exists():
        missing = [
            str(path)
            for path in (physical_path, price_path)
            if not path.exists()
        ]
        raise ContractError(
            f"{market} missing primary native inputs: {', '.join(missing)}"
        )
    physical = validate_native_table(
        pd.read_csv(physical_path),
        PHYSICAL_COLUMNS - {"market_scale_mw"},
        table_name=f"{market}.physical",
        expected_market=market,
    )
    price = validate_native_table(
        pd.read_csv(price_path),
        PRICE_COLUMNS,
        table_name=f"{market}.price",
        expected_market=market,
        non_nullable={
            "interval_start_utc",
            "interval_end_utc",
            "market",
            "price_market",
            "price_location",
            "da_lmp_usd_mwh",
            "source_quality_flags",
        },
    )
    validate_hourly_index(
        physical, start=start, end=end, table_name=f"{market}.physical"
    )
    validate_hourly_index(
        price, start=start, end=end, table_name=f"{market}.price"
    )
    if not physical["interval_start_utc"].equals(
        price["interval_start_utc"]
    ):
        raise ContractError(f"{market} physical and price indices differ")
    duplicate = set(physical.columns) & set(price.columns)
    duplicate -= {
        "interval_start_utc",
        "interval_end_utc",
        "market",
        "source_quality_flags",
    }
    if duplicate:
        raise ContractError(
            f"{market} input tables overlap unexpectedly: {sorted(duplicate)}"
        )
    merged = physical.merge(
        price,
        on=["interval_start_utc", "interval_end_utc", "market"],
        how="inner",
        validate="one_to_one",
        suffixes=("_physical", "_price"),
    )
    return merged


def build_panel(
    *,
    data_root: Path,
    output_root: Path,
    scenario_path: Path,
    dc_power: dict[str, pd.Series],
    calendar: StudyCalendar | None = None,
) -> dict[str, Any]:
    calendar = calendar or make_calendar()
    scenario = load_scenario(scenario_path)
    output_root.mkdir(parents=True, exist_ok=True)
    scales: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    outputs: dict[str, str] = {}
    expected_index: pd.DatetimeIndex | None = None
    for site in scenario["sites"]:
        market = site["market"]
        frame = _load_market(data_root, market, calendar)
        index = pd.DatetimeIndex(frame["interval_start_utc"])
        if expected_index is None:
            expected_index = index
        elif not index.equals(expected_index):
            raise ContractError(f"{market} differs from exact common index")
        if market not in dc_power:
            raise ContractError(f"missing modeled DC power for {market}")
        scale = calibrate_scale(
            frame,
            market=market,
            train_start=pd.Timestamp(calendar.train_start_utc),
            train_end=pd.Timestamp(calendar.train_end_utc),
        )
        derived = add_grid_features(frame, dc_power[market], scale)
        destination = output_root / f"{market}.csv"
        derived.to_csv(destination, index=False)
        outputs[str(destination)] = sha256(destination)
        scales[market] = scale.to_dict()
        diagnostics[market] = diagnostic_summary(derived)
    manifest = {
        "schema_version": "energy-model-v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_mode": False,
        "calendar": calendar.to_dict(),
        "scenario": str(scenario_path),
        "scenario_sha256": sha256(scenario_path),
        "market_order": list(MARKETS),
        "exact_common_index": True,
        "training_only_calibration": True,
        "outputs": outputs,
        "scales": scales,
        "diagnostics": diagnostics,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def fixture_market_frame(
    market: str,
    *,
    calendar: StudyCalendar | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Create explicitly synthetic test-only input; never used by live builds."""
    calendar = calendar or make_calendar()
    index = pd.date_range(
        calendar.start_utc,
        calendar.end_utc,
        freq="1h",
        inclusive="left",
    )
    rng = np.random.default_rng(seed)
    hour = np.arange(len(index))
    gross = (
        20_000.0
        + 3_000.0 * np.sin(2 * np.pi * hour / 24)
        + rng.normal(0, 50, len(index))
    )
    wind = 2_000.0 + 400.0 * np.sin(2 * np.pi * hour / 72)
    solar = np.maximum(0.0, 4_000.0 * np.sin(2 * np.pi * (hour % 24 - 6) / 24))
    return pd.DataFrame(
        {
            "interval_start_utc": index,
            "interval_end_utc": index + pd.Timedelta(hours=1),
            "market": market,
            "gross_demand_mw": gross,
            "wind_mw": wind,
            "solar_mw": solar,
            "net_load_mw": gross - wind - solar,
            "net_load_method": "synthetic_fixture_only:gross-wind-solar",
            "market_scale_mw": np.nan,
            "source_quality_flags": "SYNTHETIC_FIXTURE_ONLY",
            "price_market": "fixture",
            "price_location": "fixture",
            "da_lmp_usd_mwh": 35 + 10 * np.sin(2 * np.pi * hour / 24),
            "rt_lmp_usd_mwh": np.nan,
            "energy_component_usd_mwh": np.nan,
            "congestion_component_usd_mwh": np.nan,
            "loss_component_usd_mwh": np.nan,
        }
    )


def blocked_preflight(
    data_root: Path,
    capabilities: list[dict[str, Any]],
    calendar: StudyCalendar | None = None,
) -> dict[str, Any]:
    calendar = calendar or make_calendar()
    missing: dict[str, list[str]] = {}
    for market in MARKETS:
        paths = [
            _market_input(data_root, market, "physical_hourly"),
            _market_input(data_root, market, "price_hourly"),
        ]
        absent = [str(path) for path in paths if not path.exists()]
        if absent:
            missing[market] = absent
    status = "BLOCKED" if missing else "READY"
    return {
        "schema_version": "energy-model-v3",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "calendar": calendar.to_dict(),
        "all_six_required": True,
        "shortening_dropping_interpolation_forbidden": True,
        "missing_primary_inputs": missing,
        "capabilities": capabilities,
    }
