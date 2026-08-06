"""Build the timestamped CAISO-archetype energy model v2.

The build requires real CAISO price data and never falls back to synthesis.
Net demand comes from CAISO Today's Outlook historical five-minute data.
The paired reference signals are shifted together across IANA time zones.

No PPO training is launched by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.protocol import V2_2025_PROTOCOL, load_protocol  # noqa: E402

DATA_ROOT = ROOT / "data" / "energy_model_v2"
SCENARIO_ROOT = ROOT / "env" / "scenarios"
PROTOCOL = load_protocol()

CAISO_OUTLOOK = "https://www.caiso.com/outlook/history"
CAISO_OASIS = "https://oasis.caiso.com/oasisapi/SingleZip"
CAISO_NODE = "TH_NP15_GEN-APND"

REFERENCE_ZONE = "America/Los_Angeles"
EXPECTED_ROWS = 31 * 24 * 12

YEAR = 2025
YEAR_ROOT = DATA_ROOT / str(YEAR)
RAW_ROOT = YEAR_ROOT / "raw"
PROCESSED_ROOT = YEAR_ROOT / "processed"
FIG_ROOT = ROOT / "output" / "energy_model_v2" / str(YEAR)
CAISO_OUTLOOK_RAW = RAW_ROOT / "caiso_outlook"
CAISO_OASIS_RAW = RAW_ROOT / "caiso_oasis"
PRICE_CSV = RAW_ROOT / "caiso_dam_np15.csv"
PRICE_METADATA = RAW_ROOT / "caiso_dam_np15.metadata.json"
START_UTC = pd.Timestamp(f"{YEAR}-05-01T07:00:00Z")
END_UTC = pd.Timestamp(f"{YEAR}-06-01T07:00:00Z")
TARGET_UTC = pd.date_range(
    START_UTC, END_UTC, freq="5min", inclusive="left"
)

SLOT_GROUPS = {
    "us": [
        ("pacific", "America/Los_Angeles"),
        ("mountain", "America/Denver"),
        ("central", "America/Chicago"),
        ("eastern", "America/New_York"),
    ],
    "global": [
        ("pacific", "America/Los_Angeles"),
        ("central", "America/Chicago"),
        ("amsterdam", "Europe/Amsterdam"),
        ("singapore", "Asia/Singapore"),
    ],
}

CELL_GROUPS = {
    "ad": ["a", "b", "c", "d"],
    "eh": ["e", "f", "g", "h"],
}


def configure_year(year: int) -> None:
    global YEAR, YEAR_ROOT, RAW_ROOT, PROCESSED_ROOT, FIG_ROOT
    global CAISO_OUTLOOK_RAW, CAISO_OASIS_RAW, PRICE_CSV, PRICE_METADATA
    global START_UTC, END_UTC, TARGET_UTC
    YEAR = year
    YEAR_ROOT = DATA_ROOT / str(year)
    RAW_ROOT = YEAR_ROOT / "raw"
    PROCESSED_ROOT = YEAR_ROOT / "processed"
    FIG_ROOT = ROOT / "output" / "energy_model_v2" / str(year)
    CAISO_OUTLOOK_RAW = RAW_ROOT / "caiso_outlook"
    CAISO_OASIS_RAW = RAW_ROOT / "caiso_oasis"
    PRICE_CSV = RAW_ROOT / "caiso_dam_np15.csv"
    PRICE_METADATA = RAW_ROOT / "caiso_dam_np15.metadata.json"
    START_UTC = pd.Timestamp(f"{year}-05-01T07:00:00Z")
    END_UTC = pd.Timestamp(f"{year}-06-01T07:00:00Z")
    TARGET_UTC = pd.date_range(
        START_UTC, END_UTC, freq="5min", inclusive="left"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_outlook_file(day: pd.Timestamp, name: str) -> pd.DataFrame:
    CAISO_OUTLOOK_RAW.mkdir(parents=True, exist_ok=True)
    date = day.strftime("%Y%m%d")
    path = CAISO_OUTLOOK_RAW / f"{date}_{name}.csv"
    if not path.exists():
        url = f"{CAISO_OUTLOOK}/{date}/{name}.csv"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    return pd.read_csv(path)


def local_timestamp(day: pd.Timestamp, time: pd.Series) -> pd.DatetimeIndex:
    local_naive = pd.to_datetime(
        day.strftime("%Y-%m-%d") + " " + time.astype(str),
        errors="raise",
    )
    return pd.DatetimeIndex(local_naive).tz_localize(
        REFERENCE_ZONE,
        ambiguous="raise",
        nonexistent="raise",
    )


def load_caiso_outlook() -> pd.DataFrame:
    rows = []
    for day in pd.date_range(
        f"{YEAR}-04-30", f"{YEAR}-06-02", freq="1D"
    ):
        net = fetch_outlook_file(day, "netdemand")
        fuel = fetch_outlook_file(day, "fuelsource")
        net["Net demand"] = pd.to_numeric(
            net["Net demand"], errors="coerce"
        )
        net = net.dropna(subset=["Net demand"]).copy()
        if (
            len(net) == 289
            and str(net.iloc[-1]["Time"]) == str(net.iloc[0]["Time"])
        ):
            net = net.iloc[:-1].copy()
        fuel["Solar"] = pd.to_numeric(fuel["Solar"], errors="coerce")
        fuel = fuel.dropna(subset=["Solar"]).copy()
        if len(net) != 288 or len(fuel) != 288:
            raise ValueError(
                f"CAISO Outlook {day.date()} is not a 288-row day: "
                f"net={len(net)}, fuel={len(fuel)}"
            )
        local = local_timestamp(day, net["Time"])
        fuel_local = local_timestamp(day, fuel["Time"])
        if not local.equals(fuel_local):
            raise ValueError(f"net/fuel timestamps differ on {day.date()}")
        rows.append(pd.DataFrame({
            "timestamp_utc": local.tz_convert("UTC"),
            "net_demand_mw": pd.to_numeric(
                net["Net demand"], errors="raise"
            ),
            "solar_mw": pd.to_numeric(fuel["Solar"], errors="raise"),
        }))
    frame = pd.concat(rows, ignore_index=True)
    if frame["timestamp_utc"].duplicated().any():
        raise ValueError("CAISO Outlook input has duplicate UTC timestamps")
    frame["wind_mw"] = 0.0
    return frame.set_index("timestamp_utc").sort_index()


def validate_price_metadata(metadata: dict[str, Any]) -> None:
    required = {
        "source",
        "url",
        "market",
        "node",
        "native_interval",
        "timestamp_semantics",
        "retrieved_at_utc",
        "units",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(
            f"price metadata missing required fields: {', '.join(missing)}"
        )
    if metadata["units"] != "USD/MWh":
        raise ValueError("price metadata units must be USD/MWh")
    if metadata["timestamp_semantics"] != "interval_start":
        raise ValueError(
            "price timestamp_semantics must be interval_start"
        )


def oasis_datetime(timestamp: pd.Timestamp) -> str:
    return timestamp.tz_convert("UTC").strftime("%Y%m%dT%H:%M-0000")


def fetch_oasis_day(day: pd.Timestamp) -> pd.DataFrame:
    CAISO_OASIS_RAW.mkdir(parents=True, exist_ok=True)
    date = day.strftime("%Y%m%d")
    path = CAISO_OASIS_RAW / f"{date}_dam_lmp_np15.csv"
    if path.exists():
        return pd.read_csv(path)

    local_start = pd.Timestamp(day.date()).tz_localize(REFERENCE_ZONE)
    local_end = local_start + pd.DateOffset(days=1)
    params = {
        "queryname": "PRC_LMP",
        "version": "12",
        "market_run_id": "DAM",
        "startdatetime": oasis_datetime(local_start),
        "enddatetime": oasis_datetime(local_end),
        "node": CAISO_NODE,
        "resultformat": "6",
    }
    delay = 2.0
    for attempt in range(7):
        response = requests.get(CAISO_OASIS, params=params, timeout=120)
        if response.status_code == 429:
            time.sleep(delay)
            delay = min(delay * 2.0, 60.0)
            continue
        response.raise_for_status()
        if response.content[:2] != b"PK":
            raise RuntimeError(
                f"CAISO OASIS returned non-ZIP for {date}: "
                f"{response.text[:200]}"
            )
        import zipfile

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = archive.namelist()
            if len(names) != 1 or not names[0].endswith(".csv"):
                error = archive.read(names[0])[:500].decode(
                    errors="replace"
                )
                raise RuntimeError(
                    f"CAISO OASIS has no CSV for {date}: {error}"
                )
            frame = pd.read_csv(archive.open(names[0]))
        frame.to_csv(path, index=False)
        time.sleep(0.5)
        return frame
    raise RuntimeError(f"CAISO OASIS rate limit persisted for {date}")


def fetch_caiso_prices() -> None:
    if PRICE_CSV.exists() and PRICE_METADATA.exists():
        return
    rows = []
    for day in pd.date_range(
        f"{YEAR}-04-30", f"{YEAR}-06-01", freq="1D"
    ):
        frame = fetch_oasis_day(day)
        lmp = frame[frame["LMP_TYPE"] == "LMP"].copy()
        if len(lmp) != 24:
            raise ValueError(
                f"CAISO DAM LMP {day.date()} has {len(lmp)} total-LMP rows"
            )
        rows.append(pd.DataFrame({
            "timestamp_utc": pd.to_datetime(
                lmp["INTERVALSTARTTIME_GMT"], utc=True
            ),
            "price_usd_mwh": pd.to_numeric(
                lmp["VALUE"] if "VALUE" in lmp else lmp["MW"],
                errors="raise",
            ),
        }))
    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values("timestamp_utc")
    if combined["timestamp_utc"].duplicated().any():
        raise ValueError("combined CAISO price data contain duplicates")
    PRICE_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PRICE_CSV, index=False)
    metadata = {
        "source": "CAISO OASIS",
        "url": CAISO_OASIS,
        "queryname": "PRC_LMP",
        "version": 12,
        "market": "DAM",
        "node": CAISO_NODE,
        "native_interval": "1 hour",
        "timestamp_semantics": "interval_start",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "units": "USD/MWh",
        "query_window": (
            f"{YEAR}-04-30 through {YEAR}-06-01 local delivery days"
        ),
    }
    PRICE_METADATA.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def load_price() -> tuple[pd.Series, dict[str, Any]]:
    fetch_caiso_prices()
    metadata = json.loads(PRICE_METADATA.read_text(encoding="utf-8"))
    validate_price_metadata(metadata)
    frame = pd.read_csv(PRICE_CSV)
    expected = {"timestamp_utc", "price_usd_mwh"}
    if not expected.issubset(frame.columns):
        raise ValueError(
            f"{PRICE_CSV} must contain {sorted(expected)}"
        )
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], utc=True, errors="coerce"
    )
    frame["price_usd_mwh"] = pd.to_numeric(
        frame["price_usd_mwh"], errors="coerce"
    )
    frame = frame.dropna().sort_values("timestamp_utc")
    if frame["timestamp_utc"].duplicated().any():
        raise ValueError("price input contains duplicate UTC timestamps")
    return frame.set_index("timestamp_utc")["price_usd_mwh"], metadata


def required_reference_index() -> pd.DatetimeIndex:
    indices = []
    for slots in SLOT_GROUPS.values():
        for _, zone in slots:
            local_wall = TARGET_UTC.tz_convert(zone).tz_localize(None)
            reference_local = local_wall.tz_localize(
                REFERENCE_ZONE,
                ambiguous="raise",
                nonexistent="raise",
            )
            indices.append(reference_local.tz_convert("UTC"))
    combined = indices[0]
    for index in indices[1:]:
        combined = combined.union(index)
    return combined.sort_values()


def expand_price(
    values: pd.Series,
    target: pd.DatetimeIndex,
    metadata: dict[str, Any],
) -> pd.Series:
    if len(values) < 2:
        raise ValueError("price input has fewer than two observations")
    cadence = values.index.to_series().diff().dropna().median()
    declared = str(metadata["native_interval"]).lower()
    if cadence <= pd.Timedelta(minutes=5):
        expanded = values.resample("5min").mean()
        expanded = expanded.reindex(
            expanded.index.union(target)
        ).interpolate("time")
    elif cadence <= pd.Timedelta(hours=1):
        # Day-ahead hourly prices clear for the delivery hour; do not invent
        # intra-hour slopes by linear interpolation.
        expanded = values.resample("5min").ffill()
    else:
        raise ValueError(
            f"unsupported price cadence {cadence}; metadata={declared}"
        )
    return expanded.reindex(target)


def reference_signals() -> tuple[pd.DataFrame, dict[str, Any]]:
    reference_index = required_reference_index()
    outlook = load_caiso_outlook()
    price, price_metadata = load_price()

    reference = outlook.reindex(reference_index).copy()
    for column in ("net_demand_mw", "solar_mw", "wind_mw"):
        if reference[column].isna().any():
            raise ValueError(
                f"CAISO Outlook coverage incomplete for {column}"
            )
    reference["price_usd_mwh"] = expand_price(
        price, reference_index, price_metadata
    )
    if reference.isna().any().any():
        missing = reference.isna().sum()
        raise ValueError(f"reference coverage incomplete:\n{missing}")
    return reference, price_metadata


def shifted_reference_index(zone: str) -> pd.DatetimeIndex:
    local_wall = TARGET_UTC.tz_convert(zone).tz_localize(None)
    return local_wall.tz_localize(
        REFERENCE_ZONE,
        ambiguous="raise",
        nonexistent="raise",
    ).tz_convert("UTC")


def build_slot(
    reference: pd.DataFrame,
    zone: str,
    normalization_scale_mw: float,
    solar_peak_mw: float,
) -> pd.DataFrame:
    lookup = shifted_reference_index(zone)
    values = reference.reindex(lookup)
    if values.isna().any().any():
        raise ValueError(f"reference lookup missing values for {zone}")
    result = pd.DataFrame({
        "timestep": np.arange(EXPECTED_ROWS),
        "timestamp_utc": TARGET_UTC.astype(str),
        "timestamp_local": TARGET_UTC.tz_convert(zone).astype(str),
        "reference_timestamp_utc": lookup.astype(str),
        "price_usd_kwh": values["price_usd_mwh"].to_numpy() / 1000.0,
        "net_demand_mw": values["net_demand_mw"].to_numpy(),
        "net_demand_signed": np.clip(
            values["net_demand_mw"].to_numpy()
            / normalization_scale_mw,
            -1.0,
            1.0,
        ),
        "solar_fraction": np.clip(
            values["solar_mw"].to_numpy() / solar_peak_mw,
            0.0,
            1.0,
        ),
    })
    if len(result) != EXPECTED_ROWS:
        raise AssertionError("processed slot is not a complete May calendar")
    if result.isna().any().any():
        raise AssertionError("processed slot contains missing values")
    return result


def site_config(
    scenario: str,
    slot: str,
    cell: str,
) -> dict[str, Any]:
    prefix = "US" if scenario == "us" else "Global"
    return {
        "name": f"{prefix}-{slot.title()}",
        "cell": f"data/cells/cell_{cell}.csv",
        "tier_curves": f"data/cells/cell_{cell}_tiers.csv",
        "solar": (
            f"data/energy_model_v2/{YEAR}/processed/"
            f"{scenario}_{slot}.csv"
        ),
        "price": (
            f"data/energy_model_v2/{YEAR}/processed/"
            f"{scenario}_{slot}.csv"
        ),
        "net_demand": (
            f"data/energy_model_v2/{YEAR}/processed/"
            f"{scenario}_{slot}.csv"
        ),
        "net_demand_column": "net_demand_signed",
        "machines": f"data/machines/machines_{cell}.csv",
        "rated_power_mw": PROTOCOL["proxy_dc"]["rated_power_mw"],
        "capacity": PROTOCOL["proxy_dc"]["capacity"],
        "memory_capacity": PROTOCOL["proxy_dc"]["memory_capacity"],
        "batch_distributions": (
            f"data/jobs/batch_distributions_{cell}.json"
        ),
    }


def write_scenarios() -> list[Path]:
    written = []
    for scenario, slots in SLOT_GROUPS.items():
        for group, cells in CELL_GROUPS.items():
            suffix = "" if group == "ad" else "_eh"
            config = {
                "_energy_model": (
                    f"v2-{YEAR}: one real CAISO price/net-demand archetype "
                    "shifted by IANA local wall time; no regional re-averaging"
                ),
                "_capacity_model": (
                    "equal 100 MW proxy DCs; source cell utilization maps to "
                    "unit normalized capacity at every site"
                ),
                "_protocol": "env/protocols/v2_2025.yaml",
                "power_model": "data/power_model_params.json",
                "per_cell_power": True,
                "batch": {
                    "flexibility_factor": PROTOCOL["batch"][
                        "primary_flexibility_factor"
                    ],
                    "deadline_penalty_weight": PROTOCOL["objective"][
                        "deadline_penalty_weight"
                    ],
                    "urgency_horizon_steps": PROTOCOL["batch"][
                        "urgency_horizon_steps"
                    ],
                },
                "sites": [
                    site_config(scenario, slot, cell)
                    for (slot, _), cell in zip(slots, cells)
                ],
            }
            path = (
                SCENARIO_ROOT
                / f"{scenario}_model{suffix}_v2_{YEAR}.yaml"
            )
            path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            written.append(path)
    return written


def hourly_profile(frame: pd.DataFrame, value: str) -> np.ndarray:
    utc_hour = TARGET_UTC.hour
    data = frame[value].to_numpy()
    return np.array([data[utc_hour == hour].mean() for hour in range(24)])


def plot_reference(
    reference_slot: pd.DataFrame,
    processed: dict[str, pd.DataFrame],
) -> list[Path]:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    paths = []

    local = pd.to_datetime(reference_slot["timestamp_local"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(local, reference_slot["price_usd_kwh"] * 1000.0)
    axes[0].set_ylabel("CAISO price ($/MWh)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(local, reference_slot["net_demand_mw"])
    axes[1].set_ylabel("CAISO net demand (MW)")
    axes[1].set_xlabel("Reference local calendar")
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"Energy model v2 reference: real CAISO May {YEAR}")
    fig.tight_layout()
    path = FIG_ROOT / "reference_month.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    for scenario in SLOT_GROUPS:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for slot, _ in SLOT_GROUPS[scenario]:
            frame = processed[f"{scenario}_{slot}"]
            axes[0].plot(
                range(24),
                hourly_profile(frame, "price_usd_kwh") * 1000.0,
                label=slot,
            )
            axes[1].plot(
                range(24),
                hourly_profile(frame, "net_demand_signed"),
                label=slot,
            )
        axes[0].set_ylabel("Average price ($/MWh)")
        axes[1].set_ylabel("Average normalized net demand")
        axes[1].set_xlabel("Experiment UTC hour")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(ncol=2)
        fig.suptitle(
            f"Energy model v2 shifted profiles — {scenario.upper()}"
        )
        fig.tight_layout()
        path = FIG_ROOT / f"{scenario}_shifted_daily_profiles.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def diagnostics(processed: dict[str, pd.DataFrame]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, frame in processed.items():
        price = frame["price_usd_kwh"].to_numpy()
        demand = frame["net_demand_signed"].to_numpy()
        price_profile = hourly_profile(frame, "price_usd_kwh")
        demand_profile = hourly_profile(frame, "net_demand_signed")
        result[name] = {
            "rows": len(frame),
            "price_mean_usd_mwh": float(price.mean() * 1000.0),
            "price_std_usd_mwh": float(price.std() * 1000.0),
            "net_demand_mean_signed": float(demand.mean()),
            "price_net_demand_correlation": float(
                np.corrcoef(price, demand)[0, 1]
            ),
            "average_price_peak_utc_hour": int(price_profile.argmax()),
            "average_net_demand_peak_utc_hour": int(
                demand_profile.argmax()
            ),
        }
    return result


def build() -> None:
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    reference, price_metadata = reference_signals()

    la_lookup = shifted_reference_index(REFERENCE_ZONE)
    la_values = reference.reindex(la_lookup)
    normalization_scale = float(
        np.max(np.abs(la_values["net_demand_mw"]))
    )
    solar_peak = max(float(la_values["solar_mw"].max()), 1.0)

    processed = {}
    for scenario, slots in SLOT_GROUPS.items():
        for slot, zone in slots:
            frame = build_slot(
                reference, zone, normalization_scale, solar_peak
            )
            path = PROCESSED_ROOT / f"{scenario}_{slot}.csv"
            frame.to_csv(path, index=False)
            processed[f"{scenario}_{slot}"] = frame

    scenarios = write_scenarios()
    figures = plot_reference(processed["us_pacific"], processed)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": (
            "one real CAISO price/net-demand archetype shifted by target "
            "local wall time; price level unchanged"
        ),
        "protocol": {
            "path": str(V2_2025_PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(V2_2025_PROTOCOL),
        },
        "capacity_model": PROTOCOL["proxy_dc"]["capacity_semantics"],
        "reference_zone": REFERENCE_ZONE,
        "calendar": {
            "start_utc": str(START_UTC),
            "end_utc": str(END_UTC),
            "rows": EXPECTED_ROWS,
            "cadence": "5 minutes",
        },
        "slots": SLOT_GROUPS,
        "workload_anchor": (
            "workload timestep 0 is assigned to experiment start UTC as a "
            "modeling convention; original Google UTC anchor was discarded"
        ),
        "net_demand_source": {
            "url_pattern": (
                f"{CAISO_OUTLOOK}/YYYYMMDD/netdemand.csv"
            ),
            "fuel_url_pattern": (
                f"{CAISO_OUTLOOK}/YYYYMMDD/fuelsource.csv"
            ),
            "native_interval": "5 minutes",
            "timezone": REFERENCE_ZONE,
            "raw_files": {
                str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                for path in sorted(CAISO_OUTLOOK_RAW.glob("*.csv"))
            },
        },
        "price_source": {
            **price_metadata,
            "raw_sha256": sha256(PRICE_CSV),
        },
        "net_demand_signed_scale_mw": normalization_scale,
        "net_demand_signed_scale": (
            "net_demand_mw divided by the maximum absolute net demand in the "
            "Pacific May-2025 reference window; clipped to [-1, 1]"
        ),
        "processed": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in sorted(PROCESSED_ROOT.glob("*.csv"))
        },
        "scenarios": [
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in scenarios
        ],
        "figures": [
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in figures
        ],
        "diagnostics": diagnostics(processed),
    }
    (YEAR_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(processed)} market slots and {len(scenarios)} scenarios")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        choices=[2023, 2024, 2025],
        help="CAISO May archetype year. The active thesis model uses 2025.",
    )
    args = parser.parse_args()
    configure_year(args.year)
    build()


if __name__ == "__main__":
    main()
