"""Build deterministic thesis figures from the hash-bound V6 energy windows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
FACTORY_MANIFEST = (
    ROOT / "output" / "energy_model_v3" / "ramp_v6" / "factory_manifest.json"
)
OUTPUT_DIR = ROOT / "docs" / "figures" / "energy_model_v3"
FIGURE_PATH = OUTPUT_DIR / "six_market_normalized_net_load.png"
DAILY_FIGURE_PATH = OUTPUT_DIR / "single_day_normalized_net_load.png"
MANIFEST_PATH = OUTPUT_DIR / "figure_manifest.json"
CAISO_TIMEZONE = ZoneInfo("America/Los_Angeles")

MARKET_ORDER = (
    "CAISO_NP15",
    "ERCOT_LZ_NORTH",
    "NYISO_NYC_J",
    "MISO_MINN_HUB",
    "SPP_NORTH_HUB",
    "ISONE_NEMA",
)
MARKET_LABELS = {
    "CAISO_NP15": "CAISO NP15",
    "ERCOT_LZ_NORTH": "ERCOT North",
    "NYISO_NYC_J": "NYISO Zone J",
    "MISO_MINN_HUB": "MISO Minnesota",
    "SPP_NORTH_HUB": "SPP North",
    "ISONE_NEMA": "ISO-NE NEMA",
}
MARKET_COLORS = {
    "CAISO_NP15": "#0072B2",
    "ERCOT_LZ_NORTH": "#D55E00",
    "NYISO_NYC_J": "#009E73",
    "MISO_MINN_HUB": "#CC79A7",
    "SPP_NORTH_HUB": "#E69F00",
    "ISONE_NEMA": "#56B4E9",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256_candidates(path: Path) -> set[str]:
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    crlf = payload.replace(b"\n", b"\r\n")
    return {
        hashlib.sha256(path.read_bytes()).hexdigest(),
        hashlib.sha256(payload).hexdigest(),
        hashlib.sha256(crlf).hexdigest(),
    }


def load_active_decision_rows() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(FACTORY_MANIFEST.read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    window_count = 0

    for split in ("train", "validation", "test"):
        for window_id, record in sorted(manifest["windows"][split].items()):
            panel_path = (
                ROOT / Path(record["artifact_root"]) / "canonical_panel.csv"
            )
            expected_hash = record["source_hashes"]["canonical_panel"]
            if expected_hash not in text_sha256_candidates(panel_path):
                raise ValueError(f"{window_id} canonical panel hash mismatch")

            panel = pd.read_csv(panel_path)
            panel["timestamp_utc"] = pd.to_datetime(
                panel["timestamp_utc"], utc=True, errors="raise"
            )
            day = pd.Timestamp(record["day"], tz="UTC")
            active = panel.loc[
                (panel["timestamp_utc"] >= day)
                & (panel["timestamp_utc"] < day + pd.Timedelta(days=1))
            ].copy()

            if set(active["market_id"]) != set(MARKET_ORDER):
                raise ValueError(f"{window_id} does not cover all six markets")
            counts = active.groupby("market_id").size()
            if not (counts == 24).all():
                raise ValueError(f"{window_id} lacks 24 active hours per market")
            if active["quality_ok"].astype(bool).eq(False).any():
                raise ValueError(f"{window_id} contains a failed quality row")

            active["split"] = split
            frames.append(active)
            window_count += 1

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["timestamp_utc", "market_id"])
    key = ["timestamp_utc", "market_id"]
    if combined.duplicated(key).any():
        raise ValueError("active decision rows contain duplicate market-hours")

    expected_rows = window_count * 24 * len(MARKET_ORDER)
    if len(combined) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} active rows, found {len(combined)}"
        )
    if (combined["market_scale_mw"] <= 0.0).any():
        raise ValueError("market scale must be positive")

    combined["normalized_net_load"] = (
        combined["net_load_mw"] / combined["market_scale_mw"]
    )
    metadata = {
        "window_count": window_count,
        "active_hours_per_market": window_count * 24,
        "start_utc": combined["timestamp_utc"].min().isoformat(),
        "end_utc": combined["timestamp_utc"].max().isoformat(),
        "market_order": list(MARKET_ORDER),
    }
    return combined, metadata


def plot_normalized_net_load(frame: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    figure, axis = plt.subplots(figsize=(12.0, 5.8))

    for market in MARKET_ORDER:
        rows = frame.loc[frame["market_id"] == market]
        axis.plot(
            rows["timestamp_utc"],
            rows["normalized_net_load"],
            color=MARKET_COLORS[market],
            linewidth=0.55,
            alpha=0.82,
            label=MARKET_LABELS[market],
        )

    validation_start = pd.Timestamp("2026-02-01T00:00:00Z")
    test_start = pd.Timestamp("2026-03-01T00:00:00Z")
    test_end = pd.Timestamp("2026-05-01T00:00:00Z")
    axis.axvspan(
        validation_start,
        test_start,
        color="#7F8C8D",
        alpha=0.08,
        linewidth=0,
    )
    axis.axvspan(
        test_start,
        test_end,
        color="#E69F00",
        alpha=0.06,
        linewidth=0,
    )
    axis.axvline(validation_start, color="#666666", linewidth=0.8, linestyle="--")
    axis.axvline(test_start, color="#666666", linewidth=0.8, linestyle="--")
    axis.text(
        validation_start + pd.Timedelta(days=2),
        0.98,
        "Validation",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="top",
        color="#555555",
        fontsize=9,
    )
    axis.text(
        test_start + pd.Timedelta(days=2),
        0.98,
        "Sealed test",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="top",
        color="#8A5A00",
        fontsize=9,
    )

    axis.set_title("Hourly normalized net load across six market cases", pad=12)
    axis.set_ylabel("Net load / training Q95 gross demand ($N_{m,t}/S_m$)")
    axis.set_xlabel("UTC month")
    axis.set_ylim(bottom=0.0)
    axis.yaxis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        handlelength=2.5,
    )
    figure.text(
        0.01,
        0.01,
        "Source: hash-bound V6 daily canonical panels; raw hourly values, "
        "no smoothing.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    figure.subplots_adjust(bottom=0.27, top=0.90, left=0.09, right=0.99)
    figure.savefig(FIGURE_PATH, dpi=240, bbox_inches="tight")
    plt.close(figure)


def select_duck_curve_day(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    caiso = frame.loc[
        (frame["market_id"] == "CAISO_NP15") & (frame["split"] == "train")
    ].copy()
    caiso["local_timestamp"] = caiso["timestamp_utc"].dt.tz_convert(
        CAISO_TIMEZONE
    )
    caiso["local_date"] = caiso["local_timestamp"].dt.date
    caiso["local_hour"] = caiso["local_timestamp"].dt.hour

    candidates: list[dict[str, Any]] = []
    for local_date, rows in caiso.groupby("local_date"):
        if len(rows) != 24:
            continue
        midday = rows.loc[rows["local_hour"].between(10, 15)]
        evening = rows.loc[rows["local_hour"].between(17, 22)]
        if len(midday) != 6 or len(evening) != 6:
            continue
        trough = midday.loc[midday["normalized_net_load"].idxmin()]
        peak = evening.loc[evening["normalized_net_load"].idxmax()]
        candidates.append(
            {
                "local_date": str(local_date),
                "score": float(
                    peak["normalized_net_load"] - trough["normalized_net_load"]
                ),
                "trough_timestamp_utc": trough["timestamp_utc"].isoformat(),
                "trough_normalized_net_load": float(
                    trough["normalized_net_load"]
                ),
                "peak_timestamp_utc": peak["timestamp_utc"].isoformat(),
                "peak_normalized_net_load": float(peak["normalized_net_load"]),
            }
        )
    if not candidates:
        raise ValueError("no complete CAISO training day is available")

    selected = max(candidates, key=lambda value: (value["score"], value["local_date"]))
    start_local = pd.Timestamp(selected["local_date"], tz=CAISO_TIMEZONE)
    end_local = start_local + pd.Timedelta(days=1)
    selected_frame = frame.loc[
        (frame["timestamp_utc"] >= start_local.tz_convert("UTC"))
        & (frame["timestamp_utc"] < end_local.tz_convert("UTC"))
    ].copy()
    counts = selected_frame.groupby("market_id").size()
    if set(counts.index) != set(MARKET_ORDER) or not (counts == 24).all():
        raise ValueError("selected duck-curve day lacks 24 hours for every market")
    selected.update(
        {
            "selection_split": "train",
            "selection_market": "CAISO_NP15",
            "selection_timezone": str(CAISO_TIMEZONE),
            "selection_criterion": (
                "maximum evening(17-22) minus midday(10-15) normalized net "
                "load among complete CAISO training days"
            ),
            "simultaneous_start_utc": start_local.tz_convert("UTC").isoformat(),
            "simultaneous_end_utc_exclusive": end_local.tz_convert("UTC").isoformat(),
        }
    )
    return selected_frame, selected


def plot_single_day(
    frame: pd.DataFrame,
    selection: dict[str, Any],
) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 5.6))
    for market in MARKET_ORDER:
        rows = frame.loc[frame["market_id"] == market]
        local_time = rows["timestamp_utc"].dt.tz_convert(CAISO_TIMEZONE)
        axis.plot(
            local_time,
            rows["normalized_net_load"],
            color=MARKET_COLORS[market],
            linewidth=2.2 if market == "CAISO_NP15" else 1.25,
            alpha=1.0 if market == "CAISO_NP15" else 0.78,
            label=MARKET_LABELS[market],
            zorder=4 if market == "CAISO_NP15" else 2,
        )

    start_local = pd.Timestamp(selection["local_date"], tz=CAISO_TIMEZONE)
    axis.axvspan(
        start_local + pd.Timedelta(hours=8),
        start_local + pd.Timedelta(hours=18),
        color="#F0E442",
        alpha=0.09,
        linewidth=0,
        zorder=0,
    )
    trough_time = pd.Timestamp(selection["trough_timestamp_utc"]).tz_convert(
        CAISO_TIMEZONE
    )
    peak_time = pd.Timestamp(selection["peak_timestamp_utc"]).tz_convert(
        CAISO_TIMEZONE
    )
    trough_value = selection["trough_normalized_net_load"]
    peak_value = selection["peak_normalized_net_load"]
    axis.scatter(
        [trough_time, peak_time],
        [trough_value, peak_value],
        color=MARKET_COLORS["CAISO_NP15"],
        edgecolor="white",
        linewidth=0.8,
        s=48,
        zorder=6,
    )
    axis.annotate(
        "CAISO midday trough",
        xy=(trough_time, trough_value),
        xytext=(-66, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.8},
        fontsize=9,
    )
    axis.annotate(
        "CAISO evening peak",
        xy=(peak_time, peak_value),
        xytext=(-92, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.8},
        fontsize=9,
    )

    axis.set_title(
        f"Single-day normalized net load: {selection['local_date']} "
        "(Pacific time)",
        pad=12,
    )
    axis.set_ylabel("Net load / training Q95 gross demand ($N_{m,t}/S_m$)")
    axis.set_xlabel("Hour in America/Los_Angeles")
    axis.set_ylim(bottom=0.0)
    axis.yaxis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M", tz=CAISO_TIMEZONE)
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        handlelength=2.5,
    )
    figure.text(
        0.01,
        0.01,
        "Source: simultaneous hash-bound V6 hourly rows; CAISO highlighted. "
        "Yellow band marks 08:00-18:00 Pacific.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    figure.subplots_adjust(bottom=0.27, top=0.89, left=0.09, right=0.99)
    figure.savefig(DAILY_FIGURE_PATH, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, metadata = load_active_decision_rows()
    plot_normalized_net_load(frame)
    daily_frame, daily_selection = select_duck_curve_day(frame)
    plot_single_day(daily_frame, daily_selection)

    manifest = {
        "schema_version": "energy-model-v3-thesis-figures-v1",
        "source_factory_manifest": {
            "path": str(FACTORY_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(FACTORY_MANIFEST),
        },
        "source_scope": {
            **metadata,
            "rows": len(frame),
            "value": "net_load_mw / market_scale_mw",
            "smoothing": False,
            "active_decision_hours_only": True,
        },
        "single_day_selection": daily_selection,
        "outputs": {
            str(FIGURE_PATH.relative_to(ROOT)).replace("\\", "/"): sha256_file(
                FIGURE_PATH
            ),
            str(DAILY_FIGURE_PATH.relative_to(ROOT)).replace(
                "\\", "/"
            ): sha256_file(DAILY_FIGURE_PATH),
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Wrote {FIGURE_PATH.relative_to(ROOT)} and "
        f"{DAILY_FIGURE_PATH.relative_to(ROOT)} from "
        f"{metadata['window_count']} verified windows"
    )


if __name__ == "__main__":
    main()
