"""Fetch a full calendar year of EIA grid data and build canonical panels.

Replaces the opaque pre-built panels with a reproducible pull from the EIA
Open Data API. Three series per balancing authority -- demand, wind, and
solar -- for the four market cases, at hourly resolution.

Two conventions are worth stating up front because they are easy to get wrong
and expensive to get wrong silently.

Hour labelling. EIA's hourly ``period`` labels the END of the interval it
describes, so the row labelled 21:00 covers 20:00-21:00 UTC. This was checked
empirically by comparing each market's solar curve against astronomical solar
noon across four dates: fourteen of sixteen comparisons favour hour-ending,
several to within a few minutes. Panels here are stamped hour-BEGINNING, so
panel row T takes the EIA row labelled T+1. See ``HOUR_LABEL_IS_ENDING``.

Negative renewables. CAISO reports small negative solar overnight, a
measurement artifact rather than real generation. Values are clipped at zero
and the count is reported.

Prices are deliberately absent. Day-ahead cost is not part of this study.

    python scripts/fetch_eia_panels.py --report   # fetch and summarize only
    python scripts/fetch_eia_panels.py --write    # also write panels
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data" / "four_market_2025"

API = "https://api.eia.gov/v2/electricity/rto/"

# market case -> EIA balancing-authority respondent code
MARKETS = {
    "CAISO_NP15": "CISO",
    "MISO_MINN_HUB": "MISO",
    "SPP_NORTH_HUB": "SWPP",
    "ISONE_NEMA": "ISNE",
}

# EIA hourly periods label the end of their interval; panels are hour-beginning.
HOUR_LABEL_IS_ENDING = True

# The simulated year, in panel (hour-beginning) terms.
YEAR_START = pd.Timestamp("2025-01-01T00:00:00Z")
YEAR_END = pd.Timestamp("2025-12-31T23:00:00Z")

# Every episode needs three warm hours before its day and a three-hour tail
# after it, so the retained span runs from 21:00 the day before the first
# episode to 02:00 the day after the last.
PANEL_START = YEAR_START - pd.Timedelta(hours=3)
PANEL_END = YEAR_END + pd.Timedelta(hours=3)

VALIDATION_MONTH = 5  # May is held out; every other month trains.

REQUEST_PAUSE_S = 0.3


def _get(path: str, facets: list[tuple[str, str]], start: str, end: str, key: str) -> list[dict]:
    params = [
        ("api_key", key),
        ("frequency", "hourly"),
        ("data[0]", "value"),
        ("start", start),
        ("end", end),
        ("length", "5000"),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
    ] + facets
    url = API + path + "/data/?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.loads(response.read().decode())
    return payload["response"]["data"]


def read_api_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("EIA_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("EIA_API_KEY not found in .env")


def fetch_series(respondent: str, kind: str, key: str) -> pd.Series:
    """One hourly series, indexed by the panel timestamp it belongs to."""
    if kind == "demand":
        path, facets = "region-data", [("facets[respondent][]", respondent),
                                       ("facets[type][]", "D")]
    else:
        fuel = {"wind": "WND", "solar": "SUN"}[kind]
        path, facets = "fuel-type-data", [("facets[respondent][]", respondent),
                                          ("facets[fueltype][]", fuel)]

    # Requests are chunked by quarter to stay well under the 5000-row cap.
    lo = PANEL_START + pd.Timedelta(hours=1 if HOUR_LABEL_IS_ENDING else 0)
    hi = PANEL_END + pd.Timedelta(hours=1 if HOUR_LABEL_IS_ENDING else 0)
    rows: list[dict] = []
    cursor = lo
    while cursor <= hi:
        stop = min(cursor + pd.Timedelta(days=92), hi)
        rows += _get(path, facets, cursor.strftime("%Y-%m-%dT%H"),
                     stop.strftime("%Y-%m-%dT%H"), key)
        cursor = stop + pd.Timedelta(hours=1)
        time.sleep(REQUEST_PAUSE_S)

    if not rows:
        raise RuntimeError(f"no rows returned for {respondent} {kind}")

    stamps = pd.to_datetime([r["period"] for r in rows], format="%Y-%m-%dT%H", utc=True)
    values = pd.to_numeric([r["value"] for r in rows], errors="coerce")
    series = pd.Series(values, index=stamps).groupby(level=0).last().sort_index()

    # Re-stamp from hour-ending to hour-beginning.
    if HOUR_LABEL_IS_ENDING:
        series.index = series.index - pd.Timedelta(hours=1)
    return series


def build_frame(key: str) -> tuple[pd.DataFrame, dict]:
    """Assemble the long-form hourly frame for all four markets."""
    index = pd.date_range(PANEL_START, PANEL_END, freq="h", tz="UTC")
    notes: dict[str, dict] = {}
    parts = []

    for market, respondent in MARKETS.items():
        columns = {}
        for kind in ("demand", "wind", "solar"):
            series = fetch_series(respondent, kind, key).reindex(index)
            columns[kind] = series
        frame = pd.DataFrame(columns, index=index)

        missing = {k: int(frame[k].isna().sum()) for k in frame}
        # Short gaps are bridged; anything longer is a real hole and is reported.
        frame = frame.interpolate(limit=3, limit_direction="both")
        still_missing = {k: int(frame[k].isna().sum()) for k in frame}

        negatives = {k: int((frame[k] < 0).sum()) for k in ("wind", "solar")}
        for k in ("wind", "solar"):
            frame[k] = frame[k].clip(lower=0.0)

        notes[market] = {"missing_before_fill": missing,
                         "missing_after_fill": still_missing,
                         "negatives_clipped": negatives}

        out = pd.DataFrame({
            "timestamp_utc": index,
            "market_id": market,
            "gross_demand_mw": frame["demand"].to_numpy(),
            "wind_mw": frame["wind"].to_numpy(),
            "solar_mw": frame["solar"].to_numpy(),
        })
        out["net_load_mw"] = (out.gross_demand_mw - out.wind_mw - out.solar_mw)
        parts.append(out)

    return pd.concat(parts, ignore_index=True), notes


def training_mask(stamps: pd.Series) -> np.ndarray:
    """True for hours inside the simulated year but outside the holdout month."""
    inside = (stamps >= YEAR_START) & (stamps <= YEAR_END)
    return (inside & (stamps.dt.month != VALIDATION_MONTH)).to_numpy()


def compute_scales(frame: pd.DataFrame) -> dict[str, float]:
    """Ramp normalizer S_m: gross-demand Q95 over training hours only."""
    mask = training_mask(frame["timestamp_utc"])
    scales = {}
    for market, group in frame[mask].groupby("market_id"):
        scales[market] = float(group["gross_demand_mw"].quantile(0.95))
    return scales


def write_panels(frame: pd.DataFrame, scales: dict[str, float]) -> tuple[int, dict]:
    """Write one 30-hour window per simulated day, plus the calendar."""
    frame = frame.copy()
    frame["market_scale_mw"] = frame["market_id"].map(scales)

    # Placeholder forecasts so the panels validate immediately; the real ones
    # are written later by scripts/build_causal_forecasts.py.
    frame = frame.sort_values(["market_id", "timestamp_utc"])
    for quantity, column in (("gross", "gross_demand_mw"), ("net", "net_load_mw")):
        for hour in (1, 2, 3):
            frame[f"forecast_{quantity}_h{hour}_mw"] = frame[column]
    frame["forecast_issue_time_utc"] = frame["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["forecast_vintage_id"] = "persistence-placeholder"
    frame["quality_ok"] = True

    windows_root = OUT_ROOT / "windows"
    windows_root.mkdir(parents=True, exist_ok=True)

    days = pd.date_range(YEAR_START, YEAR_END, freq="D", tz="UTC")
    indexed = frame.set_index("timestamp_utc").sort_index()
    train, validation = [], []

    for day in days:
        lo = day - pd.Timedelta(hours=3)
        hi = day + pd.Timedelta(hours=26)
        window = indexed.loc[lo:hi].reset_index()
        expected = 30 * len(MARKETS)
        if len(window) != expected:
            raise RuntimeError(f"{day.date()}: {len(window)} rows, expected {expected}")

        window_id = f"{day.date()}-daily"
        directory = windows_root / window_id
        directory.mkdir(parents=True, exist_ok=True)
        window.to_csv(directory / "canonical_panel.csv", index=False)

        entry = {"window_id": window_id, "day": str(day.date()), "month": int(day.month),
                 "panel_path": f"data/four_market_2025/windows/{window_id}/canonical_panel.csv"}
        (validation if day.month == VALIDATION_MONTH else train).append(entry)

    calendar = {
        "markets": list(MARKETS),  # canonical order, matches MARKET_TO_CELL
        "frozen_stats_path": "data/four_market_2025/frozen_stats.json",
        "forecast_model": "persistence-placeholder",
        "hour_label_convention": "panel timestamps are hour-beginning UTC",
        "validation_month": VALIDATION_MONTH,
        "train": train,
        "validation": validation,
        "test": [],
    }
    (OUT_ROOT / "calendar.json").write_text(
        json.dumps(calendar, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # The environment also standardizes its level and forecast features, so
    # the frozen statistics carry means and standard deviations alongside the
    # ramp normalizer. All are fitted on training hours only.
    mask = training_mask(frame["timestamp_utc"])
    training = frame[mask]
    grouped = training.groupby("market_id")
    stats = {
        "fit_start_utc": YEAR_START.isoformat(),
        "fit_end_utc": YEAR_END.isoformat(),
        "fit_months": [f"2025-{month:02d}" for month in range(1, 13)
                       if month != VALIDATION_MONTH],
        "gross_q95_mw": scales,
        "gross_level_mean_mw": {k: float(v) for k, v in
                                grouped["gross_demand_mw"].mean().items()},
        "gross_level_std_mw": {k: float(v) for k, v in
                               grouped["gross_demand_mw"].std().items()},
        "net_level_mean_mw": {k: float(v) for k, v in
                              grouped["net_load_mw"].mean().items()},
        "net_level_std_mw": {k: float(v) for k, v in
                             grouped["net_load_mw"].std().items()},
        "stats_id": "four-market-2025-train-only-stats-v1",
    }
    (OUT_ROOT / "frozen_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return len(days), {"train_days": len(train), "validation_days": len(validation)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="fetch and summarize only")
    parser.add_argument("--write", action="store_true", help="also write panels")
    args = parser.parse_args()
    if not (args.report or args.write):
        parser.error("pass --report or --write")

    key = read_api_key()
    print(f"fetching {PANEL_START} .. {PANEL_END}  (hour-beginning panel stamps)")
    frame, notes = build_frame(key)

    print(f"\nrows: {len(frame)}  markets: {frame.market_id.nunique()}  "
          f"hours: {frame.timestamp_utc.nunique()}")
    for market, note in notes.items():
        print(f"  {market:16s} gaps_filled={note['missing_before_fill']} "
              f"remaining={note['missing_after_fill']} clipped={note['negatives_clipped']}")

    scales = compute_scales(frame)
    print("\nramp normalizer S_m (training hours only):")
    for market in sorted(scales):
        print(f"  {market:16s} {scales[market]:10,.0f} MW")

    mask = training_mask(frame["timestamp_utc"])
    print(f"\ntraining hours: {int(mask.sum()) // len(MARKETS)}  "
          f"holdout month: {VALIDATION_MONTH}")

    if args.write:
        days, counts = write_panels(frame, scales)
        print(f"\nwrote {days} windows to {OUT_ROOT}")
        print(f"  train days={counts['train_days']}  validation days={counts['validation_days']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
