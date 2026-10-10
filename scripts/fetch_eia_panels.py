"""Fetch a full calendar year of EIA grid data and build canonical monthly panels.

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

Panel layout. One file per calendar month, holding the warm hour before the
month and every hour of the month. The scheduler runs each month as one
continuous episode; the warm hour gives the first decision its lookback and
the first ramp its starting point.

    python scripts/fetch_eia_panels.py --report    # fetch and summarize only
    python scripts/fetch_eia_panels.py --write     # also write the 2025 panels
    python scripts/fetch_eia_panels.py --test-raw  # fetch the May 2026 test window

``--write`` stores placeholder forecasts, the development-year statistics, and
the calendar; ``scripts/build_data_splits.py`` then writes the real forecasts,
the test panel, and the cross-validation data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from energy_model_v3.splits import DEVELOPMENT_MONTHS, frozen_stats  # noqa: E402
from env.ramp_v6.models import FORECAST_HOURS, HISTORY_HOURS  # noqa: E402

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
YEAR = 2025
YEAR_START = pd.Timestamp(f"{YEAR}-01-01T00:00:00Z")
YEAR_END = pd.Timestamp(f"{YEAR}-12-31T23:00:00Z")

# Every month needs HISTORY_HOURS warm hours before it, so the retained span
# starts that many hours before the year. Nothing after the year is needed:
# each month closes with empty queues in its own final hour.
PANEL_START = YEAR_START - pd.Timedelta(hours=HISTORY_HOURS)
PANEL_END = YEAR_END

# Test month, outside the simulated year. The raw pull starts a week early so
# the forecast inputs (lags up to 168 hours) exist for its first hours, and it
# ends twelve hours late so the last issue hours' longest-lead targets exist.
# Scales and forecasts are not computed here; the test month only reuses
# quantities frozen from training data.
TEST_MONTH = "2026-05"
TEST_ROOT = ROOT / "data" / "four_market_2026_test"
TEST_FETCH_START = pd.Timestamp("2026-04-23T00:00:00Z")
TEST_FETCH_END = pd.Timestamp("2026-06-01T11:00:00Z")

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


def fetch_series(
    respondent: str, kind: str, key: str,
    start: pd.Timestamp = PANEL_START, end: pd.Timestamp = PANEL_END,
) -> pd.Series:
    """One hourly series over panel stamps ``start``..``end``, hour-beginning."""
    if kind == "demand":
        path, facets = "region-data", [("facets[respondent][]", respondent),
                                       ("facets[type][]", "D")]
    else:
        fuel = {"wind": "WND", "solar": "SUN"}[kind]
        path, facets = "fuel-type-data", [("facets[respondent][]", respondent),
                                          ("facets[fueltype][]", fuel)]

    # Requests are chunked by quarter to stay well under the 5000-row cap.
    lo = start + pd.Timedelta(hours=1 if HOUR_LABEL_IS_ENDING else 0)
    hi = end + pd.Timedelta(hours=1 if HOUR_LABEL_IS_ENDING else 0)
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


def build_frame(
    key: str, start: pd.Timestamp = PANEL_START, end: pd.Timestamp = PANEL_END,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the long-form hourly frame for all four markets."""
    index = pd.date_range(start, end, freq="h", tz="UTC")
    notes: dict[str, dict] = {}
    parts = []

    for market, respondent in MARKETS.items():
        columns = {}
        for kind in ("demand", "wind", "solar"):
            series = fetch_series(respondent, kind, key, start, end).reindex(index)
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


def development_stats(frame: pd.DataFrame) -> dict:
    """S_m and the observation standardizers, fitted on all twelve 2025 months.

    Cross-validation folds refit these on their own training months; see
    ``scripts/build_data_splits.py``.
    """
    return frozen_stats(frame, DEVELOPMENT_MONTHS, "four-market-2025-development-stats-v2")


def compute_scales(frame: pd.DataFrame) -> dict[str, float]:
    """Ramp normalizer S_m: gross-demand Q95 over the development months."""
    return development_stats(frame)["gross_q95_mw"]


def month_entries() -> list[dict]:
    """Calendar entries for the twelve monthly panels, in order."""
    entries = []
    for month in range(1, 13):
        start = pd.Timestamp(f"{YEAR}-{month:02d}-01T00:00:00Z")
        end = start + pd.offsets.MonthBegin(1)
        month_id = f"{YEAR}-{month:02d}"
        entries.append({
            "window_id": month_id,
            "month_id": month_id,
            "month": month,
            "hours": int((end - start) / pd.Timedelta(hours=1)),
            "history_hours": HISTORY_HOURS,
            "panel_path": f"data/four_market_2025/months/{month_id}/canonical_panel.csv",
        })
    return entries


def write_panels(frame: pd.DataFrame, scales: dict[str, float]) -> tuple[int, dict]:
    """Write one continuous panel per calendar month, plus the calendar."""
    frame = frame.copy()
    frame["market_scale_mw"] = frame["market_id"].map(scales)

    # Placeholder forecasts so the panels validate immediately; the real ones
    # are written later by scripts/build_causal_forecasts.py.
    frame = frame.sort_values(["market_id", "timestamp_utc"])
    for quantity, column in (("gross", "gross_demand_mw"), ("net", "net_load_mw")):
        for hour in FORECAST_HOURS:
            frame[f"forecast_{quantity}_h{hour}_mw"] = frame[column]
    frame["forecast_issue_time_utc"] = frame["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["forecast_vintage_id"] = "persistence-placeholder"
    frame["quality_ok"] = True

    months_root = OUT_ROOT / "months"
    months_root.mkdir(parents=True, exist_ok=True)

    indexed = frame.set_index("timestamp_utc").sort_index()
    entries = month_entries()
    for entry in entries:
        start = pd.Timestamp(f"{entry['month_id']}-01T00:00:00Z")
        lo = start - pd.Timedelta(hours=HISTORY_HOURS)
        hi = start + pd.Timedelta(hours=entry["hours"] - 1)
        block = indexed.loc[lo:hi].reset_index()
        expected = (entry["hours"] + HISTORY_HOURS) * len(MARKETS)
        if len(block) != expected:
            raise RuntimeError(f"{entry['month_id']}: {len(block)} rows, expected {expected}")
        block = block.sort_values(["timestamp_utc", "market_id"])

        directory = months_root / entry["month_id"]
        directory.mkdir(parents=True, exist_ok=True)
        block.to_csv(directory / "canonical_panel.csv", index=False, lineterminator="\n")

    # The calendar lists the development months, the cross-validation folds,
    # and the test month; scripts/build_data_splits.py records the real
    # forecast model once it has written the forecasts.
    from scripts.build_data_splits import calendar_payload

    calendar = calendar_payload()
    calendar["forecast_model"] = "persistence-placeholder"
    (OUT_ROOT / "calendar.json").write_text(
        json.dumps(calendar, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    # The environment also standardizes its level and forecast features, so
    # the frozen statistics carry means and standard deviations alongside the
    # ramp normalizer, all fitted on the twelve development months.
    (OUT_ROOT / "frozen_stats.json").write_text(
        json.dumps(development_stats(frame), indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    return len(entries), {"development_months": len(entries)}


def write_test_raw(key: str) -> Path:
    """Fetch the test window and store the raw hourly series, nothing derived."""
    frame, notes = build_frame(key, TEST_FETCH_START, TEST_FETCH_END)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_ROOT / "raw_hourly.csv"
    frame.sort_values(["timestamp_utc", "market_id"]).to_csv(path, index=False)
    (TEST_ROOT / "raw_notes.json").write_text(json.dumps({
        "test_month": TEST_MONTH,
        "fetch_start_utc": TEST_FETCH_START.isoformat(),
        "fetch_end_utc": TEST_FETCH_END.isoformat(),
        "hour_label_convention": "panel timestamps are hour-beginning UTC",
        "markets": notes,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for market, note in notes.items():
        print(f"  {market:16s} gaps_filled={note['missing_before_fill']} "
              f"remaining={note['missing_after_fill']} clipped={note['negatives_clipped']}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="fetch and summarize only")
    parser.add_argument("--write", action="store_true", help="also write panels")
    parser.add_argument("--test-raw", action="store_true",
                        help=f"fetch only the {TEST_MONTH} test window and store raw series")
    args = parser.parse_args()
    if not (args.report or args.write or args.test_raw):
        parser.error("pass --report, --write, or --test-raw")

    key = read_api_key()
    if args.test_raw:
        print(f"fetching test window {TEST_FETCH_START} .. {TEST_FETCH_END}")
        print(f"wrote {write_test_raw(key)}")
        return 0
    print(f"fetching {PANEL_START} .. {PANEL_END}  (hour-beginning panel stamps)")
    frame, notes = build_frame(key)

    print(f"\nrows: {len(frame)}  markets: {frame.market_id.nunique()}  "
          f"hours: {frame.timestamp_utc.nunique()}")
    for market, note in notes.items():
        print(f"  {market:16s} gaps_filled={note['missing_before_fill']} "
              f"remaining={note['missing_after_fill']} clipped={note['negatives_clipped']}")

    scales = compute_scales(frame)
    print("\nramp normalizer S_m (development months):")
    for market in sorted(scales):
        print(f"  {market:16s} {scales[market]:10,.0f} MW")

    if args.write:
        months, _ = write_panels(frame, scales)
        print(f"\nwrote {months} monthly panels to {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
