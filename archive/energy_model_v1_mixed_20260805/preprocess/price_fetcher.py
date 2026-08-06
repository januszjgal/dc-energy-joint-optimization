"""Fetch wholesale electricity prices for each DC location.

Produces one CSV per market with columns: timestep, price_usd_kwh
Timestep is a 5-minute interval index matching the workload extractor.

Data sources:
    - US markets: EIA Open Data API (hourly wholesale prices)
    - ENTSO-E (Netherlands): ENTSO-E Transparency Platform
    - Singapore: Energy Market Company (USEP)

For markets where API access is difficult, a calibrated synthetic model
is provided as fallback.

Usage:
    python preprocess/price_fetcher.py --eia-key YOUR_EIA_API_KEY
    python preprocess/price_fetcher.py --mode synthetic  # use model instead
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "prices"

# Market definitions with calibration parameters
MARKETS = {
    "caiso": {
        "label": "CAISO (Oregon/California)",
        "base_price_usd_mwh": 45.0,  # May 2019 average ~$45/MWh
        "peak_multiplier": 1.4,
        "solar_discount": 0.3,
        "timezone_offset_h": -8,  # UTC-8
        "eia_region": "CAL",
    },
    "miso": {
        "label": "MISO (Iowa)",
        "base_price_usd_mwh": 28.0,  # May 2019 average ~$28/MWh
        "peak_multiplier": 1.3,
        "solar_discount": 0.15,
        "timezone_offset_h": -6,  # UTC-6
        "eia_region": "MIDW",
    },
    "southern_co": {
        "label": "Southern Company (Georgia)",
        "base_price_usd_mwh": 35.0,  # May 2019 average ~$35/MWh
        "peak_multiplier": 1.35,
        "solar_discount": 0.2,
        "timezone_offset_h": -5,  # UTC-5
        "eia_region": "SE",
    },
    "duke_carolinas": {
        "label": "Duke Energy Carolinas (SC)",
        "base_price_usd_mwh": 32.0,  # May 2019 average ~$32/MWh
        "peak_multiplier": 1.3,
        "solar_discount": 0.2,
        "timezone_offset_h": -5,  # UTC-5
        "eia_region": "CAR",
    },
    "entso_e_nl": {
        "label": "ENTSO-E Netherlands",
        "base_price_usd_mwh": 42.0,  # May 2019 average ~€38/MWh ≈ $42/MWh
        "peak_multiplier": 1.5,
        "solar_discount": 0.35,
        "timezone_offset_h": 1,  # UTC+1
        "eia_region": None,
    },
    "ema_singapore": {
        "label": "EMA Singapore (USEP)",
        "base_price_usd_mwh": 85.0,  # May 2019 USEP average ~S$115/MWh ≈ $85/MWh
        "peak_multiplier": 1.6,
        "solar_discount": 0.1,
        "timezone_offset_h": 8,  # UTC+8
        "eia_region": None,
    },
}

# May 2019: 31 days, 288 five-minute intervals per day
MAY_DAYS = 31
STEPS_PER_DAY = 288
TOTAL_STEPS = MAY_DAYS * STEPS_PER_DAY


def _load_env_key() -> str | None:
    """Read EIA_API_KEY from the repo-root .env file or environment.

    Mirrors net_demand_fetcher._load_env_key so both fetchers resolve the key
    the same way (the key lives in the gitignored .env; never pass it on the
    command line in shared shells).
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("EIA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("EIA_API_KEY")


def fetch_eia_prices(
    api_key: str, region: str, year: int = 2019
) -> pd.DataFrame | None:
    """Fetch hourly wholesale electricity prices from EIA API v2.

    Returns DataFrame with columns: datetime (UTC), price_usd_mwh
    or None if the fetch fails. ``year`` selects the May window (peer-review
    M2b shifts this to test market-condition generalization).
    """
    url = "https://api.eia.gov/v2/electricity/rto/wholesale-prices/data/"
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": region,
        "start": f"{year}-05-01T00",
        "end": f"{year}-05-31T23",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 750,  # 31 days * 24 hours = 744
    }

    try:
        print(f"  Fetching EIA data for region {region}...")
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        records = []
        for entry in data.get("response", {}).get("data", []):
            dt = pd.to_datetime(entry["period"])
            price = float(entry["value"]) if entry["value"] is not None else np.nan
            records.append({"datetime": dt, "price_usd_mwh": price})

        if not records:
            print(f"  Warning: No EIA data returned for {region}")
            return None

        df = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)
        df = df.dropna(subset=["price_usd_mwh"])
        print(f"  Got {len(df)} hourly price records")
        return df

    except Exception as e:
        print(f"  EIA fetch failed for {region}: {e}")
        return None


def generate_synthetic_prices(market: dict, seed: int = 42) -> pd.DataFrame:
    """Generate calibrated synthetic electricity prices for May 2019.

    Uses a diurnal pattern with noise, calibrated to the market's
    base price and characteristics.
    """
    rng = np.random.default_rng(seed)
    timesteps = np.arange(TOTAL_STEPS)

    # Local hour of day (accounting for timezone offset)
    offset_steps = market["timezone_offset_h"] * 12  # 12 five-min steps per hour
    local_step = (timesteps + offset_steps) % STEPS_PER_DAY
    local_hour = local_step / 12.0  # fractional hour [0, 24)

    base = market["base_price_usd_mwh"]

    # Diurnal demand pattern (peaks at ~18:00 local, trough at ~04:00 local)
    demand_shape = 0.5 * (1 + np.sin(2 * np.pi * (local_hour - 6) / 24.0))
    peak_factor = market["peak_multiplier"] - 1.0

    # Solar depression (prices drop during solar hours ~10:00-16:00)
    solar_hours = np.exp(-0.5 * ((local_hour - 13) / 3.0) ** 2)
    solar_effect = market["solar_discount"] * solar_hours

    # Weekend effect (lower prices on weekends)
    day_of_week = (timesteps // STEPS_PER_DAY) % 7
    # May 1, 2019 was a Wednesday (day_of_week index 0 = Wed)
    is_weekend = np.isin((day_of_week + 2) % 7, [5, 6])  # Sat=5, Sun=6
    weekend_discount = np.where(is_weekend, 0.15, 0.0)

    # Random noise (~5% of base)
    noise = rng.normal(0, 0.05 * base, size=TOTAL_STEPS)

    price_usd_mwh = base * (
        1.0
        + peak_factor * demand_shape
        - solar_effect
        - weekend_discount
    ) + noise

    # Ensure no negative prices (rare but possible in real markets)
    price_usd_mwh = np.maximum(price_usd_mwh, 0.0)

    # Convert $/MWh to $/kWh
    price_usd_kwh = price_usd_mwh / 1000.0

    return pd.DataFrame({
        "timestep": timesteps,
        "price_usd_kwh": price_usd_kwh,
    })


def interpolate_to_5min(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate hourly prices to 5-minute resolution."""
    hourly_df = hourly_df.set_index("datetime")
    df_5min = hourly_df.resample("5min").interpolate(method="linear")
    df_5min = df_5min.dropna()

    # Filter to May only
    df_5min = df_5min[df_5min.index.month == 5]

    df_5min = df_5min.reset_index()
    df_5min["timestep"] = range(len(df_5min))

    # Convert $/MWh to $/kWh
    df_5min["price_usd_kwh"] = df_5min["price_usd_mwh"] / 1000.0
    return df_5min[["timestep", "price_usd_kwh"]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch wholesale electricity prices for DC locations"
    )
    parser.add_argument(
        "--eia-key",
        default=None,
        help="EIA API key for US market data (get from eia.gov)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "synthetic"],
        default="auto",
        help="'auto' tries real data first, falls back to synthetic. "
        "'synthetic' uses calibrated model only.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2019,
        help="May-window year (default 2019). Non-2019 writes to "
             "data/prices_<year>/ for the M2b market-shift test, leaving the "
             "2019 series the campaign trained on untouched.",
    )
    args = parser.parse_args(argv)

    eia_key = args.eia_key or _load_env_key()
    out_dir = DATA_DIR if args.year == 2019 else (
        DATA_DIR.parent / f"prices_{args.year}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "auto" and not eia_key:
        print("No EIA key (--eia-key or .env EIA_API_KEY); using synthetic.")

    for market_id, market in MARKETS.items():
        print(f"\nProcessing {market['label']}...")

        df = None

        # Try real data first (US markets via EIA)
        if (
            args.mode == "auto"
            and eia_key
            and market["eia_region"] is not None
        ):
            eia_df = fetch_eia_prices(eia_key, market["eia_region"], args.year)
            if eia_df is not None and len(eia_df) > 100:
                df = interpolate_to_5min(eia_df)
                print(f"  Using real EIA data ({len(df)} rows)")

        # Fallback to synthetic
        if df is None:
            df = generate_synthetic_prices(market)
            print(f"  Using calibrated synthetic prices ({len(df)} rows)")

        out_path = out_dir / f"{market_id}.csv"
        df.to_csv(out_path, index=False)
        print(
            f"  Saved to {out_path} "
            f"(mean=${df['price_usd_kwh'].mean():.5f}/kWh, "
            f"std=${df['price_usd_kwh'].std():.5f}/kWh)"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
