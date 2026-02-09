"""Fetch solar irradiance data from PVGIS for each DC location.

Produces one CSV per location with columns: timestep, solar_fraction
where solar_fraction is GHI normalized to [0, 1] (peak = 1.0).

The timestep index matches the 5-minute intervals used by the Google
workload extractor (288 steps per day, ~8928 for 31 days of May 2019).

Usage:
    python preprocess/solar_fetcher.py
    python preprocess/solar_fetcher.py --locations us  # US locations only
    python preprocess/solar_fetcher.py --locations global  # global locations only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "solar"

# Real Google DC locations
LOCATIONS = {
    # US locations
    "the_dalles_or": {"lat": 45.6, "lon": -121.2, "label": "The Dalles, OR", "group": "us"},
    "council_bluffs_ia": {"lat": 41.3, "lon": -95.9, "label": "Council Bluffs, IA", "group": "us"},
    "douglas_county_ga": {"lat": 33.7, "lon": -84.7, "label": "Douglas County, GA", "group": "us"},
    "berkeley_county_sc": {"lat": 33.2, "lon": -80.0, "label": "Berkeley County, SC", "group": "us"},
    # Global locations
    "eemshaven_nl": {"lat": 53.4, "lon": 6.8, "label": "Eemshaven, NL", "group": "global"},
    "singapore": {"lat": 1.3, "lon": 103.8, "label": "Singapore", "group": "global"},
}

# PVGIS API endpoint (v5.2)
PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"


def fetch_pvgis_hourly(lat: float, lon: float, year: int = 2019) -> pd.DataFrame:
    """Fetch hourly Global Horizontal Irradiance from PVGIS.

    Returns DataFrame with columns: datetime (UTC), ghi (W/m^2).
    Uses ERA5 radiation database (global coverage) since PVGIS-SARAH
    only covers Europe/Africa.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": year,
        "endyear": year,
        "pvcalculation": 0,
        "components": 1,
        "outputformat": "json",
        "raddatabase": "PVGIS-ERA5",
    }

    print(f"  Requesting PVGIS ERA5 data for ({lat}, {lon})...")
    resp = requests.get(PVGIS_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["outputs"]["hourly"]
    records = []
    for entry in hourly:
        # PVGIS returns time as "YYYYMMDD:HHMM" in UTC
        time_str = entry["time"]
        dt = pd.to_datetime(time_str, format="%Y%m%d:%H%M")
        # Total irradiance = beam + diffuse + ground-reflected
        ghi = entry["Gb(i)"] + entry["Gd(i)"] + entry["Gr(i)"]
        records.append({"datetime": dt, "ghi": ghi})

    df = pd.DataFrame(records)
    return df


def filter_may_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to May 2019 and interpolate from hourly to 5-minute resolution."""
    # Filter to May
    df = df[df["datetime"].dt.month == 5].copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # Set datetime index for resampling
    df = df.set_index("datetime")

    # Resample to 5-minute intervals with linear interpolation
    df_5min = df.resample("5min").interpolate(method="linear")

    # Drop last partial interval if any
    df_5min = df_5min.dropna()

    return df_5min.reset_index()


def normalize_to_solar_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize GHI to [0, 1] solar capacity factor.

    Peak GHI (~1000 W/m^2 at clear noon) maps to 1.0.
    Night (0 W/m^2) maps to 0.0.
    """
    # Use 1000 W/m^2 as reference peak (standard test condition)
    PEAK_GHI = 1000.0
    df["solar_fraction"] = (df["ghi"] / PEAK_GHI).clip(0.0, 1.0)
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch solar irradiance data from PVGIS"
    )
    parser.add_argument(
        "--locations",
        choices=["us", "global", "all"],
        default="all",
        help="Which location group to fetch (default: all)",
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, loc in LOCATIONS.items():
        if args.locations != "all" and loc["group"] != args.locations:
            # For "us" mode, also skip global-only locations;
            # for "global" mode, include US locations that are shared
            # (Oregon, Iowa are in both models)
            if args.locations == "global" and loc["group"] == "us":
                if name not in ("the_dalles_or", "council_bluffs_ia"):
                    continue
            elif args.locations == "us" and loc["group"] == "global":
                continue

        print(f"Fetching solar data for {loc['label']} ({name})...")
        hourly_df = fetch_pvgis_hourly(loc["lat"], loc["lon"])

        print(f"  Filtering to May 2019 and interpolating to 5-min...")
        may_df = filter_may_and_interpolate(hourly_df)
        may_df = normalize_to_solar_fraction(may_df)

        # Create timestep index (0-based, matching workload extractor)
        may_df["timestep"] = range(len(may_df))
        out_df = may_df[["timestep", "solar_fraction"]].copy()

        out_path = DATA_DIR / f"{name}.csv"
        out_df.to_csv(out_path, index=False)
        print(
            f"  Saved {len(out_df)} rows to {out_path} "
            f"(mean={out_df['solar_fraction'].mean():.4f})"
        )

    print("Done.")


if __name__ == "__main__":
    main()
