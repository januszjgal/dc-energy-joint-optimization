"""Fetch regional grid net demand for each DC location.

Net demand = total system load minus utility-scale renewable generation
(solar + wind). This is the "duck curve" quantity that hyperscale DC load
contributes to.

Produces one CSV per region with columns: timestep, net_demand_normalized
where net_demand_normalized ∈ [0, 1] is divided by the region's historical
peak over the window.

Data sources:
    - US balancing authorities: EIA Open Data API v2 (EIA-930 hourly data)
        - Demand: electricity/rto/region-data with type=D
        - Renewables: electricity/rto/fuel-type-data with fueltype in {SUN, WND}
    - ENTSO-E NL: ENTSO-E Transparency Platform (optional --entsoe-key)
    - Singapore: synthesized (EMA does not publish renewable breakdown; low
      RES share means net demand ≈ total demand)

Falls back to a calibrated synthetic model (solar-inverse + diurnal load
template) if real data is unavailable.

Window: May 1-31, 2019 — matches the existing price/solar CSVs.

Usage:
    python preprocess/net_demand_fetcher.py --eia-key YOUR_KEY
    python preprocess/net_demand_fetcher.py --mode synthetic
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "net_demand"
SOLAR_DIR = Path(__file__).resolve().parent.parent / "data" / "solar"

# Window matches the existing price / solar / workload CSVs
START_DATE = "2019-05-01T00"
END_DATE = "2019-05-31T23"
MAY_DAYS = 31
STEPS_PER_DAY = 288
TOTAL_STEPS = MAY_DAYS * STEPS_PER_DAY

# Region definitions: each maps to an EIA-930 balancing authority + (for
# synthetic fallback) the solar CSV used as the renewable-supply proxy and
# a peak-load magnitude used for normalization.
REGIONS = {
    "caiso": {
        "label": "CAISO (US-West)",
        "eia_ba": "CISO",
        "solar_csv": "the_dalles_or.csv",
        "timezone_offset_h": -8,
        "synth_peak_load": 40000.0,  # MW, approx CAISO May 2019 peak
        "synth_solar_share": 0.18,   # fraction of peak that solar supplies midday
        "synth_wind_share": 0.06,
    },
    "miso": {
        "label": "MISO (US-Central)",
        "eia_ba": "MISO",
        "solar_csv": "council_bluffs_ia.csv",
        "timezone_offset_h": -6,
        "synth_peak_load": 95000.0,
        "synth_solar_share": 0.03,
        "synth_wind_share": 0.14,
    },
    "southern_co": {
        "label": "Southern Co (US-Southeast-1)",
        "eia_ba": "SOCO",
        "solar_csv": "douglas_county_ga.csv",
        "timezone_offset_h": -5,
        "synth_peak_load": 32000.0,
        "synth_solar_share": 0.04,
        "synth_wind_share": 0.0,
    },
    "duke_carolinas": {
        "label": "Duke Carolinas (US-Southeast-2)",
        "eia_ba": "DUK",
        "solar_csv": "berkeley_county_sc.csv",
        "timezone_offset_h": -5,
        "synth_peak_load": 19000.0,
        "synth_solar_share": 0.05,
        "synth_wind_share": 0.0,
    },
    "entso_e_nl": {
        "label": "ENTSO-E Netherlands",
        "eia_ba": None,
        "solar_csv": "eemshaven_nl.csv",
        "timezone_offset_h": 1,
        "synth_peak_load": 19000.0,
        "synth_solar_share": 0.10,
        "synth_wind_share": 0.15,
    },
    "ema_singapore": {
        "label": "EMA Singapore",
        "eia_ba": None,
        "solar_csv": "singapore.csv",
        "timezone_offset_h": 8,
        "synth_peak_load": 7500.0,
        "synth_solar_share": 0.02,
        "synth_wind_share": 0.0,
    },
}


# ----------------------------------------------------------------------
# EIA fetch
# ----------------------------------------------------------------------


def _eia_get(api_key: str, route: str, params: dict) -> list[dict]:
    """Page through an EIA v2 endpoint and return all data records."""
    base = f"https://api.eia.gov/v2/electricity/rto/{route}/data/"
    params = {**params, "api_key": api_key, "length": 5000, "offset": 0}
    all_records: list[dict] = []
    while True:
        resp = requests.get(base, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json().get("response", {}).get("data", [])
        if not data:
            break
        all_records.extend(data)
        if len(data) < params["length"]:
            break
        params["offset"] += params["length"]
    return all_records


def fetch_eia_demand(api_key: str, ba: str) -> pd.Series | None:
    """Hourly demand (MW) for a BA over the May 2019 window."""
    print(f"  Fetching demand for {ba}...")
    records = _eia_get(
        api_key,
        "region-data",
        {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": ba,
            "facets[type][]": "D",
            "start": START_DATE,
            "end": END_DATE,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        },
    )
    if not records:
        print(f"  No demand data for {ba}")
        return None
    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["period"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("datetime")["value"].dropna().sort_index()
    s = s[~s.index.duplicated(keep="first")]
    print(f"  Got {len(s)} hourly demand records")
    return s


def fetch_eia_fuel(api_key: str, ba: str, fuel: str) -> pd.Series | None:
    """Hourly generation (MW) for a fuel type within a BA."""
    print(f"  Fetching {fuel} generation for {ba}...")
    records = _eia_get(
        api_key,
        "fuel-type-data",
        {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": ba,
            "facets[fueltype][]": fuel,
            "start": START_DATE,
            "end": END_DATE,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        },
    )
    if not records:
        return None
    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["period"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("datetime")["value"].dropna().sort_index()
    s = s[~s.index.duplicated(keep="first")]
    print(f"  Got {len(s)} hourly {fuel} records")
    return s


def fetch_eia_net_demand(api_key: str, ba: str) -> pd.Series | None:
    """Compute hourly net demand = D - SUN - WND for a BA."""
    demand = fetch_eia_demand(api_key, ba)
    if demand is None or len(demand) < 100:
        return None
    sun = fetch_eia_fuel(api_key, ba, "SUN")
    wnd = fetch_eia_fuel(api_key, ba, "WND")
    net = demand.copy()
    if sun is not None:
        net = net.subtract(sun.reindex(net.index, fill_value=0.0), fill_value=0.0)
    if wnd is not None:
        net = net.subtract(wnd.reindex(net.index, fill_value=0.0), fill_value=0.0)
    net = net.clip(lower=0.0)
    return net


# ----------------------------------------------------------------------
# Synthetic fallback
# ----------------------------------------------------------------------


def synthesize_net_demand(region: dict) -> pd.DataFrame:
    """Compose net demand from a diurnal residential load template minus
    a solar-fraction-driven renewable supply.

    Shape:
      net_demand[t] ~= peak_load * load_shape(local_hour, dow)
                       - peak_load * solar_share * solar_fraction[t]
                       - peak_load * wind_share * wind_shape(t)

    Returns a 5-min resolution dataframe with timestep and net_demand_MW.
    The resulting series has the duck-curve shape: midday trough (solar
    sucks supply out) + evening ramp (load rises while solar disappears).
    """
    rng = np.random.default_rng(0)
    t = np.arange(TOTAL_STEPS)

    offset_steps = region["timezone_offset_h"] * 12
    local_step = (t + offset_steps) % STEPS_PER_DAY
    local_hour = local_step / 12.0

    # Residential + commercial diurnal load shape (peaks at ~19:00 local,
    # trough at ~04:00). Calibrated to give a ~0.5..1.0 normalized range.
    morning = np.exp(-0.5 * ((local_hour - 8) / 2.5) ** 2)
    evening = np.exp(-0.5 * ((local_hour - 19) / 3.0) ** 2)
    base = 0.55 + 0.15 * morning + 0.30 * evening
    load_shape = np.clip(base, 0.4, 1.0)

    # Weekend discount (~10%) - May 1 2019 was Wednesday
    dow = (t // STEPS_PER_DAY) % 7
    is_weekend = np.isin((dow + 2) % 7, [5, 6])
    load_shape = load_shape * np.where(is_weekend, 0.9, 1.0)

    load_mw = region["synth_peak_load"] * load_shape

    # Solar supply (read from the NSRDB CSV for this region if available)
    solar_path = SOLAR_DIR / region["solar_csv"]
    if solar_path.exists():
        solar_df = pd.read_csv(solar_path)
        solar_fraction = solar_df["solar_fraction"].to_numpy(dtype=np.float64)
        # Match length
        if len(solar_fraction) >= TOTAL_STEPS:
            solar_fraction = solar_fraction[:TOTAL_STEPS]
        else:
            solar_fraction = np.pad(
                solar_fraction,
                (0, TOTAL_STEPS - len(solar_fraction)),
                mode="edge",
            )
    else:
        solar_fraction = np.zeros(TOTAL_STEPS)

    solar_supply_mw = region["synth_peak_load"] * region["synth_solar_share"] * solar_fraction

    # Wind: pseudo-stochastic with persistence (AR(1)-ish)
    wind_share = region["synth_wind_share"]
    if wind_share > 0:
        noise = rng.normal(0, 1, size=TOTAL_STEPS)
        # Simple persistent stochastic process
        wind_shape = np.zeros(TOTAL_STEPS)
        wind_shape[0] = 0.5
        for k in range(1, TOTAL_STEPS):
            wind_shape[k] = 0.995 * wind_shape[k - 1] + 0.005 * (0.5 + 0.3 * noise[k])
        wind_shape = np.clip(wind_shape, 0.0, 1.0)
    else:
        wind_shape = np.zeros(TOTAL_STEPS)

    wind_supply_mw = region["synth_peak_load"] * wind_share * wind_shape

    net_mw = np.maximum(load_mw - solar_supply_mw - wind_supply_mw, 0.0)

    return pd.DataFrame({"timestep": t, "net_demand_mw": net_mw})


# ----------------------------------------------------------------------
# Interpolation + normalization
# ----------------------------------------------------------------------


def hourly_to_5min(hourly: pd.Series) -> np.ndarray:
    """Resample an hourly UTC Series to 5-min and truncate to May 2019."""
    df = hourly.to_frame("net_demand_mw")
    df = df.resample("5min").interpolate(method="linear")
    df = df.dropna()
    df = df[df.index.month == 5]
    df = df.iloc[:TOTAL_STEPS]
    return df["net_demand_mw"].to_numpy(dtype=np.float64)


def normalize(net_mw: np.ndarray, peak_floor: float = 1.0) -> np.ndarray:
    """Divide by historical peak to get a [0, 1] series.

    peak_floor guards against degenerate cases. We use the 99.5th percentile
    (not the max) to avoid one outlier hour pinning the normalization.
    """
    peak = max(float(np.quantile(net_mw, 0.995)), peak_floor)
    return np.clip(net_mw / peak, 0.0, 1.0)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def _load_env_key() -> str | None:
    """Try to read EIA_API_KEY from a .env file or environment."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("EIA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("EIA_API_KEY")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch regional net demand")
    parser.add_argument("--eia-key", default=None, help="EIA API key")
    parser.add_argument(
        "--mode",
        choices=["auto", "synthetic"],
        default="auto",
        help="auto: try real then fall back; synthetic: always use model",
    )
    args = parser.parse_args(argv)

    eia_key = args.eia_key or _load_env_key()
    if args.mode == "auto" and not eia_key:
        print("WARNING: No EIA key provided; using synthetic for US BAs too.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for region_id, region in REGIONS.items():
        print(f"\nProcessing {region['label']}...")
        net_mw = None
        source = "synthetic"

        if args.mode == "auto" and eia_key and region["eia_ba"] is not None:
            try:
                series = fetch_eia_net_demand(eia_key, region["eia_ba"])
                if series is not None and len(series) > 100:
                    net_mw = hourly_to_5min(series)
                    source = "EIA-930"
            except Exception as e:
                print(f"  EIA fetch failed: {e}")

        if net_mw is None:
            df_synth = synthesize_net_demand(region)
            net_mw = df_synth["net_demand_mw"].to_numpy()

        # Pad / trim to TOTAL_STEPS exactly
        if len(net_mw) < TOTAL_STEPS:
            net_mw = np.pad(net_mw, (0, TOTAL_STEPS - len(net_mw)), mode="edge")
        else:
            net_mw = net_mw[:TOTAL_STEPS]

        net_norm = normalize(net_mw)

        out_path = DATA_DIR / f"{region_id}.csv"
        pd.DataFrame(
            {
                "timestep": np.arange(TOTAL_STEPS),
                "net_demand_normalized": net_norm.astype(np.float32),
                "net_demand_mw": net_mw.astype(np.float32),
            }
        ).to_csv(out_path, index=False)
        print(
            f"  Saved to {out_path}  source={source}  "
            f"mean={net_norm.mean():.3f}  peak={net_mw.max():.0f} MW"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
