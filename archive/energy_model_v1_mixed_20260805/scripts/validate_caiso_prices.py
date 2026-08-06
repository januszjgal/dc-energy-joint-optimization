"""Real-CAISO price provenance check (peer-review M7, non-destructive).

The thesis price series are calibrated-synthetic (regional May-2019 average x
diurnal pattern + noise) because hourly wholesale LMP is not uniformly
available — two US scenario regions (Southern Co./GA, Duke/SC) are vertically
integrated utilities with no public market price, and the EIA API exposes no
hourly price route. This script does NOT change the training environment; it
fetches REAL CAISO day-ahead LMP (the one US region with a free public hourly
market, via CAISO OASIS) for May 2019 and compares it to our synthetic CAISO
series, so the paper can state how well the synthetic profile tracks reality.

Writes output/review_campaign/caiso_price_validation.json
"""

from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OASIS = "http://oasis.caiso.com/oasisapi/SingleZip"
NODE = "TH_NP15_GEN-APND"  # NP15 trading hub (Northern California)
NS = {"m": "http://www.caiso.com/soa/OASISReport_v1.xsd"}


def fetch_caiso_dam_lmp(year: int, max_retries: int = 6) -> pd.Series | None:
    """Real CAISO day-ahead total LMP ($/MWh), hourly, for May of `year`.

    Returns a Series indexed by UTC hour, or None on failure. Honors OASIS
    rate limits with exponential backoff.
    """
    params = {
        "queryname": "PRC_LMP", "version": "1", "market_run_id": "DAM",
        "startdatetime": f"{year}0501T07:00-0000",
        "enddatetime": f"{year}0601T07:00-0000",  # 31-day window (OASIS max)
        "node": NODE, "resultformat": "6",
    }
    delay = 5.0
    for attempt in range(max_retries):
        try:
            r = requests.get(OASIS, params=params, timeout=120)
            if r.status_code == 429 or (b"Acceptable Use Policy" in r.content):
                print(f"  {year}: rate-limited, backing off {delay:.0f}s "
                      f"(attempt {attempt+1})")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.content[:2] != b"PK":
                print(f"  {year}: non-zip response ({r.status_code}): {r.text[:120]}")
                return None
            z = zipfile.ZipFile(io.BytesIO(r.content))
            root = ET.fromstring(z.read(z.namelist()[0]))
            recs = []
            for item in root.iterfind(".//m:REPORT_DATA", NS):
                fields = {c.tag.split("}")[-1]: c.text for c in item}
                if fields.get("LMP_TYPE") != "LMP":  # total LMP only
                    continue
                recs.append((fields.get("INTERVAL_START_GMT"),
                             float(fields.get("VALUE"))))
            if not recs:
                print(f"  {year}: zip parsed but no LMP records")
                return None
            s = pd.Series(dict(recs)).sort_index()
            s.index = pd.to_datetime(s.index, utc=True)
            print(f"  {year}: {len(s)} real DAM LMP hours, "
                  f"mean=${s.mean():.2f}/MWh")
            return s
        except Exception as e:
            print(f"  {year}: error {type(e).__name__}: {e}; retry in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


def main() -> None:
    out: dict = {"node": NODE, "market": "DAM", "years": {}}
    real = {y: fetch_caiso_dam_lmp(y) for y in (2019, 2024)}

    synth = pd.read_csv(ROOT / "data" / "prices" / "caiso.csv")
    # synthetic is 5-min $/kWh; collapse to hourly $/MWh
    synth_hourly = (synth["price_usd_kwh"].to_numpy().reshape(-1, 12).mean(axis=1)
                    * 1000.0)

    for year, s in real.items():
        if s is None:
            out["years"][str(year)] = {"status": "fetch_failed"}
            continue
        real_h = s.to_numpy()
        n = min(len(real_h), len(synth_hourly))
        rh, sh = real_h[:n], synth_hourly[:n]
        # diurnal shape: average over hour-of-day
        hod = np.arange(n) % 24
        real_diurnal = np.array([rh[hod == h].mean() for h in range(24)])
        synth_diurnal = np.array([sh[hod == h].mean() for h in range(24)])
        out["years"][str(year)] = {
            "status": "ok",
            "n_hours": int(n),
            "real_mean_usd_mwh": float(rh.mean()),
            "synth_mean_usd_mwh": float(sh.mean()),
            "real_std_usd_mwh": float(rh.std()),
            "synth_std_usd_mwh": float(sh.std()),
            "hourly_correlation": float(np.corrcoef(rh, sh)[0, 1]),
            "diurnal_shape_correlation": float(
                np.corrcoef(real_diurnal, synth_diurnal)[0, 1]),
            "real_peak_to_trough_ratio": float(real_diurnal.max() / real_diurnal.min()),
            "synth_peak_to_trough_ratio": float(synth_diurnal.max() / synth_diurnal.min()),
        }
        v = out["years"][str(year)]
        print(f"\n{year}: real ${v['real_mean_usd_mwh']:.2f} vs synth "
              f"${v['synth_mean_usd_mwh']:.2f}/MWh | hourly r={v['hourly_correlation']:.2f} "
              f"| diurnal-shape r={v['diurnal_shape_correlation']:.2f}")

    path = ROOT / "output" / "review_campaign" / "caiso_price_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
