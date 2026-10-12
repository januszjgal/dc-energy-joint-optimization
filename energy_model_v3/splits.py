"""Development months, cross-validation folds, the held-out test month, and fleets.

The twelve months of 2025 are the development data. Four-fold cross-validation
holds out three interleaved months per fold, so every fold trains on nine months
that span all four seasons and every 2025 month is validated exactly once. The
final policies train on all twelve months. May 2026, served by a second fleet
built from cells e-h, is the test month; it is scored once, at the end.

Each split specification ("spec") names its training, validation, and test
months. Everything fitted to data -- forecast weights, the market scales S_m,
the observation standardization, and the objective references C_R and C_Q --
is fitted on the spec's training months only.

Months follow the clock of the Google trace (UTC-7, see ``env/ramp_v6/clock.py``):
"2025-07" runs from 07:00 UTC on 1 July to 07:00 UTC on 1 August.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from env.ramp_v6.clock import month_labels  # noqa: F401  (re-exported)


DEVELOPMENT_MONTHS = tuple(f"2025-{month:02d}" for month in range(1, 13))
CV_FOLDS = {
    "k1": ("2025-01", "2025-05", "2025-09"),
    "k2": ("2025-02", "2025-06", "2025-10"),
    "k3": ("2025-03", "2025-07", "2025-11"),
    "k4": ("2025-04", "2025-08", "2025-12"),
}
TEST_MONTHS = ("2026-05",)
FINAL_SPEC = "final"
CV_SPECS = tuple(f"cv-{fold}" for fold in CV_FOLDS)
SPECS = CV_SPECS + (FINAL_SPEC,)

# Market order is fixed everywhere. The training fleet comes from cells a-d and
# the test fleet from cells e-h, matched to the same markets in the same order.
TRAIN_FLEET = (
    ("CAISO", "a"),
    ("MISO", "b"),
    ("SPP", "c"),
    ("ISONE", "d"),
)
TEST_FLEET = (
    ("CAISO", "e"),
    ("MISO", "f"),
    ("SPP", "g"),
    ("ISONE", "h"),
)
# Site sizes are measured against cell c, the largest training cell: cell c's
# site has compute capacity 1 and 500 MW. Test cells may be larger than 1.
REFERENCE_CELL = "c"

_folds_flat = [month for months in CV_FOLDS.values() for month in months]
if sorted(_folds_flat) != list(DEVELOPMENT_MONTHS) or len(set(_folds_flat)) != 12:
    raise RuntimeError("cross-validation folds must partition the development months")


def spec_months(spec: str) -> dict[str, tuple[str, ...]]:
    """Training, validation, and test months of one split specification."""
    if spec == FINAL_SPEC:
        return {"train": DEVELOPMENT_MONTHS, "validation": (), "test": TEST_MONTHS}
    if spec in CV_SPECS:
        held_out = CV_FOLDS[spec.removeprefix("cv-")]
        train = tuple(month for month in DEVELOPMENT_MONTHS if month not in held_out)
        return {"train": train, "validation": held_out, "test": ()}
    raise ValueError(f"unknown split specification {spec!r}; expected one of {SPECS}")


def fleet_for_split(split: str) -> tuple[tuple[str, str], ...]:
    """The test month is served by the e-h fleet; every 2025 month by a-d."""
    if split == "test":
        return TEST_FLEET
    if split in {"train", "validation"}:
        return TRAIN_FLEET
    raise ValueError(f"unknown split {split!r}")


def frozen_stats(frame: pd.DataFrame, months: tuple[str, ...], stats_id: str) -> dict[str, Any]:
    """Market scales and observation standardizers fitted on ``months`` only.

    ``frame`` is long-form with timestamp_utc, market_id, gross_demand_mw, and
    net_load_mw. S_m is the 95th percentile of gross demand; the level means and
    standard deviations standardize the observation's grid features.
    """
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True))
    inside = month_labels(stamps).isin(months)
    training = frame.loc[inside].drop_duplicates(subset=["timestamp_utc", "market_id"])
    present = set(month_labels(pd.DatetimeIndex(pd.to_datetime(training["timestamp_utc"], utc=True))))
    if present != set(months):
        raise ValueError("frozen statistics need every requested training month")
    grouped = training.groupby("market_id")
    return {
        "fit_start_utc": pd.to_datetime(training["timestamp_utc"], utc=True).min().isoformat(),
        "fit_end_utc": pd.to_datetime(training["timestamp_utc"], utc=True).max().isoformat(),
        "fit_months": list(months),
        "gross_q95_mw": {k: float(v) for k, v in grouped["gross_demand_mw"].quantile(0.95).items()},
        "gross_level_mean_mw": {k: float(v) for k, v in grouped["gross_demand_mw"].mean().items()},
        "gross_level_std_mw": {k: float(v) for k, v in grouped["gross_demand_mw"].std().items()},
        "net_level_mean_mw": {k: float(v) for k, v in grouped["net_load_mw"].mean().items()},
        "net_level_std_mw": {k: float(v) for k, v in grouped["net_load_mw"].std().items()},
        "stats_id": stats_id,
    }
