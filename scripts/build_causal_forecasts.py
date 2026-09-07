"""Regenerate strictly causal gross-demand and net-load forecasts.

The forecasts shipped in the original panels have two defects. All three
horizon columns hold one series offset by one row, so a three-hour-ahead
forecast is identical to a one-hour-ahead forecast of the same target hour;
and the series is 1.9x to 3.1x worse than naive persistence at one hour,
which means it never sees the most recent observation.

This module fits one ridge regression per (market, quantity, horizon), using
only features observable at the issue hour, on the training calendar alone.
Ridge is solved in closed form so the module needs nothing beyond numpy.

Run with --report to print an accuracy comparison without touching any file.
Run with --write to rewrite the forecast columns of every monthly panel.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PANEL_ROOT = ROOT / "data" / "four_market_2025" / "months"
CALENDAR_PATH = ROOT / "data" / "four_market_2025" / "calendar.json"

HORIZONS = (1, 2, 3)
QUANTITIES = ("gross", "net")
QUANTITY_COLUMN = {"gross": "gross_demand_mw", "net": "net_load_mw"}

# Fit period. Matches the training calendar exactly, so that no validation
# hour influences any coefficient. The holdout month sits in the middle of the
# year, so the training set is not contiguous: models must not be fitted on
# the months either side of the holdout and then scored on it.
YEAR_START = pd.Timestamp("2025-01-01T00:00:00Z")
YEAR_END = pd.Timestamp("2025-12-31T23:00:00Z")
VALIDATION_MONTH = 5


def is_training(index: pd.DatetimeIndex) -> np.ndarray:
    """True for hours inside the simulated year but outside the holdout month."""
    inside = (index >= YEAR_START) & (index <= YEAR_END)
    return inside & (index.month != VALIDATION_MONTH)


def is_validation(index: pd.DatetimeIndex) -> np.ndarray:
    """True for hours in the holdout month."""
    inside = (index >= YEAR_START) & (index <= YEAR_END)
    return inside & (index.month == VALIDATION_MONTH)

# Ridge penalties searched by rolling-origin cross-validation.
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
CV_FOLDS = 4

VINTAGE_ID = "causal-ridge-shortlag-v2"


@dataclass(frozen=True)
class FittedModel:
    """One ridge model plus the standardization it was fitted under."""

    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    alpha: float
    n_train: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.feature_mean) / self.feature_std
        return standardized @ self.coefficients + self.intercept


def load_hourly_series() -> pd.DataFrame:
    """Assemble one continuous hourly frame per market from the monthly panels.

    Each monthly panel also carries the four warm hours that close the month
    before it, so a (timestamp, market) pair can appear twice. Duplicates are
    identical by construction; the first is kept.
    """
    frames = [pd.read_csv(path) for path in sorted(PANEL_ROOT.glob("*/canonical_panel.csv"))]
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp_utc"] = pd.to_datetime(panel["timestamp_utc"], utc=True)
    panel = panel.drop_duplicates(subset=["timestamp_utc", "market_id"])
    return panel.sort_values(["market_id", "timestamp_utc"]).reset_index(drop=True)


def build_features(market_frame: pd.DataFrame) -> pd.DataFrame:
    """Build the causal feature matrix for one market.

    Every column is a function of observations at or before the issue hour.
    Nothing here may reference a row later than the one it sits on; that
    property is what makes the resulting forecast admissible, and it is
    checked by ``assert_causal`` below.
    """
    frame = market_frame.set_index("timestamp_utc").asfreq("h")
    gross = frame["gross_demand_mw"]
    net = frame["net_load_mw"]

    features = pd.DataFrame(index=frame.index)

    # Short lags. Their absence is why the original forecasts lost to
    # persistence; lag 0 alone reproduces a persistence forecast.
    for lag in (0, 1, 2, 3):
        features[f"gross_lag{lag}"] = gross.shift(lag)
        features[f"net_lag{lag}"] = net.shift(lag)

    # Same hour yesterday and same hour last week.
    for lag in (24, 25, 168):
        features[f"gross_lag{lag}"] = gross.shift(lag)
        features[f"net_lag{lag}"] = net.shift(lag)

    # Trailing means, shifted so the current hour is excluded from its own mean.
    for window in (24, 168):
        features[f"gross_mean{window}"] = gross.shift(1).rolling(window).mean()
        features[f"net_mean{window}"] = net.shift(1).rolling(window).mean()

    # Recent movement. The objective is about ramps, so the rate the series is
    # already moving at is more informative than its level alone.
    features["gross_ramp1"] = gross.diff(1)
    features["net_ramp1"] = net.diff(1)
    features["gross_ramp3"] = gross.diff(3)
    features["net_ramp3"] = net.diff(3)

    # Renewable output is observable at the issue hour and drives the gap
    # between gross demand and net load.
    features["wind"] = frame["wind_mw"]
    features["solar"] = frame["solar_mw"]

    # Calendar position.
    hour = frame.index.hour.to_numpy()
    dow = frame.index.dayofweek.to_numpy()
    features["hod_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hod_cos"] = np.cos(2 * np.pi * hour / 24)
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return features


def assert_causal(features: pd.DataFrame, market_frame: pd.DataFrame) -> None:
    """Fail loudly if any feature column depends on a future observation.

    Shifting a series forward in time and rebuilding must leave every feature
    unchanged on the overlapping rows. A feature that peeked ahead would move.
    """
    truncated = market_frame.iloc[:-6].copy()
    rebuilt = build_features(truncated)
    common = features.index.intersection(rebuilt.index)
    lhs = features.loc[common]
    rhs = rebuilt.loc[common]
    mismatch = ~np.isclose(lhs.to_numpy(dtype=float), rhs.to_numpy(dtype=float),
                           equal_nan=True, rtol=0, atol=1e-9)
    if mismatch.any():
        bad = lhs.columns[mismatch.any(axis=0)].tolist()
        raise AssertionError(f"non-causal feature columns detected: {bad}")


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Closed-form ridge on standardized inputs, intercept left unpenalized."""
    y_mean = float(y.mean())
    centered = y - y_mean
    gram = x.T @ x + alpha * np.eye(x.shape[1])
    coefficients = np.linalg.solve(gram, x.T @ centered)
    return coefficients, y_mean


def select_alpha(x: np.ndarray, y: np.ndarray) -> float:
    """Rolling-origin cross-validation.

    Folds expand forward in time so a model is never scored on an hour that
    precedes hours it was fitted on.
    """
    n = len(y)
    if n < CV_FOLDS * 2:
        return 1.0
    edges = np.linspace(n // 2, n, CV_FOLDS + 1, dtype=int)
    scores: dict[float, list[float]] = {alpha: [] for alpha in ALPHA_GRID}
    for start, stop in zip(edges[:-1], edges[1:]):
        if start == stop:
            continue
        x_train, y_train = x[:start], y[:start]
        x_valid, y_valid = x[start:stop], y[start:stop]
        for alpha in ALPHA_GRID:
            coefficients, intercept = fit_ridge(x_train, y_train, alpha)
            error = x_valid @ coefficients + intercept - y_valid
            scores[alpha].append(float(np.mean(np.abs(error))))
    return min(ALPHA_GRID, key=lambda a: float(np.mean(scores[a])) if scores[a] else np.inf)


def fit_market_quantity_horizon(
    features: pd.DataFrame, target: pd.Series, horizon: int
) -> FittedModel:
    """Fit one direct-multistep model predicting ``horizon`` hours ahead.

    A separate model per horizon is what makes the three columns genuinely
    different. Recursive one-step forecasting, or copying one series into
    three columns, would not.
    """
    aligned_target = target.shift(-horizon)
    usable = features.notna().all(axis=1) & aligned_target.notna()
    in_window = is_training(features.index)
    mask = usable & in_window

    x_raw = features.loc[mask].to_numpy(dtype=float)
    y = aligned_target.loc[mask].to_numpy(dtype=float)
    if len(y) == 0:
        raise ValueError("no training rows survive the feature and calendar masks")

    mean = x_raw.mean(axis=0)
    std = x_raw.std(axis=0)
    std[std < 1e-12] = 1.0
    x = (x_raw - mean) / std

    alpha = select_alpha(x, y)
    coefficients, intercept = fit_ridge(x, y, alpha)
    return FittedModel(coefficients, intercept, mean, std, alpha, int(len(y)))


def persistence_baseline(target: pd.Series, horizon: int) -> pd.Series:
    """The forecast to beat: assume the target hour equals the issue hour."""
    return target.copy()


def build_all(panel: pd.DataFrame) -> tuple[dict[tuple[str, str, int], FittedModel], pd.DataFrame]:
    """Fit every model and return predictions indexed by (timestamp, market)."""
    models: dict[tuple[str, str, int], FittedModel] = {}
    predictions: list[pd.DataFrame] = []
    fallbacks: list[tuple[str, str, int, float, float]] = []

    for market, market_frame in panel.groupby("market_id", sort=True):
        features = build_features(market_frame)
        assert_causal(features, market_frame)
        frame = market_frame.set_index("timestamp_utc").asfreq("h")

        out = pd.DataFrame(index=frame.index)
        out["market_id"] = market

        for quantity in QUANTITIES:
            target = frame[QUANTITY_COLUMN[quantity]]
            for horizon in HORIZONS:
                model = fit_market_quantity_horizon(features, target, horizon)
                models[(market, quantity, horizon)] = model

                complete = features.notna().all(axis=1)
                column = f"forecast_{quantity}_h{horizon}_mw"
                out[column] = np.nan
                if complete.any():
                    values = model.predict(features.loc[complete].to_numpy(dtype=float))
                    out.loc[complete, column] = values
                # Rows without a full history (the first week of the panel)
                # fall back to persistence rather than being left empty.
                out[column] = out[column].fillna(target)

                # Invariant: a forecast must never be worse than assuming the
                # target hour equals the issue hour. Otherwise adding it to the
                # observation could actively mislead the scheduler. Any model
                # that fails in-sample is replaced by persistence outright.
                actual = target.shift(-horizon)
                # Judged on training hours only. Letting February decide which
                # models to keep would leak the validation period into the
                # forecasts, which is the one thing the split exists to prevent.
                scored = (
                    actual.notna()
                    & is_training(out.index)
                )
                model_mae = float((out[column][scored] - actual[scored]).abs().mean())
                persist_mae = float((target[scored] - actual[scored]).abs().mean())
                if not model_mae < persist_mae:
                    out[column] = target
                    fallbacks.append((market, quantity, horizon, model_mae, persist_mae))

        predictions.append(out.reset_index().rename(columns={"index": "timestamp_utc"}))

    for market, quantity, horizon, model_mae, persist_mae in fallbacks:
        print(f"  fell back to persistence: {market} {quantity} h{horizon} "
              f"(model {model_mae:.0f} MW vs persistence {persist_mae:.0f} MW)")

    return models, pd.concat(predictions, ignore_index=True)


def report(panel: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Print new-vs-old-vs-persistence accuracy, split by fit period."""
    merged = panel.merge(
        predictions, on=["timestamp_utc", "market_id"], suffixes=("_old", "_new")
    )
    print(f"{'market':16s}{'qty':6s}{'h':>3s}"
          f"{'new':>8s}{'old':>8s}{'persist':>9s}   "
          f"{'new(val)':>9s}{'persist(val)':>13s}")
    for market, group in merged.groupby("market_id", sort=True):
        group = group.sort_values("timestamp_utc").set_index("timestamp_utc")
        is_val = pd.Series(is_validation(group.index), index=group.index)
        for quantity in QUANTITIES:
            actual_series = group[QUANTITY_COLUMN[quantity]]
            for horizon in HORIZONS:
                actual = actual_series.shift(-horizon)
                new = group[f"forecast_{quantity}_h{horizon}_mw_new"]
                old = group[f"forecast_{quantity}_h{horizon}_mw_old"]
                valid = actual.notna()

                def mae(series: pd.Series, mask: pd.Series) -> float:
                    sel = valid & mask
                    return float((series[sel] - actual[sel]).abs().mean())

                everywhere = pd.Series(True, index=group.index)
                print(f"{market:16s}{quantity:6s}{horizon:3d}"
                      f"{mae(new, everywhere):8.0f}{mae(old, everywhere):8.0f}"
                      f"{mae(actual_series, everywhere):9.0f}   "
                      f"{mae(new, is_val):9.0f}{mae(actual_series, is_val):13.0f}")


def write_back(predictions: pd.DataFrame) -> int:
    """Rewrite forecast columns in every window panel, leaving all else intact."""
    lookup = predictions.set_index(["timestamp_utc", "market_id"])
    touched = 0
    for path in sorted(PANEL_ROOT.glob("*/canonical_panel.csv")):
        frame = pd.read_csv(path)
        stamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
        keys = pd.MultiIndex.from_arrays([stamps, frame["market_id"]])
        for quantity in QUANTITIES:
            for horizon in HORIZONS:
                column = f"forecast_{quantity}_h{horizon}_mw"
                frame[column] = lookup[column].reindex(keys).to_numpy()
        # The issue hour is the row's own hour, which is what makes the
        # forecast admissible at that row's decision.
        frame["forecast_issue_time_utc"] = stamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        frame["forecast_vintage_id"] = VINTAGE_ID
        frame.to_csv(path, index=False)
        touched += 1
    return touched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="print accuracy, write nothing")
    parser.add_argument("--write", action="store_true", help="rewrite panel forecast columns")
    args = parser.parse_args()

    if not (args.report or args.write):
        parser.error("pass --report or --write")

    panel = load_hourly_series()
    models, predictions = build_all(panel)

    chosen = sorted({model.alpha for model in models.values()})
    print(f"fitted {len(models)} models on {YEAR_START.date()}..{YEAR_END.date()}"
          f" excluding month {VALIDATION_MONTH}"
          f"  alphas chosen: {chosen}")
    report(panel, predictions)

    if args.write:
        touched = write_back(predictions)
        calendar = json.loads(CALENDAR_PATH.read_text(encoding="utf-8"))
        calendar["forecast_model"] = VINTAGE_ID
        CALENDAR_PATH.write_text(
            json.dumps(calendar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"rewrote {touched} panels and updated calendar.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
