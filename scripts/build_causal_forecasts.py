"""Least-squares gross-demand and net-load forecasts from past values.

Forecast leads 1, 3, 6, and 12 are measured from each row's issue hour. The
decision at hour t reads the row at t-1, so these forecasts predict the
current hour t and hours t+2, t+5, and t+11.

One ordinary least-squares regression is fitted per (market, quantity, lead)
on a given set of training months. The weights are fitted once and then held
fixed; only the inputs change from hour to hour as new observations arrive.

The input set was chosen by four-fold cross-validation over the 2025 months
(``--select``): among nested candidate sets, the smallest whose mean
validation error is within one standard error of the best set's mean.

    python scripts/build_causal_forecasts.py --select   # compare input sets (writes nothing)
    python scripts/build_causal_forecasts.py --report   # CV and test accuracy (writes nothing)
    python scripts/build_causal_forecasts.py --write    # final forecasts into the panels

``--write`` fits on all twelve months of 2025, rewrites the forecast columns of
the committed 2025 panels, and builds the May 2026 test panel from the raw test
pull with the same fixed weights. Cross-validation folds get their own
forecasts from ``scripts/build_data_splits.py``.
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

from energy_model_v3.splits import (  # noqa: E402
    CV_FOLDS, DEVELOPMENT_MONTHS, TEST_MONTHS, month_labels,
)
from env.ramp_v6.models import HISTORY_HOURS  # noqa: E402

PANEL_ROOT = ROOT / "data" / "four_market_2025" / "months"
CALENDAR_PATH = ROOT / "data" / "four_market_2025" / "calendar.json"
FROZEN_STATS_PATH = ROOT / "data" / "four_market_2025" / "frozen_stats.json"
TEST_ROOT = ROOT / "data" / "four_market_2026_test"
TEST_RAW_PATH = TEST_ROOT / "raw_hourly.csv"

HORIZONS = (1, 3, 6, 12)  # Lead hours from the issue row, not the decision hour.
QUANTITIES = ("gross", "net")
QUANTITY_COLUMN = {"gross": "gross_demand_mw", "net": "net_load_mw"}

# Nested candidate input sets, smallest first. Each adds to the one before.
INPUT_SETS = {
    "lags": (
        "now", "change_1h", "target_hour_yesterday", "target_hour_last_week",
    ),
    "lags+issue_yesterday": (
        "now", "change_1h", "issue_hour_yesterday",
        "target_hour_yesterday", "target_hour_last_week",
    ),
    "+hour_of_day": (
        "now", "change_1h", "issue_hour_yesterday",
        "target_hour_yesterday", "target_hour_last_week", "hour_sin", "hour_cos",
    ),
    "+wind_solar": (
        "now", "change_1h", "issue_hour_yesterday",
        "target_hour_yesterday", "target_hour_last_week", "hour_sin", "hour_cos",
        "wind", "solar",
    ),
    "+day_of_week": (
        "now", "change_1h", "issue_hour_yesterday",
        "target_hour_yesterday", "target_hour_last_week", "hour_sin", "hour_cos",
        "wind", "solar", "weekday_sin", "weekday_cos",
    ),
}
# Chosen by --select with the one-standard-error rule.
INPUT_SET = "+wind_solar"

VINTAGE_ID = "causal-least-squares-v4"


@dataclass(frozen=True)
class FittedModel:
    """One least-squares model: an intercept plus one weight per input."""

    coefficients: np.ndarray
    intercept: float
    n_train: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features @ self.coefficients + self.intercept


def load_hourly_series() -> pd.DataFrame:
    """Assemble one continuous hourly frame per market from the 2025 panels.

    Each monthly panel also carries the warm hour that closes the month before
    it, so a (timestamp, market) pair can appear twice. Duplicates are
    identical by construction; the first is kept.
    """
    frames = [pd.read_csv(path) for path in sorted(PANEL_ROOT.glob("*/canonical_panel.csv"))]
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp_utc"] = pd.to_datetime(panel["timestamp_utc"], utc=True)
    panel = panel.drop_duplicates(subset=["timestamp_utc", "market_id"])
    return panel.sort_values(["market_id", "timestamp_utc"]).reset_index(drop=True)


def load_test_series() -> pd.DataFrame:
    """The raw May 2026 test pull, with a week of history before the month."""
    frame = pd.read_csv(TEST_RAW_PATH)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame.sort_values(["market_id", "timestamp_utc"]).reset_index(drop=True)


def build_features(
    market_frame: pd.DataFrame, quantity: str, horizon: int,
    inputs: tuple[str, ...] = INPUT_SETS[INPUT_SET],
) -> pd.DataFrame:
    """Build the causal inputs of one quantity's model at one lead.

    The lag inputs are past values of the series being predicted. The two
    "target hour" inputs look back 24 and 168 hours from the hour being
    predicted, which is at or before the issue hour for every lead up to 24.
    Nothing here may reference a row later than the one it sits on; that
    property is what makes the resulting forecast admissible, and it is
    checked by ``assert_causal`` below.
    """
    if not 1 <= horizon <= 24:
        raise ValueError("the target-hour inputs are causal only for leads of 1 to 24 hours")
    frame = market_frame.set_index("timestamp_utc").asfreq("h")
    series = frame[QUANTITY_COLUMN[quantity]]
    hour = frame.index.hour.to_numpy()
    weekday = frame.index.dayofweek.to_numpy()
    candidates = {
        # The latest value alone reproduces a persistence forecast.
        "now": series,
        # The objective is about ramps, so the rate the series is already
        # moving at is informative beyond its level.
        "change_1h": series.diff(1),
        "issue_hour_yesterday": series.shift(24),
        # The hour being predicted, one day and one week earlier.
        "target_hour_yesterday": series.shift(24 - horizon),
        "target_hour_last_week": series.shift(168 - horizon),
        "hour_sin": pd.Series(np.sin(2 * np.pi * hour / 24), index=frame.index),
        "hour_cos": pd.Series(np.cos(2 * np.pi * hour / 24), index=frame.index),
        "wind": frame["wind_mw"],
        "solar": frame["solar_mw"],
        "weekday_sin": pd.Series(np.sin(2 * np.pi * weekday / 7), index=frame.index),
        "weekday_cos": pd.Series(np.cos(2 * np.pi * weekday / 7), index=frame.index),
    }
    return pd.DataFrame({name: candidates[name] for name in inputs}, index=frame.index)


def assert_causal(
    features: pd.DataFrame, market_frame: pd.DataFrame, quantity: str, horizon: int,
    inputs: tuple[str, ...],
) -> None:
    """Fail loudly if any feature column depends on a future observation.

    Removing the last hours of the series and rebuilding must leave every
    feature unchanged on the overlapping rows. A feature that peeked ahead
    would move.
    """
    truncated = market_frame.iloc[:-6].copy()
    rebuilt = build_features(truncated, quantity, horizon, inputs)
    common = features.index.intersection(rebuilt.index)
    lhs = features.loc[common]
    rhs = rebuilt.loc[common]
    mismatch = ~np.isclose(lhs.to_numpy(dtype=float), rhs.to_numpy(dtype=float),
                           equal_nan=True, rtol=0, atol=1e-9)
    if mismatch.any():
        bad = lhs.columns[mismatch.any(axis=0)].tolist()
        raise AssertionError(f"non-causal feature columns detected: {bad}")


def fit_least_squares(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Ordinary least squares with an intercept, solved by numpy's lstsq."""
    design = np.column_stack([np.ones(len(y)), x])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    return solution[1:], float(solution[0])


def in_months(index: pd.DatetimeIndex, months: tuple[str, ...]) -> np.ndarray:
    return np.asarray(month_labels(index).isin(months))


# The oldest input (the target hour one week earlier) lies 168 - h hours before
# the issue hour, so a row's inputs and target span 169 consecutive hours.
INPUT_SPAN_HOURS = 169


def fit_rows(index: pd.DatetimeIndex, horizon: int, train_months: tuple[str, ...]) -> np.ndarray:
    """Rows whose every input hour and target hour lie in training months.

    Requiring only the issue and target hours would still let a row just after
    a held-out month read that month through its lagged inputs, so a held-out
    month would shape the weights. Purging every row whose 169-hour span
    touches a non-training hour keeps the fit to training data alone.
    """
    training_hour = pd.Series(in_months(index, train_months).astype(float), index=index)
    span_is_training = training_hour.rolling(INPUT_SPAN_HOURS, min_periods=INPUT_SPAN_HOURS).min()
    return (span_is_training.shift(-horizon) == 1.0).to_numpy()


def fit_market_quantity_horizon(
    features: pd.DataFrame, target: pd.Series, horizon: int, train_months: tuple[str, ...],
) -> FittedModel:
    """Fit one direct multi-step model predicting ``horizon`` hours ahead.

    A separate model per horizon is what makes the forecast columns genuinely
    different. Recursive one-step forecasting, or copying one series into
    all columns, would not.
    """
    aligned_target = target.shift(-horizon)
    usable = features.notna().all(axis=1) & aligned_target.notna()
    mask = usable & fit_rows(features.index, horizon, train_months)

    x = features.loc[mask].to_numpy(dtype=float)
    y = aligned_target.loc[mask].to_numpy(dtype=float)
    if len(y) == 0:
        raise ValueError("no training rows survive the feature and calendar masks")

    coefficients, intercept = fit_least_squares(x, y)
    return FittedModel(coefficients, intercept, int(len(y)))


def fit_forecasts(
    panel: pd.DataFrame, train_months: tuple[str, ...],
    inputs: tuple[str, ...] = INPUT_SETS[INPUT_SET],
) -> tuple[dict[tuple[str, str, int], FittedModel], list[tuple[str, str, int, float, float]]]:
    """Fit every (market, quantity, lead) model on ``train_months``.

    Returns the models and any that failed to beat persistence on their own
    training hours. Those are replaced by persistence when predicting: a
    forecast worse than the last observation could actively mislead the
    scheduler. The check uses training hours only, at both ends.
    """
    models: dict[tuple[str, str, int], FittedModel] = {}
    fallbacks: list[tuple[str, str, int, float, float]] = []
    for market, market_frame in panel.groupby("market_id", sort=True):
        frame = market_frame.set_index("timestamp_utc").asfreq("h")
        for quantity in QUANTITIES:
            target = frame[QUANTITY_COLUMN[quantity]]
            for horizon in HORIZONS:
                features = build_features(market_frame, quantity, horizon, inputs)
                assert_causal(features, market_frame, quantity, horizon, inputs)
                model = fit_market_quantity_horizon(features, target, horizon, train_months)
                models[(market, quantity, horizon)] = model
                actual = target.shift(-horizon)
                complete = features.notna().all(axis=1)
                scored = complete & actual.notna() & fit_rows(frame.index, horizon, train_months)
                prediction = model.predict(features.loc[scored].to_numpy(dtype=float))
                model_mae = float(np.mean(np.abs(prediction - actual[scored].to_numpy())))
                persist_mae = float((target[scored] - actual[scored]).abs().mean())
                if not model_mae < persist_mae:
                    fallbacks.append((market, quantity, horizon, model_mae, persist_mae))
    return models, fallbacks


def predict(
    series: pd.DataFrame,
    models: dict[tuple[str, str, int], FittedModel],
    fallbacks: list[tuple[str, str, int, float, float]] = (),
    inputs: tuple[str, ...] = INPUT_SETS[INPUT_SET],
) -> pd.DataFrame:
    """Apply fixed models to a long-form hourly frame; one column per lead.

    Rows without a complete input history (the first week of a series) and
    any model that fell back get the persistence value.
    """
    replaced = {(market, quantity, horizon) for market, quantity, horizon, *_ in fallbacks}
    outputs = []
    for market, market_frame in series.groupby("market_id", sort=True):
        frame = market_frame.set_index("timestamp_utc").asfreq("h")
        out = pd.DataFrame(index=frame.index)
        out["market_id"] = market
        for quantity in QUANTITIES:
            target = frame[QUANTITY_COLUMN[quantity]]
            for horizon in HORIZONS:
                column = f"forecast_{quantity}_h{horizon}_mw"
                features = build_features(market_frame, quantity, horizon, inputs)
                complete = features.notna().all(axis=1)
                out[column] = np.nan
                if (market, quantity, horizon) not in replaced and complete.any():
                    out.loc[complete, column] = models[(market, quantity, horizon)].predict(
                        features.loc[complete].to_numpy(dtype=float)
                    )
                out[column] = out[column].fillna(target)
        outputs.append(out.reset_index().rename(columns={"index": "timestamp_utc"}))
    return pd.concat(outputs, ignore_index=True)


def score(
    series: pd.DataFrame, predictions: pd.DataFrame, issue_months: tuple[str, ...],
) -> pd.DataFrame:
    """MAE, RMSE, and MAPE of every model and of persistence on ``issue_months``.

    Issue hours in those months are scored against the realized value at the
    target hour, which may fall just after the month.
    """
    base = series.drop(columns=[c for c in series.columns if c.startswith("forecast_")])
    merged = base.merge(predictions, on=["timestamp_utc", "market_id"])
    rows = []
    for market, group in merged.groupby("market_id", sort=True):
        group = group.set_index("timestamp_utc").asfreq("h")
        issue = in_months(group.index, issue_months)
        for quantity in QUANTITIES:
            realized = group[QUANTITY_COLUMN[quantity]]
            for horizon in HORIZONS:
                actual = realized.shift(-horizon)
                selected = issue & actual.notna().to_numpy()
                truth = actual[selected].to_numpy(dtype=float)
                row = {"market": market, "quantity": quantity, "horizon": horizon,
                       "hours": int(selected.sum())}
                for name, forecast in (
                    ("ls", group[f"forecast_{quantity}_h{horizon}_mw"]),
                    ("persistence", realized),
                ):
                    error = forecast[selected].to_numpy(dtype=float) - truth
                    row[f"{name}_mae"] = float(np.mean(np.abs(error)))
                    row[f"{name}_rmse"] = float(np.sqrt(np.mean(error**2)))
                    row[f"{name}_mape"] = float(100.0 * np.mean(np.abs(error / truth)))
                rows.append(row)
    return pd.DataFrame(rows)


def cross_validate(
    panel: pd.DataFrame, inputs: tuple[str, ...] = INPUT_SETS[INPUT_SET],
) -> dict[str, pd.DataFrame]:
    """Fit on each fold's nine training months and score its three held-out months."""
    results = {}
    for fold, held_out in CV_FOLDS.items():
        train = tuple(month for month in DEVELOPMENT_MONTHS if month not in held_out)
        models, fallbacks = fit_forecasts(panel, train, inputs)
        results[fold] = score(panel, predict(panel, models, fallbacks, inputs), held_out)
    return results


def select_inputs(panel: pd.DataFrame) -> str:
    """One-standard-error rule over the nested candidate input sets.

    A fold's score is the mean, over the 32 models, of model MAE divided by
    persistence MAE on that fold's held-out months. The chosen set is the
    smallest whose mean fold score is within one standard error (across the
    four folds) of the best set's mean.
    """
    summary = {}
    for name, inputs in INPUT_SETS.items():
        folds = cross_validate(panel, inputs)
        ratios = [float((frame.ls_mae / frame.persistence_mae).mean()) for frame in folds.values()]
        summary[name] = (float(np.mean(ratios)), float(np.std(ratios, ddof=1) / np.sqrt(len(ratios))), ratios)
    best = min(summary, key=lambda name: summary[name][0])
    threshold = summary[best][0] + summary[best][1]
    chosen = next(name for name in INPUT_SETS if summary[name][0] <= threshold)
    print(f"{'input set':24s}{'inputs':>7s}{'mean':>9s}{'SE':>9s}   fold ratios (MAE / persistence MAE)")
    for name, (mean, se, ratios) in summary.items():
        flag = "  <- best" if name == best else ""
        flag += "  <- chosen" if name == chosen else ""
        print(f"{name:24s}{len(INPUT_SETS[name]):7d}{mean:9.4f}{se:9.4f}   "
              + " ".join(f"{r:.4f}" for r in ratios) + flag)
    print(f"one-standard-error threshold: {threshold:.4f}")
    return chosen


def write_forecast_columns(
    source: Path, destination: Path, lookup: pd.DataFrame, vintage: str,
    market_scales: dict[str, float] | None = None,
) -> None:
    """Copy one panel, replacing its forecast columns (and optionally S_m)."""
    frame = pd.read_csv(source)
    stamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    keys = pd.MultiIndex.from_arrays([stamps, frame["market_id"]])
    for quantity in QUANTITIES:
        for horizon in HORIZONS:
            column = f"forecast_{quantity}_h{horizon}_mw"
            frame[column] = lookup[column].reindex(keys).to_numpy()
    if frame.filter(like="forecast_").isna().any().any():
        raise ValueError(f"{source}: forecasts missing for some panel rows")
    # The issue hour is the row's own hour, which is what makes the forecast
    # admissible at that row's decision.
    frame["forecast_issue_time_utc"] = stamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["forecast_vintage_id"] = vintage
    if market_scales is not None:
        frame["market_scale_mw"] = frame["market_id"].map(market_scales)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, lineterminator="\n")


def write_development_panels(predictions: pd.DataFrame, market_scales: dict[str, float]) -> int:
    """Rewrite forecast columns and S_m in every committed 2025 panel."""
    lookup = predictions.set_index(["timestamp_utc", "market_id"])
    touched = 0
    for path in sorted(PANEL_ROOT.glob("*/canonical_panel.csv")):
        write_forecast_columns(path, path, lookup, VINTAGE_ID, market_scales)
        touched += 1
    return touched


def write_test_panel(predictions: pd.DataFrame, market_scales: dict[str, float]) -> Path:
    """Build the May 2026 test panel: its warm hour, the month, fixed forecasts.

    The market scales are the frozen development-year values, as for every
    quantity the test month reuses.
    """
    (month_id,) = TEST_MONTHS
    raw = load_test_series()
    start = pd.Timestamp(f"{month_id}-01T00:00:00Z")
    end = start + pd.offsets.MonthBegin(1)
    window = raw[(raw.timestamp_utc >= start - pd.Timedelta(hours=HISTORY_HOURS))
                 & (raw.timestamp_utc < end)].copy()
    window = window.merge(predictions, on=["timestamp_utc", "market_id"], how="left")
    window["market_scale_mw"] = window["market_id"].map(market_scales)
    window["forecast_issue_time_utc"] = window["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    window["forecast_vintage_id"] = VINTAGE_ID
    window["quality_ok"] = True
    columns = pd.read_csv(sorted(PANEL_ROOT.glob("*/canonical_panel.csv"))[0], nrows=0).columns
    window["timestamp_utc"] = window["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    window = window.sort_values(["timestamp_utc", "market_id"])[list(columns)]
    if window.isna().any().any():
        raise ValueError("test panel has missing values")
    path = TEST_ROOT / "months" / month_id / "canonical_panel.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    window.to_csv(path, index=False, lineterminator="\n")
    return path


def print_scores(title: str, frame: pd.DataFrame) -> None:
    gain = 1.0 - frame.ls_mae / frame.persistence_mae
    print(f"\n{title}: MAE below persistence {gain.min():.1%}..{gain.max():.1%} "
          f"(mean {gain.mean():.1%}); MAE/persistence {(frame.ls_mae / frame.persistence_mae).mean():.4f}")
    print(frame.round(1).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="store_true", help="compare input sets by CV, write nothing")
    parser.add_argument("--report", action="store_true", help="CV and test accuracy, write nothing")
    parser.add_argument("--write", action="store_true", help="write final forecasts into the panels")
    args = parser.parse_args()
    if not (args.select or args.report or args.write):
        parser.error("pass --select, --report, or --write")

    panel = load_hourly_series()
    if args.select:
        chosen = select_inputs(panel)
        print(f"chosen input set: {chosen}  (configured INPUT_SET = {INPUT_SET})")
        return 0

    models, fallbacks = fit_forecasts(panel, DEVELOPMENT_MONTHS)
    for market, quantity, horizon, model_mae, persist_mae in fallbacks:
        print(f"  fell back to persistence: {market} {quantity} h{horizon} "
              f"(model {model_mae:.0f} MW vs persistence {persist_mae:.0f} MW)")
    rows = sorted({model.n_train for model in models.values()})
    print(f"fitted {len(models)} models on all of 2025 ({INPUT_SET}: "
          f"{', '.join(INPUT_SETS[INPUT_SET])}); training rows per model {rows[0]}..{rows[-1]}")

    if args.report:
        for fold, frame in cross_validate(panel).items():
            print_scores(f"cross-validation fold {fold} {CV_FOLDS[fold]}", frame)
        test = load_test_series()
        print_scores(f"test {TEST_MONTHS}", score(test, predict(test, models, fallbacks), TEST_MONTHS))

    if args.write:
        scales = json.loads(FROZEN_STATS_PATH.read_text(encoding="utf-8"))["gross_q95_mw"]
        touched = write_development_panels(predict(panel, models, fallbacks), scales)
        test_path = write_test_panel(predict(load_test_series(), models, fallbacks), scales)
        calendar = json.loads(CALENDAR_PATH.read_text(encoding="utf-8"))
        calendar["forecast_model"] = VINTAGE_ID
        CALENDAR_PATH.write_text(
            json.dumps(calendar, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"rewrote {touched} development panels, wrote {test_path}, updated calendar.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
