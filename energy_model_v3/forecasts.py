"""Leakage-safe reconstructed rolling-origin forecasts for live v3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calendar import StudyCalendar
from .contract import ContractError, validate_forecasts

LAGS = (1, 3, 24, 168)
TRAILING_WINDOWS = (24, 168)
HORIZONS = (1, 3)


@dataclass(frozen=True)
class RidgeModel:
    intercept: float
    coefficients: tuple[float, ...]
    feature_names: tuple[str, ...]
    alpha: float
    train_rows: int
    train_sha256: str
    earliest_training_target_utc: str
    latest_training_target_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "feature_names": list(self.feature_names),
            "alpha": self.alpha,
            "train_rows": self.train_rows,
            "train_sha256": self.train_sha256,
            "earliest_training_target_utc": self.earliest_training_target_utc,
            "latest_training_target_utc": self.latest_training_target_utc,
        }


def _features(
    values: pd.Series,
    timestamps: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    data: dict[str, Any] = {}
    for lag in LAGS:
        data[f"observed_lag_{lag}h"] = values.shift(horizon + lag)
    for window in TRAILING_WINDOWS:
        data[f"observed_trailing_mean_{window}h"] = (
            values.shift(horizon + 1).rolling(window, min_periods=window).mean()
        )
    target_hour = timestamps.hour.to_numpy()
    target_dow = timestamps.dayofweek.to_numpy()
    data["target_hour_sin"] = np.sin(2 * np.pi * target_hour / 24)
    data["target_hour_cos"] = np.cos(2 * np.pi * target_hour / 24)
    data["target_dow_sin"] = np.sin(2 * np.pi * target_dow / 7)
    data["target_dow_cos"] = np.cos(2 * np.pi * target_dow / 7)
    return pd.DataFrame(data, index=values.index)


def _fit_ridge(
    features: pd.DataFrame,
    target: pd.Series,
    training_timestamps: pd.DatetimeIndex,
    *,
    alpha: float,
) -> RidgeModel:
    x = features.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    if len(x) < 24 * 30:
        raise ContractError("reconstructed forecast has insufficient training rows")
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0] = 1.0
    normalized = (x - means) / scales
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ y,
    )
    coefficients = beta[1:] / scales
    intercept = float(beta[0] - np.dot(means, coefficients))
    payload = pd.concat(
        [
            pd.Series(training_timestamps.astype(str), name="target_utc"),
            features.reset_index(drop=True),
            target.reset_index(drop=True),
        ],
        axis=1,
    ).to_csv(index=False).encode()
    return RidgeModel(
        intercept=intercept,
        coefficients=tuple(float(value) for value in coefficients),
        feature_names=tuple(features.columns),
        alpha=alpha,
        train_rows=len(features),
        train_sha256=hashlib.sha256(payload).hexdigest(),
        earliest_training_target_utc=training_timestamps.min().isoformat(),
        latest_training_target_utc=training_timestamps.max().isoformat(),
    )


def reconstruct_forecasts(
    frame: pd.DataFrame,
    *,
    market: str,
    calendar: StudyCalendar,
    target_column: str = "net_load_mw",
    alpha: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = frame.sort_values("interval_start_utc").reset_index(drop=True)
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(ordered["interval_start_utc"], utc=True)
    )
    values = pd.to_numeric(ordered[target_column], errors="raise")
    train_end = pd.Timestamp(calendar.train_end_utc)
    forecasts = []
    models: dict[str, Any] = {}
    for horizon in HORIZONS:
        features = _features(values, timestamps, horizon)
        usable = features.notna().all(axis=1) & values.notna()
        issues = timestamps - pd.Timedelta(hours=horizon)
        horizon_models: list[dict[str, Any]] = []

        def emit(
            selection: np.ndarray,
            model: RidgeModel,
            *,
            vintage: pd.Timestamp,
            capability: str,
        ) -> None:
            if not selection.any():
                return
            model_id = (
                f"{market}-{horizon}h-{vintage:%Y%m%dT%H%M%SZ}-"
                f"{model.train_sha256[:12]}"
            )
            prediction = (
                model.intercept
                + features.loc[selection].to_numpy()
                @ np.asarray(model.coefficients)
            )
            forecasts.append(
                pd.DataFrame(
                    {
                        "interval_start_utc": timestamps[selection],
                        "market": market,
                        "forecast_target": target_column,
                        "forecast_value_mw": prediction,
                        "forecast_issue_utc": issues[selection],
                        "forecast_vintage_utc": vintage,
                        "forecast_horizon_hours": horizon,
                        "forecast_capability": capability,
                        "forecast_model_id": model_id,
                        "source_quality_flags": (
                            "rolling_origin_lagged_observations_calendar_features"
                        ),
                    }
                )
            )
            record = model.to_dict()
            record.update(
                {
                    "model_id": model_id,
                    "vintage_utc": vintage.isoformat(),
                    "horizon_hours": horizon,
                    "capability": capability,
                }
            )
            horizon_models.append(record)

        training_targets = usable & (timestamps < train_end)
        training_vintages = pd.DatetimeIndex(
            issues[training_targets]
        ).floor("D").unique()
        for vintage in training_vintages:
            history = usable & (timestamps < vintage)
            if int(history.sum()) < 24 * 30:
                continue
            selection = (
                training_targets
                & (issues >= vintage)
                & (issues < vintage + pd.Timedelta(days=1))
            )
            model = _fit_ridge(
                features.loc[history],
                values.loc[history],
                timestamps[history],
                alpha=alpha,
            )
            emit(
                selection,
                model,
                vintage=vintage,
                capability="reconstructed_causal_expanding_ridge",
            )

        final_history = usable & (timestamps < train_end)
        final_model = _fit_ridge(
            features.loc[final_history],
            values.loc[final_history],
            timestamps[final_history],
            alpha=alpha,
        )
        post_train = usable & (issues >= train_end)
        emit(
            post_train,
            final_model,
            vintage=train_end,
            capability="reconstructed_train_only_frozen_ridge",
        )
        models[f"{horizon}h"] = {
            "training_refit_cadence": "daily_expanding_window",
            "validation_test_policy": "frozen_at_training_boundary",
            "models": horizon_models,
        }
    result = validate_forecasts(pd.concat(forecasts, ignore_index=True))
    actual = ordered.set_index("interval_start_utc")[target_column]
    actual.index = pd.to_datetime(actual.index, utc=True)
    diagnostics: dict[str, Any] = {}
    for horizon, group in result.groupby("forecast_horizon_hours"):
        observed = actual.reindex(group["interval_start_utc"]).to_numpy()
        error = group["forecast_value_mw"].to_numpy() - observed
        split = np.select(
            [
                group["interval_start_utc"] < pd.Timestamp(calendar.train_end_utc),
                group["interval_start_utc"]
                < pd.Timestamp(calendar.validation_end_utc),
            ],
            ["train", "validation"],
            default="sealed_test",
        )
        diagnostics[f"{int(horizon)}h"] = {}
        for name in ("train", "validation", "sealed_test"):
            selected = error[split == name]
            diagnostics[f"{int(horizon)}h"][name] = {
                "rows": int(len(selected)),
                "mae_mw": float(np.mean(np.abs(selected))),
                "rmse_mw": float(np.sqrt(np.mean(selected**2))),
                "bias_mw": float(np.mean(selected)),
            }
    return result, {
        "market": market,
        "target": target_column,
        "training_boundary_utc": calendar.train_end_utc,
        "future_realized_values_used_as_features": False,
        "models": models,
        "errors": diagnostics,
    }


def write_forecasts(
    destination: Path,
    frames: dict[str, pd.DataFrame],
    *,
    calendar: StudyCalendar,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"markets": {}}
    all_forecasts = []
    for market, frame in frames.items():
        forecast, metadata = reconstruct_forecasts(
            frame, market=market, calendar=calendar
        )
        path = destination / f"{market}.csv"
        forecast.to_csv(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata["artifact"] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        manifest["markets"][market] = metadata
        all_forecasts.append(forecast)
    validate_forecasts(pd.concat(all_forecasts, ignore_index=True))
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
