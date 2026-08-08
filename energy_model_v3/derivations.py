"""Training-only scales and primary hourly ramp evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .contract import ContractError


@dataclass(frozen=True)
class MarketScale:
    market: str
    train_start_utc: str
    train_end_utc: str
    gross_demand_q95_mw: float
    net_load_q90_fraction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calibrate_scale(
    frame: pd.DataFrame,
    *,
    market: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> MarketScale:
    ts = pd.to_datetime(frame["interval_start_utc"], utc=True)
    train = frame.loc[(ts >= train_start) & (ts < train_end)].copy()
    if train.empty:
        raise ContractError(f"{market} has no training rows")
    gross = pd.to_numeric(train["gross_demand_mw"], errors="raise")
    net = pd.to_numeric(train["net_load_mw"], errors="raise")
    scale = float(gross.quantile(0.95))
    if not np.isfinite(scale) or scale <= 0:
        raise ContractError(f"{market} training Q95 gross demand is invalid")
    q90 = float((net / scale).quantile(0.90))
    return MarketScale(
        market=market,
        train_start_utc=pd.Timestamp(train_start).isoformat(),
        train_end_utc=pd.Timestamp(train_end).isoformat(),
        gross_demand_q95_mw=scale,
        net_load_q90_fraction=q90,
    )


def add_grid_features(
    frame: pd.DataFrame,
    dc_power_mw: pd.Series,
    scale: MarketScale,
    *,
    residual_tail_weight: float = 0.10,
) -> pd.DataFrame:
    """Add causal hourly grid features; no circular wrap or future filling."""
    result = frame.copy().sort_values("interval_start_utc").reset_index(drop=True)
    if len(result) != len(dc_power_mw):
        raise ContractError("DC power and market frame lengths differ")
    if not 0 <= residual_tail_weight < 1:
        raise ContractError("residual tail weight must be in [0, 1)")
    gross = pd.to_numeric(result["gross_demand_mw"], errors="raise")
    net = pd.to_numeric(result["net_load_mw"], errors="raise")
    dc = pd.Series(dc_power_mw, index=result.index, dtype=float)
    if dc.isna().any() or (dc < 0).any():
        raise ContractError("modeled DC power must be finite and non-negative")
    s = scale.gross_demand_q95_mw
    result["market_scale_mw"] = s
    result["gross_level_norm"] = gross / s
    result["net_level_norm"] = net / s
    result["dc_power_mw"] = dc
    result["prior_dc_power_mw"] = dc.shift(1)
    result["p_over_s"] = dc / s
    result["p_over_d"] = dc / gross.replace(0.0, np.nan)
    result["net_residual_above_train_q90"] = (
        result["net_level_norm"] - scale.net_load_q90_fraction
    ).clip(lower=0.0)

    weighted = pd.Series(0.0, index=result.index)
    for hours, weight in ((1, 0.40), (3, 0.60)):
        base = (net - net.shift(hours)) / (s * hours)
        augmented = ((net + dc) - (net + dc).shift(hours)) / (s * hours)
        incremental = augmented.pow(2) - base.pow(2)
        result[f"net_ramp_{hours}h_norm_per_h"] = base
        result[f"augmented_ramp_{hours}h_norm_per_h"] = augmented
        result[f"incremental_ramp_{hours}h"] = incremental
        weighted = weighted + weight * incremental.fillna(0.0)
    result["incremental_ramp_weighted"] = weighted
    result["incremental_residual_tail"] = residual_tail_weight * (
        result["net_residual_above_train_q90"]
        * result["p_over_s"]
    )
    result["grid_ramp_evidence"] = (
        (1.0 - residual_tail_weight) * weighted
        + result["incremental_residual_tail"]
    )
    result["da_energy_cost_usd"] = (
        pd.to_numeric(result["da_lmp_usd_mwh"], errors="raise") * dc
    )
    return result


def validate_episode_support(
    frame: pd.DataFrame,
    *,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
) -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(frame["interval_start_utc"], utc=True)
    )
    warm = pd.date_range(
        episode_start - pd.Timedelta(hours=3),
        episode_start,
        freq="1h",
        inclusive="left",
    )
    tail = pd.date_range(
        episode_end,
        episode_end + pd.Timedelta(hours=3),
        freq="1h",
        inclusive="left",
    )
    missing = warm.union(tail).difference(index)
    if len(missing):
        raise ContractError(
            f"episode support lacks {len(missing)} warm-history/terminal-tail rows"
        )


def diagnostic_summary(panel: pd.DataFrame) -> dict[str, object]:
    numeric = panel.select_dtypes(include=[np.number]).replace(
        [np.inf, -np.inf], np.nan
    )
    complete = numeric.dropna(axis=1, how="any")
    rank = int(np.linalg.matrix_rank(complete.to_numpy())) if not complete.empty else 0
    singular = (
        np.linalg.svd(complete.to_numpy(), compute_uv=False)
        if not complete.empty
        else np.array([])
    )
    tol = (
        max(complete.shape) * np.finfo(float).eps * singular.max()
        if singular.size
        else 0.0
    )
    effective_rank = int((singular > tol).sum())
    ramps: dict[str, dict[str, float]] = {}
    for horizon in (1, 3):
        column = f"net_ramp_{horizon}h_norm_per_h"
        values = pd.to_numeric(panel[column], errors="coerce").dropna()
        ramps[f"{horizon}h"] = {
            "q05": float(values.quantile(0.05)),
            "q50": float(values.quantile(0.50)),
            "q90": float(values.quantile(0.90)),
            "q95": float(values.quantile(0.95)),
            "max": float(values.max()),
        }
    return {
        "rows": len(panel),
        "numeric_columns": len(numeric.columns),
        "matrix_rank": rank,
        "effective_rank": effective_rank,
        "ramp_distributions": ramps,
    }
