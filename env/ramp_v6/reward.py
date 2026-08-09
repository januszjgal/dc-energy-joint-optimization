"""Causal ramp-window reward terms for protocol v6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RampWindowTerms:
    native_fraction_s_per_hour: float
    adjusted_fraction_s_per_hour: float
    incremental_squared_impact: float
    native_tail_burden: float
    adjusted_tail_burden: float
    incremental_tail_burden: float


def closed_window_terms(
    net_now_mw: float,
    net_then_mw: float,
    power_now_mw: float,
    power_then_mw: float,
    gross_q95_scale_mw: float,
    horizon_hours: int,
    tail_q90_fraction_s_per_hour: float,
) -> RampWindowTerms:
    """Score a window only when its right endpoint has been executed."""
    denominator = gross_q95_scale_mw * horizon_hours
    native = (net_now_mw - net_then_mw) / denominator
    adjusted = (
        (net_now_mw + power_now_mw)
        - (net_then_mw + power_then_mw)
    ) / denominator
    native_tail = max(abs(native) - tail_q90_fraction_s_per_hour, 0.0) ** 2
    adjusted_tail = (
        max(abs(adjusted) - tail_q90_fraction_s_per_hour, 0.0) ** 2
    )
    return RampWindowTerms(
        native_fraction_s_per_hour=native,
        adjusted_fraction_s_per_hour=adjusted,
        incremental_squared_impact=adjusted**2 - native**2,
        native_tail_burden=native_tail,
        adjusted_tail_burden=adjusted_tail,
        incremental_tail_burden=adjusted_tail - native_tail,
    )


def smooth_worst_market_positive_harm(
    impacts: list[float] | np.ndarray,
    temperature: float,
) -> float:
    """Smoothly penalize the worst positive market harm without a tolerance."""
    values = np.asarray(impacts, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("market impacts must be a non-empty vector")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("worst-market temperature must be positive")
    positive = temperature * np.logaddexp(0.0, values / temperature)
    maximum = float(positive.max())
    return float(
        maximum
        + temperature
        * np.log(np.exp((positive - maximum) / temperature).mean())
    )


def causal_anticipatory_potential(
    forecast_up_fraction: list[float] | np.ndarray,
    previous_power_fraction: list[float] | np.ndarray,
    queued_work_fraction: float,
) -> float:
    """Causal state potential for forecast-aware queue and power positioning."""
    forecast = np.maximum(
        np.asarray(forecast_up_fraction, dtype=np.float64), 0.0
    )
    power = np.asarray(previous_power_fraction, dtype=np.float64)
    if forecast.ndim != 1 or power.shape != forecast.shape or forecast.size == 0:
        raise ValueError("forecast and power fractions must be aligned vectors")
    if (
        np.any(~np.isfinite(forecast))
        or np.any(~np.isfinite(power))
        or not np.isfinite(queued_work_fraction)
        or queued_work_fraction < 0.0
    ):
        raise ValueError("causal potential inputs must be finite and non-negative")
    risk = float(forecast.mean())
    positioned_power = float(np.mean(forecast * np.maximum(power, 0.0)))
    return -(risk * float(queued_work_fraction) + positioned_power)
