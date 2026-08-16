"""Causal ramp-window reward terms for protocol v6."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RampWindowTerms:
    native_fraction_s_per_hour: float
    adjusted_fraction_s_per_hour: float
    incremental_squared_impact: float


def closed_window_terms(
    net_now_mw: float,
    net_then_mw: float,
    power_now_mw: float,
    power_then_mw: float,
    gross_q95_scale_mw: float,
    horizon_hours: int,
) -> RampWindowTerms:
    """Score a window only when its right endpoint has been executed."""
    denominator = gross_q95_scale_mw * horizon_hours
    native = (net_now_mw - net_then_mw) / denominator
    adjusted = (
        (net_now_mw + power_now_mw)
        - (net_then_mw + power_then_mw)
    ) / denominator
    return RampWindowTerms(
        native_fraction_s_per_hour=native,
        adjusted_fraction_s_per_hour=adjusted,
        incremental_squared_impact=adjusted**2 - native**2,
    )
