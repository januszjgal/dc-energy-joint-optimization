"""Causal ramp-window reward terms for protocol v6."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RampWindowTerms:
    native_fraction_s_per_hour: float
    no_flexibility_fraction_s_per_hour: float
    adjusted_fraction_s_per_hour: float
    incremental_squared_impact: float


def closed_window_terms(
    net_now_mw: float,
    net_then_mw: float,
    power_now_mw: float,
    power_then_mw: float,
    no_flex_power_now_mw: float,
    no_flex_power_then_mw: float,
    gross_q95_scale_mw: float,
    horizon_hours: int,
) -> RampWindowTerms:
    """Score a window only when its right endpoint has been executed.

    The impact is measured from the no-flexibility schedule on the same grid;
    the native ramp, with no fleet at all, is kept for reporting.
    """
    denominator = gross_q95_scale_mw * horizon_hours
    native = (net_now_mw - net_then_mw) / denominator
    no_flexibility = (
        (net_now_mw + no_flex_power_now_mw)
        - (net_then_mw + no_flex_power_then_mw)
    ) / denominator
    adjusted = (
        (net_now_mw + power_now_mw)
        - (net_then_mw + power_then_mw)
    ) / denominator
    return RampWindowTerms(
        native_fraction_s_per_hour=native,
        no_flexibility_fraction_s_per_hour=no_flexibility,
        adjusted_fraction_s_per_hour=adjusted,
        incremental_squared_impact=adjusted**2 - no_flexibility**2,
    )
