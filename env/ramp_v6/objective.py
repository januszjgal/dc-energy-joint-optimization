"""Fixed-scale regional net-load peak and one-hour ramp objective."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


OBJECTIVE_VERSION = "joint-net-load-peak-ramp-v1"
NORMALIZATION_METHOD = "training-month-mean-no-flexibility-v1"


@dataclass(frozen=True)
class JointObjective:
    ramp_weight: float = 0.5
    peak_weight: float = 0.5
    ramp_reference: float = 1.0
    peak_reference: float = 1.0

    def validate(self) -> None:
        values = (
            self.ramp_weight, self.peak_weight,
            self.ramp_reference, self.peak_reference,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("objective weights and references must be finite")
        if min(self.ramp_weight, self.peak_weight) < 0.0 or not math.isclose(
            self.ramp_weight + self.peak_weight, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("objective weights must be non-negative and sum to one")
        if min(self.ramp_reference, self.peak_reference) <= 0.0:
            raise ValueError("objective references must be strictly positive")

    def score(self, ramp_mean_squared: float, normalized_peak: float) -> float:
        return (
            self.ramp_weight * ramp_mean_squared / self.ramp_reference
            + self.peak_weight * normalized_peak / self.peak_reference
        )

    def reward_components(
        self, ramp_squared: float, peak_increment: float, decision_steps: int
    ) -> tuple[float, float]:
        if decision_steps <= 0:
            raise ValueError("objective requires at least one decision slot")
        return (
            -self.ramp_weight * ramp_squared / (decision_steps * self.ramp_reference),
            -self.peak_weight * peak_increment / self.peak_reference,
        )

    def as_dict(self) -> dict[str, str | float]:
        return {
            "version": OBJECTIVE_VERSION,
            "normalization": NORMALIZATION_METHOD,
            "ramp_weight": self.ramp_weight,
            "peak_weight": self.peak_weight,
            "ramp_reference": self.ramp_reference,
            "peak_reference": self.peak_reference,
            "ramp_metric": "mean_hourly_sum_of_regional_normalized_squared_adjusted_ramps",
            "peak_metric": "sum_of_regional_normalized_decision_month_net_load_maxima",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JointObjective:
        if (
            payload.get("version") != OBJECTIVE_VERSION
            or payload.get("normalization") != NORMALIZATION_METHOD
        ):
            raise ValueError("incompatible joint objective metadata")
        result = cls(**{
            name: float(payload[name])
            for name in ("ramp_weight", "peak_weight", "ramp_reference", "peak_reference")
        })
        result.validate()
        return result


def update_peak(previous: float | None, current: float) -> tuple[float, float]:
    """Initialize on the first decision, then accumulate record increases."""
    if not math.isfinite(current) or (
        previous is not None and not math.isfinite(previous)
    ):
        raise ValueError("peak observations must be finite")
    peak = current if previous is None else max(previous, current)
    return peak, peak - (0.0 if previous is None else previous)


def trajectory_scores(
    net_load_mw: np.ndarray, power_mw: np.ndarray, market_scales_mw: np.ndarray
) -> tuple[float, float]:
    """Absolute reference metrics; row zero is warm history, not a peak sample."""
    net = np.asarray(net_load_mw, dtype=np.float64)
    power = np.asarray(power_mw, dtype=np.float64)
    scales = np.asarray(market_scales_mw, dtype=np.float64)
    if net.ndim != 2 or net.shape != power.shape or net.shape[0] < 2:
        raise ValueError("trajectory requires matching warm-plus-decision matrices")
    if scales.shape != (net.shape[1],) or np.any(scales <= 0.0):
        raise ValueError("trajectory requires one positive scale per market")
    if not all(np.isfinite(values).all() for values in (net, power, scales)):
        raise ValueError("trajectory inputs must be finite")
    adjusted = (net + power) / scales
    ramp = float(np.square(np.diff(adjusted, axis=0)).sum(axis=1).mean())
    peak = float(adjusted[1:].max(axis=0).sum())
    return ramp, peak
