"""Linear power model: converts CPU utilization to power consumption.

Calibrated from Google PowerData2019. The model is:
    P = idle_power + slope * cpu_utilization

where all values are in normalized utilization space [0, 1].
To get actual power in kW, multiply by the DC's rated power capacity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PowerModel:
    """Linear power model mapping CPU utilization to power utilization."""

    idle_power: float  # Power draw at zero load (normalized)
    slope: float  # Additional power per unit CPU util
    peak_power: float  # Power at 100% CPU (idle + slope)

    def compute(self, cpu_util: float) -> float:
        """Return normalized power utilization for given CPU utilization."""
        return self.idle_power + self.slope * cpu_util

    @classmethod
    def from_json(cls, path: Path) -> PowerModel:
        """Load from power_model_params.json."""
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        return cls(
            idle_power=params["idle_power"],
            slope=params["slope"],
            peak_power=params["peak_power"],
        )

    @classmethod
    def default(cls) -> PowerModel:
        """Return a reasonable default if no calibration data is available.

        Based on typical values from the literature (Fan et al., 2007):
        idle power ~50-60% of peak, linear relationship.
        """
        return cls(idle_power=0.55, slope=0.45, peak_power=1.0)
