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
        """Load the pooled (fleet-wide) model from power_model_params.json."""
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        return cls(
            idle_power=params["idle_power"],
            slope=params["slope"],
            peak_power=params["peak_power"],
        )

    @classmethod
    def per_cell_from_json(cls, path: Path) -> dict[str, PowerModel]:
        """Load per-cell calibrated models (R² 0.75–0.80 vs pooled 0.43; each cell's
        machine mix has its own idle/slope). Mirrors CICS practice: "power models
        trained separately for each cluster" (Radovanović et al. 2023). Returns {}
        if the JSON predates the per-cell diagnostics."""
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        return {
            cell: cls(
                idle_power=m["idle_power"],
                slope=m["slope"],
                peak_power=m["peak_power"],
            )
            for cell, m in params.get("per_cell_cpu_model", {}).items()
        }

    @classmethod
    def default(cls) -> PowerModel:
        """Return a reasonable default if no calibration data is available.

        Based on typical values from the literature (Fan et al., 2007):
        idle power ~50-60% of peak, linear relationship.
        """
        return cls(idle_power=0.55, slope=0.45, peak_power=1.0)
