"""Load scenario configurations and build DataCenterSite objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from env.dc_site import DataCenterSite
from env.power_model import PowerModel
from env.workload_generator import load_cell_batch_config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_csv_values(path: Path, value_col: str) -> np.ndarray:
    """Load a CSV and return the value column as a numpy array."""
    df = pd.read_csv(path)
    return df[value_col].values.astype(np.float32)


def load_scenario(
    config_path: Path,
    batch_enabled: bool = False,
) -> tuple[list[DataCenterSite], PowerModel, dict[str, Any]]:
    """Load a scenario YAML config and return sites + power model + batch config.

    The YAML config should look like:

        power_model: data/power_model_params.json  # or "default"
        batch:                                      # optional
          flexibility_factor: 1.0
          deadline_penalty_weight: 2.0
          urgency_horizon_steps: 12
        sites:
          - name: US-West
            cell: data/cells/cell_a.csv
            solar: data/solar/the_dalles_or.csv
            price: data/prices/caiso.csv
            solar_capacity_mw: 50.0
            rated_power_mw: 100.0
            batch_distributions: data/jobs/batch_distributions_a.json
          - name: US-Central
            ...

    Args:
        config_path: Path to the scenario YAML file.
        batch_enabled: If True, load per-cell batch configs and set
            batch_fraction / batch_mean_duration_sec on each site.

    Returns:
        Tuple of (sites, power_model, batch_config) where batch_config
        is a dict with environment-wide batch settings (empty when disabled).
    """
    root_dir = Path(__file__).resolve().parent.parent

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load power model
    pm_path = config.get("power_model", "default")
    if pm_path == "default":
        power_model = PowerModel.default()
    else:
        power_model = PowerModel.from_json(root_dir / pm_path)

    # Batch config section (used only when batch_enabled)
    batch_config: dict[str, Any] = config.get("batch", {}) if batch_enabled else {}

    # Load sites
    sites = []
    for site_cfg in config["sites"]:
        workload = load_csv_values(root_dir / site_cfg["cell"], "cpu_demand_norm")
        solar = load_csv_values(root_dir / site_cfg["solar"], "solar_fraction")
        price = load_csv_values(root_dir / site_cfg["price"], "price_usd_kwh")

        # Truncate all arrays to the shortest one for this site
        min_len = min(len(workload), len(solar), len(price))
        workload = workload[:min_len]
        solar = solar[:min_len]
        price = price[:min_len]

        # Batch parameters (defaults make legacy mode identical)
        batch_fraction = 0.0
        batch_mean_duration_sec = 2100.0

        if batch_enabled:
            batch_dist_path = site_cfg.get("batch_distributions")
            if batch_dist_path:
                cell_cfg = load_cell_batch_config(root_dir / batch_dist_path)
                batch_fraction = cell_cfg.batch_fraction
                batch_mean_duration_sec = cell_cfg.mean_duration_sec

        site = DataCenterSite(
            name=site_cfg["name"],
            workload=workload,
            solar=solar,
            price=price,
            solar_capacity_mw=site_cfg.get("solar_capacity_mw", 50.0),
            rated_power_mw=site_cfg.get("rated_power_mw", 100.0),
            capacity=site_cfg.get("capacity", 1.0),
            batch_fraction=batch_fraction,
            batch_mean_duration_sec=batch_mean_duration_sec,
        )
        sites.append(site)

    return sites, power_model, batch_config
