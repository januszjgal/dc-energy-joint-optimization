"""Load scenario configurations and build DataCenterSite objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from env.dc_site import DataCenterSite
from env.power_model import PowerModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_csv_values(path: Path, value_col: str) -> np.ndarray:
    """Load a CSV and return the value column as a numpy array."""
    df = pd.read_csv(path)
    return df[value_col].values.astype(np.float32)


def load_scenario(config_path: Path) -> tuple[list[DataCenterSite], PowerModel]:
    """Load a scenario YAML config and return sites + power model.

    The YAML config should look like:

        power_model: data/power_model_params.json  # or "default"
        sites:
          - name: US-West
            cell: data/cells/cell_a.csv
            solar: data/solar/the_dalles_or.csv
            price: data/prices/caiso.csv
            solar_capacity_mw: 50.0
            rated_power_mw: 100.0
          - name: US-Central
            ...
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

    # Load sites
    sites = []
    for site_cfg in config["sites"]:
        workload = load_csv_values(root_dir / site_cfg["cell"], "cpu_demand_norm")
        solar = load_csv_values(root_dir / site_cfg["solar"], "solar_fraction")
        price = load_csv_values(root_dir / site_cfg["price"], "price_usd_kwh")

        site = DataCenterSite(
            name=site_cfg["name"],
            workload=workload,
            solar=solar,
            price=price,
            solar_capacity_mw=site_cfg.get("solar_capacity_mw", 50.0),
            rated_power_mw=site_cfg.get("rated_power_mw", 100.0),
            capacity=site_cfg.get("capacity", 1.0),
        )
        sites.append(site)

    return sites, power_model
