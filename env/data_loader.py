"""Load scenario configurations and build DataCenterSite objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from env.dc_site import DataCenterSite
from env.power_model import PowerModel
from env.workload_generator import BatchArrivalGenerator, load_cell_batch_config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_csv_values(path: Path, value_col: str) -> np.ndarray:
    """Load a CSV and return the value column as a numpy array."""
    df = pd.read_csv(path)
    return df[value_col].values.astype(np.float32)


def _load_machine_fleet(path: Path) -> dict[str, float]:
    """Load machine fleet data and return aggregate stats."""
    df = pd.read_csv(path)
    return {
        "cpu_total": float(df["cpu_capacity"].sum()),
        "memory_total": float(df["memory_capacity"].sum()),
        "machine_count": len(df),
    }


def load_scenario(
    config_path: Path,
    batch_enabled: bool = False,
    dynamic_arrivals: bool = True,
    seed: int = 42,
) -> tuple[list[DataCenterSite], PowerModel, dict[str, Any]]:
    """Load a scenario YAML config and return sites + power model + batch config.

    Args:
        config_path: Path to the scenario YAML file.
        batch_enabled: If True, load per-cell batch configs and set
            batch_fraction / batch_mean_duration_sec on each site.
        dynamic_arrivals: If True (and batch_enabled), create synthetic
            batch arrival generators from fitted distributions instead of
            using static batch_fraction splits.
        seed: Random seed for batch arrival generators.

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

    # First pass: load all sites and collect fleet data for normalization
    raw_fleet_data: list[dict[str, float] | None] = []
    sites_data: list[dict[str, Any]] = []

    for site_cfg in config["sites"]:
        workload = load_csv_values(root_dir / site_cfg["cell"], "cpu_demand_norm")
        solar = load_csv_values(root_dir / site_cfg["solar"], "solar_fraction")
        price = load_csv_values(root_dir / site_cfg["price"], "price_usd_kwh")

        # Truncate all arrays to the shortest one for this site
        min_len = min(len(workload), len(solar), len(price))
        workload = workload[:min_len]
        solar = solar[:min_len]
        price = price[:min_len]

        # Load machine fleet data if available
        machines_path = site_cfg.get("machines")
        fleet = None
        if machines_path:
            fleet = _load_machine_fleet(root_dir / machines_path)

        # Batch parameters
        batch_fraction = 0.0
        batch_mean_duration_sec = 2100.0
        memory_cpu_ratio = 0.7
        cell_cfg_obj = None

        if batch_enabled:
            batch_dist_path = site_cfg.get("batch_distributions")
            if batch_dist_path:
                cell_cfg_obj = load_cell_batch_config(root_dir / batch_dist_path)
                batch_fraction = cell_cfg_obj.batch_fraction
                batch_mean_duration_sec = cell_cfg_obj.mean_duration_sec
                memory_cpu_ratio = cell_cfg_obj.memory_cpu_ratio

        raw_fleet_data.append(fleet)
        sites_data.append({
            "site_cfg": site_cfg,
            "workload": workload,
            "solar": solar,
            "price": price,
            "batch_fraction": batch_fraction,
            "batch_mean_duration_sec": batch_mean_duration_sec,
            "memory_cpu_ratio": memory_cpu_ratio,
            "cell_cfg_obj": cell_cfg_obj,
            "fleet": fleet,
        })

    # Normalize fleet capacities: largest DC maps to 1.0
    max_cpu = max(
        (f["cpu_total"] for f in raw_fleet_data if f is not None),
        default=0,
    )
    max_mem = max(
        (f["memory_total"] for f in raw_fleet_data if f is not None),
        default=0,
    )

    # Second pass: build DataCenterSite objects
    sites = []
    for i, sd in enumerate(sites_data):
        site_cfg = sd["site_cfg"]
        fleet = sd["fleet"]

        # CPU and memory capacity (normalized)
        if fleet and max_cpu > 0:
            capacity = fleet["cpu_total"] / max_cpu
            memory_capacity = fleet["memory_total"] / max_mem if max_mem > 0 else 1.0
        else:
            capacity = site_cfg.get("capacity", 1.0)
            memory_capacity = site_cfg.get("memory_capacity", 1.0)

        site = DataCenterSite(
            name=site_cfg["name"],
            workload=sd["workload"],
            solar=sd["solar"],
            price=sd["price"],
            solar_capacity_mw=site_cfg.get("solar_capacity_mw", 50.0),
            rated_power_mw=site_cfg.get("rated_power_mw", 100.0),
            capacity=capacity,
            memory_capacity=memory_capacity,
            memory_cpu_ratio=sd["memory_cpu_ratio"],
            batch_fraction=sd["batch_fraction"],
            batch_mean_duration_sec=sd["batch_mean_duration_sec"],
            # Fleet info
            fleet_cpu_total=fleet["cpu_total"] if fleet else 0.0,
            fleet_memory_total=fleet["memory_total"] if fleet else 0.0,
            fleet_machine_count=fleet["machine_count"] if fleet else 0,
        )

        # Create batch arrival generator if enabled
        cell_cfg_obj = sd["cell_cfg_obj"]
        if (
            batch_enabled
            and dynamic_arrivals
            and cell_cfg_obj is not None
            and cell_cfg_obj.dist_config is not None
        ):
            # Compute expected total batch demand from static split
            # (used to normalize the generated arrivals to the right scale)
            target_total = float(
                sd["workload"].sum() * sd["batch_fraction"]
            )
            site.batch_generator = BatchArrivalGenerator(
                dist_config=cell_cfg_obj.dist_config,
                interval_seconds=batch_config.get("interval_seconds", 300),
                num_timesteps=site.num_timesteps,
                target_total_demand=target_total,
                seed=seed + i,
            )

        sites.append(site)

    return sites, power_model, batch_config
