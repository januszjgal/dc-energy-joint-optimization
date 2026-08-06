"""Load scenario configurations and build DataCenterSite objects."""

from __future__ import annotations

import re
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
        dynamic_arrivals: If True (and batch_enabled), create a synthetic
            batch arrival generator only when the site has no measured
            ``tier_curves`` input. Measured service/batch curves always take
            precedence. A ``batch_distributions`` file can still provide the
            fitted mean duration used for deadlines without generating demand.
        seed: Random seed for batch arrival generators.

    Returns:
        Tuple of (sites, power_model, batch_config) where batch_config
        is a dict with environment-wide batch settings (empty when disabled).
    """
    root_dir = Path(__file__).resolve().parent.parent

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Power model (pooled fleet model; per-cell override below)
    pm_path = config.get("power_model", "default")
    if pm_path == "default":
        power_model = PowerModel.default()
    else:
        power_model = PowerModel.from_json(root_dir / pm_path)

    # Per-cell calibrated power models (R² 0.75-0.80 vs pooled 0.43; §3.2).
    # Enabled via `per_cell_power: true` in the scenario; each site is matched to
    # its cell's model by the cell letter in its workload path.
    per_cell_models: dict[str, PowerModel] = {}
    if config.get("per_cell_power", False) and pm_path != "default":
        per_cell_models = PowerModel.per_cell_from_json(root_dir / pm_path)

    batch_config: dict[str, Any] = config.get("batch", {}) if batch_enabled else {}

    # First pass: load timeseries and fleet info
    sites_data: list[dict[str, Any]] = []

    for site_cfg in config["sites"]:
        workload = load_csv_values(root_dir / site_cfg["cell"], "cpu_demand_norm")
        solar = load_csv_values(root_dir / site_cfg["solar"], "solar_fraction")
        price = load_csv_values(root_dir / site_cfg["price"], "price_usd_kwh")
        net_demand = load_csv_values(
            root_dir / site_cfg["net_demand"],
            site_cfg.get("net_demand_column", "net_demand_signed"),
        )
        net_demand_mw = load_csv_values(
            root_dir / site_cfg["net_demand"],
            "net_demand_mw",
        )

        # Optional measured per-tier curves (trace-derived service/batch split).
        # These are the complete primary demand input in batch mode; they bypass
        # synthetic arrival generation rather than validating or seeding it.
        service_curve = batch_curve = None
        tier_path = site_cfg.get("tier_curves")
        if batch_enabled and tier_path:
            service_curve = load_csv_values(root_dir / tier_path, "service_demand_norm")
            batch_curve = load_csv_values(root_dir / tier_path, "batch_demand_norm")

        # Truncate to the shortest series
        min_len = min(
            len(workload),
            len(solar),
            len(price),
            len(net_demand),
            len(net_demand_mw),
        )
        workload = workload[:min_len]
        solar = solar[:min_len]
        price = price[:min_len]
        net_demand = net_demand[:min_len]
        net_demand_mw = net_demand_mw[:min_len]
        if service_curve is not None:
            service_curve = service_curve[:min_len]
            batch_curve = batch_curve[:min_len]

        # Optional machine fleet for capacity calibration
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

        # Real tier curves define the effective batch fraction directly
        if batch_curve is not None and workload.sum() > 0:
            batch_fraction = float(batch_curve.sum() / workload.sum())

        sites_data.append(
            {
                "site_cfg": site_cfg,
                "workload": workload,
                "solar": solar,
                "price": price,
                "net_demand": net_demand,
                "net_demand_mw": net_demand_mw,
                "service_curve": service_curve,
                "batch_curve": batch_curve,
                "batch_fraction": batch_fraction,
                "batch_mean_duration_sec": batch_mean_duration_sec,
                "memory_cpu_ratio": memory_cpu_ratio,
                "cell_cfg_obj": cell_cfg_obj,
                "fleet": fleet,
            }
        )

    # Second pass: build sites
    sites = []
    for i, sd in enumerate(sites_data):
        site_cfg = sd["site_cfg"]
        fleet = sd["fleet"]

        # Workload/tier curves are already utilization fractions of each source
        # cell's own capacity. The v2 experiment maps every shape onto an equal
        # 100 MW proxy DC, so capacity is one in those same local-utilization
        # units. Raw machine totals remain provenance metadata only.
        capacity = float(site_cfg.get("capacity", 1.0))
        memory_capacity = float(site_cfg.get("memory_capacity", 1.0))
        if capacity <= 0.0 or memory_capacity <= 0.0:
            raise ValueError(
                f"{site_cfg['name']}: capacity and memory_capacity must "
                "be positive"
            )

        site = DataCenterSite(
            name=site_cfg["name"],
            workload=sd["workload"],
            solar=sd["solar"],
            price=sd["price"],
            net_demand=sd["net_demand"],
            net_demand_mw=sd["net_demand_mw"],
            rated_power_mw=site_cfg.get("rated_power_mw", 100.0),
            capacity=capacity,
            memory_capacity=memory_capacity,
            memory_cpu_ratio=sd["memory_cpu_ratio"],
            batch_fraction=sd["batch_fraction"],
            batch_mean_duration_sec=sd["batch_mean_duration_sec"],
            fleet_cpu_total=fleet["cpu_total"] if fleet else 0.0,
            fleet_memory_total=fleet["memory_total"] if fleet else 0.0,
            fleet_machine_count=fleet["machine_count"] if fleet else 0,
        )

        # Real per-tier curves take precedence over the synthetic generator
        if sd["batch_curve"] is not None:
            site.service_curve = sd["service_curve"]
            site.batch_curve = sd["batch_curve"]

        # Per-cell calibrated power model, matched by the cell letter in the
        # site's workload path (e.g. "data/cells/cell_a.csv" -> "a")
        if per_cell_models:
            m = re.search(r"cell_([a-z])", str(site_cfg["cell"]))
            if m and m.group(1) in per_cell_models:
                site.power_model = per_cell_models[m.group(1)]

        cell_cfg_obj = sd["cell_cfg_obj"]
        if (
            batch_enabled
            and dynamic_arrivals
            and sd["batch_curve"] is None
            and cell_cfg_obj is not None
            and cell_cfg_obj.dist_config is not None
        ):
            target_total = float(sd["workload"].sum() * sd["batch_fraction"])
            site.batch_generator = BatchArrivalGenerator(
                dist_config=cell_cfg_obj.dist_config,
                interval_seconds=batch_config.get("interval_seconds", 300),
                num_timesteps=site.num_timesteps,
                target_total_demand=target_total,
                seed=seed + i,
            )

        sites.append(site)

    return sites, power_model, batch_config
