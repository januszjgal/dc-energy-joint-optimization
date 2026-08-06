"""Load and validate frozen experiment protocol files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
V2_2025_PROTOCOL = ROOT / "env" / "protocols" / "v2_2025.yaml"


def load_protocol(path: Path = V2_2025_PROTOCOL) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    objective = protocol["objective"]
    batch = protocol["batch"]
    proxy = protocol["proxy_dc"]

    if proxy["capacity"] != 1.0 or proxy["rated_power_mw"] != 100.0:
        raise ValueError("v2 protocol requires unit-capacity 100 MW proxy DCs")
    if objective["gamma"] != 1.0:
        raise ValueError("v2 completion objective requires gamma=1")
    if not objective["enforce_batch_completion"]:
        raise ValueError("v2 protocol requires batch completion enforcement")
    if batch["primary_flexibility_factor"] < 0.0:
        raise ValueError("flexibility factor must be non-negative")
    oof = protocol["oof"]
    timesteps = oof["training_timesteps"]
    n_steps = oof["ppo"]["n_steps"]
    if timesteps <= 0 or timesteps % n_steps != 0:
        raise ValueError(
            "OOF training budget must be positive and rollout-aligned"
        )
    if not oof["ppo"]["domain_randomization"]:
        raise ValueError("v2 OOF protocol requires domain randomization")
    return protocol
