"""Shared deterministic fixture helpers."""

from __future__ import annotations

from pathlib import Path

from env.ramp_v6.environment import RampAwareEnv
from env.ramp_v6.fixture import load_fixture


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "ramp_v6"


def make_env(scale_multiplier: float = 1.0) -> RampAwareEnv:
    panel, sites, workload, stats, protocol = load_fixture(
        FIXTURE_ROOT, scale_multiplier=scale_multiplier
    )
    return RampAwareEnv(panel, sites, workload, stats, protocol)
