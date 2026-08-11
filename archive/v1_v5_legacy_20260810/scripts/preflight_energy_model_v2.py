"""Validate frozen invariants before any energy-model v2 training."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines import StatusQuoPolicy  # noqa: E402
from env.protocol import load_protocol  # noqa: E402
from evaluate import compute_summary, run_episode  # noqa: E402
from scripts.review_energy_model_v2 import make_env  # noqa: E402

YEAR = 2025
EXPECTED_STEPS = 8_928
SCENARIOS = (
    "us_model_v2_2025.yaml",
    "us_model_eh_v2_2025.yaml",
    "global_model_v2_2025.yaml",
    "global_model_eh_v2_2025.yaml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status_quo_summary(scenario: Path, batch: bool) -> dict:
    env = make_env(scenario, batch)
    _, history = run_episode(
        env,
        StatusQuoPolicy().predict,
        is_sb3=False,
    )
    return compute_summary(history, batch_enabled=batch)


def check_manifest() -> None:
    protocol = load_protocol()
    assert protocol["proxy_dc"]["capacity"] == 1.0
    assert protocol["batch"]["current_arrival_observed"]
    assert protocol["net_demand"]["column"] == "net_demand_signed"
    assert not protocol["ramp"]["direct_penalty_in_primary_reward"]
    assert protocol["ramp"]["evaluation_horizons_steps"] == {
        "1h": 12,
        "3h": 36,
    }
    assert protocol["routing"]["service_unrestricted"]
    assert not protocol["routing"]["movement_cost_modeled"]
    assert set(protocol["attribution"]["claims"]) == {
        "shifted_market_phase",
        "power_heterogeneity",
        "alpha",
        "rated_power",
    }
    assert len(protocol["oof"]["folds"]) == 2
    assert len(protocol["oof"]["seeds"]) == 10
    assert protocol["oof"]["training_timesteps"] == 501_760
    assert (
        protocol["oof"]["training_timesteps"]
        % protocol["oof"]["ppo"]["n_steps"]
        == 0
    )
    assert (ROOT / "scripts" / "run_oof_campaign_v2.py").exists()
    manifest_path = (
        ROOT / "data" / "energy_model_v2" / str(YEAR) / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_path = ROOT / manifest["protocol"]["path"]
    assert sha256(protocol_path) == manifest["protocol"]["sha256"]
    calendar = manifest["calendar"]
    assert calendar["rows"] == EXPECTED_STEPS
    assert calendar["cadence"] == "5 minutes"

    for relative, expected in manifest["processed"].items():
        path = ROOT / relative
        actual = sha256(path)
        assert actual == expected, f"hash mismatch: {relative}"


def check_scenario(filename: str) -> None:
    path = ROOT / "env" / "scenarios" / filename
    spatial = make_env(path, False)
    batch = make_env(path, True)

    assert spatial.max_steps == EXPECTED_STEPS
    assert batch.max_steps == EXPECTED_STEPS
    assert spatial.n_dc == batch.n_dc == 4
    assert spatial.action_space.shape == (4,)
    assert batch.action_space.shape == (12,)
    assert batch.observation_space.shape == (55,)

    obs, _ = batch.reset(seed=42)
    per_site_dims = (len(obs) - 3) // batch.n_dc
    for site in batch.sites:
        assert site.capacity == 1.0
        assert site.memory_capacity == 1.0
        assert site.service_curve is not None
        assert site.batch_curve is not None
        assert site.batch_generator is None
        aggregate = site.service_curve + site.batch_curve
        assert np.allclose(
            aggregate,
            site.workload,
            rtol=0.0,
            atol=1e-6,
        )
    for i, site in enumerate(batch.sites):
        observed_arrival = obs[i * per_site_dims + 1]
        assert np.isclose(
            observed_arrival,
            site.get_batch_demand(0),
            rtol=0.0,
            atol=1e-7,
        )

    spatial_sq = status_quo_summary(path, False)
    batch_sq = status_quo_summary(path, True)
    cost_delta = abs(
        batch_sq["total_cost"] - spatial_sq["total_cost"]
    )
    assert cost_delta < 0.01, (
        f"{filename}: spatial/batch Status Quo cost delta={cost_delta}"
    )
    assert batch_sq["work_completed_fraction"] > 1.0 - 1e-9
    assert batch_sq["total_batch_expired"] < 1e-9
    assert batch_sq["terminal_batch_pool"] < 1e-8
    assert set(batch_sq["physical_ramp_metrics"]) == {"1h", "3h"}

    print(
        f"PASS {filename}: {EXPECTED_STEPS} steps, measured tiers, "
        f"Status Quo delta=${cost_delta:.6f}"
    )


def check_signed_net_demand() -> None:
    pacific = pd.read_csv(
        ROOT
        / "data"
        / "energy_model_v2"
        / str(YEAR)
        / "processed"
        / "us_pacific.csv"
    )
    negative = pacific["net_demand_mw"] < 0.0
    assert int((pacific.loc[negative, "net_demand_signed"] >= 0.0).sum()) == 0
    assert pacific["net_demand_signed"].between(-1.0, 1.0).all()
    print(
        "PASS signed net demand: "
        f"{int(negative.sum())} negative intervals remain negative."
    )


def main() -> None:
    failures: list[str] = []
    try:
        check_manifest()
    except (AssertionError, FileNotFoundError, KeyError) as exc:
        failures.append(f"manifest: {exc}")
    for scenario in SCENARIOS:
        try:
            check_scenario(scenario)
        except (AssertionError, FileNotFoundError, KeyError) as exc:
            failures.append(f"{scenario}: {exc}")
    check_signed_net_demand()
    if failures:
        print("\nPRETRAINING GATE BLOCKED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("Energy-model v2 invariant preflight passed.")


if __name__ == "__main__":
    main()
