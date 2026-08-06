"""No-training structural ablations for energy model v2 spatial headroom."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines import StatusQuoPolicy  # noqa: E402
from env.power_model import PowerModel  # noqa: E402
from evaluate import compute_summary, run_episode  # noqa: E402
from scripts.compute_qp_optimum import build_inputs, solve_clarabel  # noqa: E402
from scripts.review_energy_model_v2 import ALPHA, make_env  # noqa: E402

YEAR = 2025
STATUS_QUO = "Status Quo (local, no deferral)"
SCENARIOS = (
    ("US a-d", "us_model_v2_2025.yaml"),
    ("Global a-d", "global_model_v2_2025.yaml"),
)
CONDITIONS = {
    "primary": {
        "alpha": ALPHA,
        "pooled_power": False,
        "synchronous_market": False,
    },
    "energy_only": {
        "alpha": 0.0,
        "pooled_power": False,
        "synchronous_market": False,
    },
    "pooled_power": {
        "alpha": ALPHA,
        "pooled_power": True,
        "synchronous_market": False,
    },
    "synchronous_market": {
        "alpha": ALPHA,
        "pooled_power": False,
        "synchronous_market": True,
    },
    "shifted_market_only": {
        "alpha": 0.0,
        "pooled_power": True,
        "synchronous_market": False,
    },
    "power_heterogeneity_only": {
        "alpha": 0.0,
        "pooled_power": False,
        "synchronous_market": True,
    },
    "degenerate_control": {
        "alpha": 0.0,
        "pooled_power": True,
        "synchronous_market": True,
    },
}
RATED_POWER_VALUES = (50.0, 100.0, 200.0)


def configure_env(
    scenario: Path,
    *,
    alpha: float,
    pooled_power: bool,
    synchronous_market: bool,
    rated_power_mw: float = 100.0,
):
    env = make_env(scenario, False, alpha)
    first = env.sites[0]
    for site in env.sites:
        site.rated_power_mw = rated_power_mw
        if pooled_power:
            site.power_model = env.power_model
        if synchronous_market:
            site.price = first.price.copy()
            site.net_demand = first.net_demand.copy()
            site.net_demand_mw = first.net_demand_mw.copy()
            site.solar = first.solar.copy()
    return env


def configure_inputs(
    scenario: Path,
    *,
    alpha: float,
    pooled_power: bool,
    synchronous_market: bool,
    rated_power_mw: float = 100.0,
) -> dict[str, Any]:
    inp = build_inputs(
        str(scenario),
        False,
        peak_penalty_weight=alpha,
    )
    inp["R"][:] = rated_power_mw
    if pooled_power:
        pooled = PowerModel.from_json(ROOT / "data" / "power_model_params.json")
        inp["idle"][:] = pooled.idle_power
        inp["slope"][:] = pooled.slope
    if synchronous_market:
        inp["pi"][:] = inp["pi"][:, [0]]
        inp["d"][:] = inp["d"][:, [0]]
        inp["d_signed"][:] = inp["d_signed"][:, [0]]
    return inp


def score_condition(
    scenario: Path,
    *,
    alpha: float,
    pooled_power: bool,
    synchronous_market: bool,
    rated_power_mw: float = 100.0,
) -> dict[str, Any]:
    env = configure_env(
        scenario,
        alpha=alpha,
        pooled_power=pooled_power,
        synchronous_market=synchronous_market,
        rated_power_mw=rated_power_mw,
    )
    _, history = run_episode(
        env,
        StatusQuoPolicy().predict,
        is_sb3=False,
    )
    status_quo = compute_summary(history, batch_enabled=False)
    optimum = solve_clarabel(
        configure_inputs(
            scenario,
            alpha=alpha,
            pooled_power=pooled_power,
            synchronous_market=synchronous_market,
            rated_power_mw=rated_power_mw,
        ),
        False,
    )
    return {
        "status_quo": status_quo,
        "spatial_qp": optimum,
        "headroom_pct": (
            100.0
            * (status_quo["total_cost"] - optimum["optimum_total"])
            / status_quo["total_cost"]
        ),
    }


def build() -> dict[str, Any]:
    output = ROOT / "output" / "energy_model_v2" / str(YEAR)
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "year": YEAR,
        "scope": "a-d calibration cells only; no PPO training",
        "conditions": CONDITIONS,
        "rated_power_values_mw": list(RATED_POWER_VALUES),
        "scenarios": {},
        "rated_power_sensitivity": {},
    }

    for label, filename in SCENARIOS:
        scenario = ROOT / "env" / "scenarios" / filename
        result["scenarios"][label] = {
            name: score_condition(scenario, **condition)
            for name, condition in CONDITIONS.items()
        }
        result["rated_power_sensitivity"][label] = {
            str(rated_power): score_condition(
                scenario,
                alpha=ALPHA,
                pooled_power=False,
                synchronous_market=False,
                rated_power_mw=rated_power,
            )
            for rated_power in RATED_POWER_VALUES
        }

    (output / "structure_ablation.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    plot(result, output / "structure_ablation.png")
    write_report(result, output / "structure_ablation.md")
    return result


def plot(result: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = list(CONDITIONS)
    x = np.arange(len(labels))
    width = 0.36
    colors = ("#35637b", "#b94d2f")

    for offset, (scenario, entries) in enumerate(
        result["scenarios"].items()
    ):
        values = [entries[label]["headroom_pct"] for label in labels]
        axes[0].bar(
            x + (offset - 0.5) * width,
            values,
            width,
            label=scenario,
            color=colors[offset],
        )
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("Spatial QP headroom (%)")
    axes[0].set_title("Structural headroom ablations")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    for scenario, entries in result["rated_power_sensitivity"].items():
        powers = list(RATED_POWER_VALUES)
        values = [entries[str(power)]["headroom_pct"] for power in powers]
        axes[1].plot(powers, values, marker="o", label=scenario)
    axes[1].set_xlabel("Rated power per proxy DC (MW)")
    axes[1].set_ylabel("Spatial QP headroom (%)")
    axes[1].set_title("Fixed-alpha rated-power sensitivity")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        "Energy model v2 structural attribution (a-d, no training)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Energy Model v2 Structural Attribution",
        "",
        "> No PPO training is involved. Ablations use a-d calibration cells only.",
        "",
        "The ablations are diagnostic, not additive causal decompositions; "
        "interactions remain between price phase, power heterogeneity, and Φ.",
        "",
        "| Scenario | Condition | Status Quo | Spatial QP | Headroom |",
        "|---|---|---:|---:|---:|",
    ]
    for scenario, entries in result["scenarios"].items():
        for name, entry in entries.items():
            lines.append(
                f"| {scenario} | {name} | "
                f"${entry['status_quo']['total_cost'] / 1e6:.3f}M | "
                f"${entry['spatial_qp']['optimum_total'] / 1e6:.3f}M | "
                f"{entry['headroom_pct']:.2f}% |"
            )

    lines += [
        "",
        "## Rated-power sensitivity (primary objective, fixed alpha)",
        "",
        "| Scenario | R | Status Quo | Spatial QP | Headroom |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario, entries in result["rated_power_sensitivity"].items():
        for power, entry in entries.items():
            lines.append(
                f"| {scenario} | {float(power):.0f} MW | "
                f"${entry['status_quo']['total_cost'] / 1e6:.3f}M | "
                f"${entry['spatial_qp']['optimum_total'] / 1e6:.3f}M | "
                f"{entry['headroom_pct']:.2f}% |"
            )

    lines += ["", "![Structural ablation](structure_ablation.png)"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
