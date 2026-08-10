"""Generate deterministic publication figures from corrected V4R evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ramp_rl.provenance import (  # noqa: E402
    CANONICAL_JSON_REPRESENTATION,
    HASH_CONTRACT_ID,
)
from ramp_rl.v4r_corrected_thesis import (  # noqa: E402
    EXPECTED_CANONICAL_SHA256,
    EXPECTED_SEALED_CANONICAL_SHA256,
    load_verified_evidence,
)


OUTPUT = ROOT / "docs" / "figures" / "ramp_v6"
MARKET_ORDER = (
    "CAISO_NP15",
    "ERCOT_LZ_NORTH",
    "ISONE_NEMA",
    "MISO_MINN_HUB",
    "NYISO_NYC_J",
    "SPP_NORTH_HUB",
)
MARKET_LABELS = ("CAISO", "ERCOT", "ISO-NE", "MISO", "NYISO", "SPP")
COLORS = {
    "blue": "#2F6690",
    "green": "#2A7F62",
    "orange": "#D28C28",
    "red": "#B34A4A",
    "gray": "#707070",
    "light_blue": "#D9EAF7",
    "light_green": "#DDEFE7",
    "light_orange": "#F6E8CD",
    "light_gray": "#ECECEC",
}
PNG_METADATA = {
    "Software": "dc-energy-joint-optimization deterministic thesis builder"
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(
        OUTPUT / name,
        dpi=200,
        bbox_inches="tight",
        metadata=PNG_METADATA,
    )
    plt.close(fig)


def _box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    fill: str,
    edge: str = "#4A4A4A",
    fontsize: float = 9,
) -> None:
    axis.add_patch(
        plt.Rectangle(
            (x, y),
            width,
            height,
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.2,
            zorder=2,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=3,
        wrap=True,
    )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": "#4A4A4A", "lw": 1.3},
        zorder=1,
    )
    if label:
        axis.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.12,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#4A4A4A",
        )


def build_experiment_lineage(evidence: dict[str, Any]) -> None:
    stages = [
        ("V1", "screen promising;\nconfirmation failed", "failed"),
        ("V2", "two preregistered\nretries failed", "failed"),
        ("V3", "4/5 seeds pass;\nMISO non-harm fails", "failed"),
        ("V4", "model containers lost;\nnot evaluated", "blocked"),
        ("V4R", "recovered fixed ensemble;\nvalidation + one test", "passed"),
        (
            "Post-hoc",
            "frozen-policy telemetry\ncorrection; not new test",
            "posthoc",
        ),
    ]
    fill = {
        "failed": "#F4DADA",
        "blocked": COLORS["light_gray"],
        "passed": COLORS["light_green"],
        "posthoc": COLORS["light_orange"],
    }
    fig, axis = plt.subplots(figsize=(12, 3.8))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4)
    axis.axis("off")
    x_positions = np.linspace(0.25, 10.25, len(stages))
    for index, (name, detail, status) in enumerate(stages):
        x = float(x_positions[index])
        _box(
            axis,
            x,
            1.25,
            1.5,
            1.5,
            f"{name}\n{detail}",
            fill=fill[status],
            fontsize=8.3,
        )
        if index < len(stages) - 1:
            _arrow(axis, (x + 1.5, 2.0), (x_positions[index + 1], 2.0))
    axis.text(
        6,
        3.45,
        "Protocol lineage: failures and the operational block remain part of the evidence",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
    )
    axis.text(
        6,
        0.55,
        (
            "March-April sealed test opened once. The later replay changes "
            "telemetry only and preserves the original result."
        ),
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )
    _save(fig, "experiment_lineage.png")


def build_system_architecture() -> None:
    fig, axis = plt.subplots(figsize=(12, 6.5))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7)
    axis.axis("off")
    boxes = [
        (0.2, 5.3, 2.2, 1.0, "Market products\nprice, demand, wind, solar", COLORS["light_blue"]),
        (0.2, 3.7, 2.2, 1.0, "ClusterData2019\nservice + no-SLO batch", COLORS["light_blue"]),
        (0.2, 2.1, 2.2, 1.0, "PowerData2019\naffine site models", COLORS["light_blue"]),
        (3.0, 4.4, 2.2, 1.2, "Energy-v3 pipeline\nUTC alignment, hashes,\ncausal forecasts", COLORS["light_green"]),
        (5.8, 4.4, 2.2, 1.2, "Daily ramp environment\nstate, queue, reward,\nstatus quo", COLORS["light_green"]),
        (8.6, 5.2, 2.5, 1.0, "Five frozen PPO actors\nown normalizers", COLORS["light_orange"]),
        (8.6, 3.7, 2.5, 1.0, "Fixed equal-action mean\nweights = 0.2", COLORS["light_orange"]),
        (8.6, 2.2, 2.5, 1.0, "Constraint-only decoder\nsimplex + EDF + transport", COLORS["light_orange"]),
        (5.8, 1.0, 2.2, 1.0, "Executed work and power\nnext state + reward", COLORS["light_green"]),
        (3.0, 1.0, 2.2, 1.0, "Evaluation and provenance\nmetrics, hashes, gates", COLORS["light_green"]),
    ]
    for x, y, width, height, text, color in boxes:
        _box(axis, x, y, width, height, text, fill=color)
    _arrow(axis, (2.4, 5.8), (3.0, 5.0))
    _arrow(axis, (2.4, 4.2), (3.0, 4.8))
    _arrow(axis, (2.4, 2.6), (3.0, 4.55))
    _arrow(axis, (5.2, 5.0), (5.8, 5.0))
    _arrow(axis, (8.0, 5.0), (8.6, 5.7))
    _arrow(axis, (9.85, 5.2), (9.85, 4.7))
    _arrow(axis, (9.85, 3.7), (9.85, 3.2))
    _arrow(axis, (8.6, 2.7), (8.0, 1.5))
    _arrow(axis, (5.8, 1.5), (5.2, 1.5))
    _arrow(axis, (6.9, 2.0), (6.9, 4.4))
    axis.set_title(
        "V4R system architecture: learned preferences are separated from hard feasibility",
        fontsize=13,
        weight="bold",
    )
    _save(fig, "system_architecture.png")


def build_rl_loop() -> None:
    fig, axis = plt.subplots(figsize=(10.5, 6))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 7)
    axis.axis("off")
    _box(axis, 0.5, 4.7, 2.1, 1.1, "Causal state\nmarket + workload + queue", fill=COLORS["light_blue"])
    _box(axis, 3.2, 4.7, 2.1, 1.1, "PPO actor\n13 bounded preferences", fill=COLORS["light_orange"])
    _box(axis, 6.0, 4.7, 2.1, 1.1, "Constraint decoder\nfeasible allocations", fill=COLORS["light_green"])
    _box(axis, 6.0, 2.6, 2.1, 1.1, "Environment transition\npower, queue, next hour", fill=COLORS["light_green"])
    _box(axis, 3.2, 2.6, 2.1, 1.1, "Ramp reward\n+ soft cost penalty", fill=COLORS["light_blue"])
    _box(axis, 0.5, 2.6, 2.1, 1.1, "PPO critic\nexpected future return", fill=COLORS["light_orange"])
    _box(axis, 3.2, 0.5, 2.1, 1.0, "Training only:\nclipped actor/critic update", fill=COLORS["light_orange"])
    _arrow(axis, (2.6, 5.25), (3.2, 5.25))
    _arrow(axis, (5.3, 5.25), (6.0, 5.25))
    _arrow(axis, (7.05, 4.7), (7.05, 3.7))
    _arrow(axis, (6.0, 3.15), (5.3, 3.15))
    _arrow(axis, (3.2, 3.15), (2.6, 3.15))
    _arrow(axis, (1.55, 2.6), (1.55, 1.75))
    _arrow(axis, (2.6, 1.0), (3.2, 1.0))
    _arrow(axis, (5.3, 1.0), (7.9, 4.7), label="updated parameters")
    _arrow(axis, (6.0, 2.85), (2.6, 5.0), label="next state")
    axis.text(
        5,
        6.55,
        "Reward-only PPO learning loop",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
    )
    axis.text(
        5,
        0.05,
        "At inference, actor and normalizer are frozen; the update box is inactive.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    _save(fig, "rl_loop.png")


def build_provenance_flow() -> None:
    fig, axis = plt.subplots(figsize=(12, 5.2))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 5.5)
    axis.axis("off")
    labels = [
        ("Native/restricted\nsource files", COLORS["light_blue"]),
        ("Source contracts +\nraw SHA-256", COLORS["light_blue"]),
        ("Canonical panel +\nforecast manifest", COLORS["light_green"]),
        ("Daily factory windows +\ntraining-only stats", COLORS["light_green"]),
        (
            "Frozen identity +\ncheckpoint recovery verification",
            COLORS["light_orange"],
        ),
        ("Single-open sealed\nevidence chain", COLORS["light_orange"]),
        ("Post-hoc corrected\ntelemetry wrapper", COLORS["light_orange"]),
    ]
    x_positions = np.linspace(0.1, 10.35, len(labels))
    for index, ((label, color), x) in enumerate(zip(labels, x_positions, strict=True)):
        _box(axis, float(x), 2.15, 1.45, 1.15, label, fill=color, fontsize=8)
        if index < len(labels) - 1:
            _arrow(axis, (x + 1.45, 2.72), (x_positions[index + 1], 2.72))
    axis.text(
        6,
        4.65,
        "Backward provenance: every published metric points to immutable inputs",
        ha="center",
        fontsize=13,
        weight="bold",
    )
    axis.text(
        6,
        1.05,
        (
            "Raw redistribution follows operator terms. Committed manifests "
            "retain queries, coverage, hashes, and permissible derived metrics."
        ),
        ha="center",
        fontsize=9,
    )
    axis.text(
        6,
        0.35,
        (
            "The correction wrapper references rather than replaces the "
            "original sealed evidence and records that it is not a second test."
        ),
        ha="center",
        fontsize=9,
        color="#8A4F13",
    )
    _save(fig, "provenance_flow.png")


def build_primary_effect(evidence: dict[str, Any]) -> None:
    records = [
        ("February\nvalidation", evidence["validation"]["result"], COLORS["blue"]),
        ("March-April\nsealed test", evidence["test"]["result"], COLORS["green"]),
    ]
    means = [record[1]["mean_incremental_ramp_impact"] for record in records]
    lowers = [record[1]["bootstrap_by_day"]["lower_95"] for record in records]
    uppers = [record[1]["bootstrap_by_day"]["upper_95"] for record in records]
    errors = np.asarray(
        [
            [mean - lower for mean, lower in zip(means, lowers, strict=True)],
            [upper - mean for mean, upper in zip(means, uppers, strict=True)],
        ]
    )
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for index, (_, _, color) in enumerate(records):
        axis.errorbar(
            index,
            means[index],
            yerr=errors[:, index : index + 1],
            fmt="o",
            markersize=8,
            capsize=6,
            color=color,
            linewidth=2,
        )
    axis.axhline(0.0, color="#333333", linewidth=1)
    axis.set_xticks(range(len(records)), [record[0] for record in records])
    axis.set_ylabel("Mean native-relative incremental squared-ramp impact")
    axis.set_title("Primary effect with deterministic day-block 95% intervals")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axis.text(
        0.99,
        0.03,
        "Lower (more negative) is better",
        transform=axis.transAxes,
        ha="right",
        fontsize=9,
        color="#444444",
    )
    _save(fig, "primary_effect_ci.png")


def build_status_quo_comparison(evidence: dict[str, Any]) -> None:
    result = evidence["test"]["result"]
    native = result[
        "per_market_policy_native_relative_incremental_ramp_impact"
    ]
    comparison = result["status_quo_comparison"]["per_market"]
    deltas = [
        comparison[market][
            "policy_minus_status_quo_incremental_ramp_impact"
        ]
        for market in MARKET_ORDER
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    axes[0].barh(
        MARKET_LABELS,
        [native[market] for market in MARKET_ORDER],
        color=COLORS["green"],
    )
    axes[0].axvline(0.0, color="#333333", linewidth=0.9)
    axes[0].set_xlabel("Policy impact relative to native grid")
    axes[0].set_title("All six policy impacts are negative vs native")
    delta_colors = [
        COLORS["green"] if value < 0 else COLORS["red"] for value in deltas
    ]
    axes[1].barh(MARKET_LABELS, deltas, color=delta_colors)
    axes[1].axvline(0.0, color="#333333", linewidth=0.9)
    axes[1].set_xlabel("Policy minus status-quo impact")
    axes[1].set_title("Policy beats status quo in 5/6; MISO is worse")
    for axis in axes:
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle(
        (
            "Persisted March-April traces: native and status-quo baselines "
            "remain distinct"
        ),
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(
        OUTPUT / "status_quo_comparison.png",
        dpi=200,
        bbox_inches="tight",
        metadata=PNG_METADATA,
    )
    plt.close(fig)


def build_ramp_period_power(evidence: dict[str, Any]) -> None:
    audit = evidence["test"]["result"]["behavior_audit"]
    values = [audit["status_quo_ramp_power"], audit["policy_ramp_power"]]
    reduction = 100 * (1 - values[1] / values[0])
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    bars = axis.bar(
        ["Status quo", "V4R policy"],
        values,
        color=[COLORS["gray"], COLORS["green"]],
        width=0.6,
    )
    axis.set_ylabel("Persisted ramp-period power audit units")
    axis.set_title(f"Ramp-period power falls {reduction:.2f}% on sealed test")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axis.text(
        0.5,
        -0.18,
        "Repeated market-hour audit sum; not a single MW ramp.",
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
        color="#444444",
    )
    _save(fig, "ramp_period_power.png")


def build_physical_ramps(evidence: dict[str, Any]) -> None:
    records = [
        ("February replay", evidence["validation"]["result"], COLORS["blue"]),
        (
            "March-April replay",
            evidence["test"]["result"],
            COLORS["green"],
        ),
    ]
    metrics = [
        ("1 h p95", "abs_adjusted_ramp_h1_fraction_s_per_hour_p95"),
        ("1 h max", "abs_adjusted_ramp_h1_fraction_s_per_hour_max"),
        ("3 h p95", "abs_adjusted_ramp_h3_fraction_s_per_hour_p95"),
        ("3 h max", "abs_adjusted_ramp_h3_fraction_s_per_hour_max"),
    ]
    positions = np.arange(len(metrics), dtype=float)
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5))
    for offset, (label, result, color) in zip(
        (-width / 2, width / 2),
        records,
        strict=True,
    ):
        axis.bar(
            positions + offset,
            [result[key] for _, key in metrics],
            width,
            label=label,
            color=color,
        )
    axis.set_xticks(positions, [label for label, _ in metrics])
    axis.set_ylabel(
        "Absolute adjusted ramp\n(fraction of market training Q95 per hour)"
    )
    axis.set_title(
        "Corrected physical magnitudes pool every market-timestep absolute value"
    )
    axis.legend(frameon=False)
    axis.text(
        0.99,
        -0.16,
        (
            "Equivalence-bound post-hoc replay; original sealed traces cannot "
            "supply these values"
        ),
        transform=axis.transAxes,
        ha="right",
        fontsize=9,
        color="#8A4F13",
    )
    _save(fig, "physical_ramp_magnitudes.png")


def build_decoder_adjustment(evidence: dict[str, Any]) -> None:
    result = evidence["test"]["result"]
    values = np.asarray(
        [
            value
            for episode in result["policy_episodes"]
            for value in episode["semantic_adjustment_l2_values"]
        ],
        dtype=float,
    )
    adjustment = result["semantic_adjustment"]
    positive = values[values > 1e-12]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    if positive.size:
        axes[0].hist(
            positive,
            bins=min(30, max(8, int(np.sqrt(positive.size)))),
            color=COLORS["blue"],
            edgecolor="white",
        )
    else:
        axes[0].text(0.5, 0.5, "No positive adjustments", ha="center")
    axes[0].set_xlabel("Semantic adjustment L2")
    axes[0].set_ylabel("Decision count")
    axes[0].set_title(
        f"Positive adjustments ({adjustment['adjustment_rate']:.1%} of decisions)"
    )
    statistic_labels = ["Mean", "p95", "Maximum"]
    statistic_values = [
        adjustment["mean_l2"],
        adjustment["p95_l2"],
        adjustment["max_l2"],
    ]
    bars = axes[1].bar(
        statistic_labels,
        statistic_values,
        color=[COLORS["green"], COLORS["orange"], COLORS["red"]],
    )
    axes[1].set_ylabel("L2 in decoded compute-work coordinates")
    axes[1].set_title(
        f"Emergency fallback rate = {adjustment['emergency_fallback_rate']:.3%}"
    )
    for bar, value in zip(bars, statistic_values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.suptitle(
        (
            "Post-hoc replay: normal constraint projection is distinct from "
            "emergency intervention"
        ),
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(
        OUTPUT / "decoder_adjustment.png",
        dpi=200,
        bbox_inches="tight",
        metadata=PNG_METADATA,
    )
    plt.close(fig)


def build_cost_secondary(evidence: dict[str, Any]) -> None:
    result = evidence["test"]["result"]
    policy_cost = sum(
        float(episode["energy_cost"])
        for episode in result["policy_episodes"]
    )
    status_cost = sum(
        float(episode["energy_cost"])
        for episode in result["status_quo_episodes"]
    )
    saving = status_cost - policy_cost
    saving_pct = 100 * saving / status_cost
    fig, axis = plt.subplots(figsize=(8, 4.8))
    bars = axis.bar(
        ["Status quo", "V4R policy"],
        [status_cost / 1e6, policy_cost / 1e6],
        color=[COLORS["gray"], COLORS["green"]],
        width=0.6,
    )
    axis.set_ylabel("Modeled day-ahead cost (USD millions)")
    axis.set_title(
        "Secondary outcome: modeled wholesale cost, not the primary objective"
    )
    for bar, value in zip(bars, (status_cost, policy_cost), strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value / 1e6,
            f"${value / 1e6:.3f}M",
            ha="center",
            va="bottom",
        )
    axis.text(
        0.5,
        0.04,
        f"Difference: -USD {saving:,.0f} ({saving_pct:.3f}%), about "
        f"USD {saving / result['episode_count']:,.0f} per episode",
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
    )
    _save(fig, "cost_secondary.png")


def main() -> None:
    _style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evidence = load_verified_evidence()
    build_experiment_lineage(evidence)
    build_system_architecture()
    build_rl_loop()
    build_provenance_flow()
    build_primary_effect(evidence)
    build_status_quo_comparison(evidence)
    build_ramp_period_power(evidence)
    build_physical_ramps(evidence)
    build_decoder_adjustment(evidence)
    build_cost_secondary(evidence)
    figure_names = [
        "cost_secondary.png",
        "decoder_adjustment.png",
        "experiment_lineage.png",
        "physical_ramp_magnitudes.png",
        "primary_effect_ci.png",
        "provenance_flow.png",
        "ramp_period_power.png",
        "rl_loop.png",
        "status_quo_comparison.png",
        "system_architecture.png",
    ]
    manifest = {
        "schema_version": "ramp-v6-v4r-thesis-figures-v3",
        "canonical_evidence_sha256": EXPECTED_CANONICAL_SHA256,
        "sealed_canonical_evidence_sha256": (
            EXPECTED_SEALED_CANONICAL_SHA256
        ),
        "canonical_evidence_representation": (
            CANONICAL_JSON_REPRESENTATION
        ),
        "hash_contract_id": HASH_CONTRACT_ID,
        "files": {
            name: hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()
            for name in figure_names
        },
    }
    (OUTPUT / "v4r_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {len(figure_names)} figures under {OUTPUT}")


if __name__ == "__main__":
    main()
