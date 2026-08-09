"""Generate thesis figures from frozen design evidence and verified V4R results."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ramp_rl.v4r_thesis import load_verified_evidence  # noqa: E402


OUTPUT = ROOT / "docs" / "figures" / "ramp_v6"
V1_RESULTS = ROOT / "output" / "ramp_rl_v6" / "live" / "final_results.json"
SOURCE_CONTRACT = (
    ROOT / "data" / "energy_model_v3" / "provenance" / "source_contract.json"
)


def build_v1_closeout() -> None:
    payload = json.loads(V1_RESULTS.read_text(encoding="utf-8"))
    stages = ["100k screen", "500k confirmation"]
    impacts = [
        payload["screen"]["aggregate"]["mean_incremental_ramp_impact"],
        payload["confirmation"]["aggregate"]["mean_incremental_ramp_impact"],
    ]
    costs = [
        payload["screen"]["aggregate"]["mean_energy_cost_ratio"],
        payload["confirmation"]["aggregate"]["mean_energy_cost_ratio"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = ["#2f6f9f", "#b44d4d"]
    axes[0].bar(stages, impacts, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Mean incremental ramp impact")
    axes[0].set_title("Validation ramp metric (lower is better)")
    axes[0].tick_params(axis="x", rotation=12)
    axes[1].bar(stages, costs, color=colors)
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("DA energy cost / status quo")
    axes[1].set_title("Validation cost ratio")
    axes[1].tick_params(axis="x", rotation=12)
    fig.suptitle("Immutable v1 validation-only campaign closeout")
    fig.text(
        0.5,
        0.01,
        "Confirmation failed strict per-seed/per-market gates; sealed test unopened.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(OUTPUT / "v1_validation_closeout.png", dpi=180)
    plt.close(fig)


def build_six_market_design() -> None:
    contract = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
    ordered = [
        ("CAISO_NP15", "CAISO NP15", "cell a"),
        ("ERCOT_LZ_NORTH", "ERCOT North", "cell b"),
        ("NYISO_NYC_J", "NYISO Zone J", "cell c"),
        ("MISO_MINN_HUB", "MISO Minnesota", "cell d"),
        ("SPP_NORTH_HUB", "SPP North", "cell e"),
        ("ISONE_NEMA", "ISO-NE NEMA", "cell f"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    for index, (market, label, cell) in enumerate(ordered):
        y = 6.25 - index * 0.85
        descriptors = contract["sources"][market]
        price = descriptors[0]["source"]
        physical = descriptors[-1]["source"]
        ax.add_patch(
            plt.Rectangle((0.2, y - 0.28), 2.4, 0.56, color="#d9eaf7", ec="#376996")
        )
        ax.text(1.4, y, f"{label}\n{cell}", ha="center", va="center", fontsize=9)
        ax.annotate("", xy=(3.1, y), xytext=(2.6, y), arrowprops={"arrowstyle": "->"})
        ax.text(3.2, y + 0.12, f"Price: {price}", fontsize=7.5, va="center")
        ax.text(3.2, y - 0.12, f"Physical: {physical}", fontsize=7.5, va="center")
    ax.add_patch(
        plt.Rectangle((8.7, 1.2), 2.8, 4.9, color="#f2f2f2", ec="#555555")
    )
    ax.text(10.1, 5.65, "Common UTC hourly panel", ha="center", weight="bold")
    ax.text(10.1, 4.85, "Sep 2025-Jan 2026\nTRAIN", ha="center", va="center")
    ax.text(10.1, 3.55, "Feb 2026\nVALIDATION", ha="center", va="center")
    ax.text(10.1, 2.25, "Mar-Apr 2026\nSEALED TEST", ha="center", va="center")
    ax.text(
        6.0,
        0.45,
        "PJM DOM / Northern Virginia: credential-blocked, excluded, not evaluated",
        ha="center",
        color="#9b2c2c",
        weight="bold",
    )
    ax.set_title("Energy model v3: six independent evaluated markets and fixed split")
    fig.tight_layout()
    fig.savefig(OUTPUT / "six_market_study_design.png", dpi=180)
    plt.close(fig)


def build_protocol_progression(evidence: dict) -> None:
    labels = ["V1 screen", "V1 confirm", "V2-A", "V2-B", "V3", "V4", "V4R"]
    impacts = [
        -1.3349297807182322e-05,
        -1.09572077294e-05,
        -4.983717818021212e-07,
        -1.412252682718701e-06,
        -1.371091986086174e-05,
        0.0,
        evidence["validation"]["result"]["mean_incremental_ramp_impact"],
    ]
    colors = ["#567c9e", "#a64b4b", "#a64b4b", "#a64b4b", "#a64b4b", "#777777", "#2f7d4a"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars = ax.bar(labels, impacts, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Validation mean incremental ramp impact")
    ax.set_title("Protocol progression: failures remain visible; V4R is the first strict pass")
    ax.tick_params(axis="x", rotation=18)
    annotations = [
        "screen pass",
        "failed seeds/markets",
        "failed",
        "failed",
        "1 seed harmed MISO",
        "blocked\n(no evaluation)",
        "all gates pass",
    ]
    for bar, note in zip(bars, annotations, strict=True):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y - 7e-7 if y < 0 else y + 4e-7,
            note,
            ha="center",
            va="top" if y < 0 else "bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(OUTPUT / "protocol_progression.png", dpi=180)
    plt.close(fig)


def build_validation_test(evidence: dict) -> None:
    validation = evidence["validation"]["result"]
    test = evidence["test"]["result"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, label, result, color in (
        (axes[0], "February validation\n28 episodes", validation, "#376996"),
        (axes[1], "March-April sealed test\n60 episodes, one opening", test, "#2f7d4a"),
    ):
        ax.bar(
            ["Ramp impact", "Cost ratio - 1"],
            [result["mean_incremental_ramp_impact"], result["energy_cost_ratio"] - 1],
            color=[color, "#d49a32"],
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=12)
    axes[0].set_ylabel("Raw value (separate units; lower is favorable)")
    fig.suptitle("Validation and sealed test shown in separate panels, never pooled")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUTPUT / "v4r_validation_and_test_separate.png", dpi=180)
    plt.close(fig)


def build_per_market_test(evidence: dict) -> None:
    values = evidence["test"]["result"]["per_market_macro"]
    order = ["CAISO_NP15", "ERCOT_LZ_NORTH", "ISONE_NEMA", "MISO_MINN_HUB", "NYISO_NYC_J", "SPP_NORTH_HUB"]
    labels = ["CAISO", "ERCOT", "ISO-NE", "MISO", "NYISO", "SPP"]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.barh(labels, [values[key] for key in order], color="#2f7d4a")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean incremental normalized squared-ramp impact")
    ax.set_title("Sealed-test ramp impact is negative in every evaluated market")
    fig.tight_layout()
    fig.savefig(OUTPUT / "v4r_per_market_test.png", dpi=180)
    plt.close(fig)


def build_cost_ramp_and_robustness(evidence: dict) -> None:
    records = [
        ("Validation", evidence["validation"]["result"], "#376996"),
        ("Sealed test", evidence["test"]["result"], "#2f7d4a"),
        ("1 GW total", evidence["robustness"]["one_gw_total"]["result"], "#7b4aa0"),
        ("c-h overlapping", evidence["robustness"]["c_h_overlapping"]["result"], "#c87533"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for label, result, color in records:
        ax.scatter(
            100 * (result["energy_cost_ratio"] - 1),
            result["mean_incremental_ramp_impact"],
            s=90,
            label=label,
            color=color,
        )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Day-ahead modeled energy-cost change vs status quo (%)")
    ax.set_ylabel("Mean incremental normalized squared-ramp impact")
    ax.set_title("Ramp-cost relationship across prespecified and post-selection views")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "v4r_cost_ramp_relationship.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    labels = [item[0] for item in records[1:]]
    values = [item[1]["mean_incremental_ramp_impact"] for item in records[1:]]
    ax.bar(labels, values, color=[item[2] for item in records[1:]])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean incremental normalized squared-ramp impact")
    ax.set_title("Post-selection robustness (c-h is overlapping, not independent)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "v4r_robustness.png", dpi=180)
    plt.close(fig)


def build_behavior_and_physical_ramps(evidence: dict) -> None:
    records = [
        ("Validation", evidence["validation"]["result"]),
        ("Sealed test", evidence["test"]["result"]),
    ]
    x = range(len(records))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.35
    ax.bar(
        [value - width / 2 for value in x],
        [result["behavior_audit"]["status_quo_ramp_power"] for _, result in records],
        width,
        label="Status quo",
        color="#999999",
    )
    ax.bar(
        [value + width / 2 for value in x],
        [result["behavior_audit"]["policy_ramp_power"] for _, result in records],
        width,
        label="V4R policy",
        color="#2f7d4a",
    )
    ax.set_xticks(list(x), [label for label, _ in records])
    ax.set_ylabel("Ramp-period power audit (persisted aggregate units)")
    ax.set_title("V4R lowers ramp-period power while serving work before demand")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "v4r_behavior.png", dpi=180)
    plt.close(fig)

    metrics = [
        ("1 h p95", "ramp_h1_adjusted_p95"),
        ("1 h max", "ramp_h1_adjusted_max"),
        ("3 h p95", "ramp_h3_adjusted_p95"),
        ("3 h max", "ramp_h3_adjusted_max"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    width = 0.35
    positions = list(range(len(metrics)))
    ax.bar(
        [value - width / 2 for value in positions],
        [records[0][1][key] for _, key in metrics],
        width,
        label="Validation",
        color="#376996",
    )
    ax.bar(
        [value + width / 2 for value in positions],
        [records[1][1][key] for _, key in metrics],
        width,
        label="Sealed test",
        color="#2f7d4a",
    )
    ax.set_xticks(positions, [label for label, _ in metrics])
    ax.set_ylabel("Adjusted ramp (fraction of training Q95 gross-demand scale per hour)")
    ax.set_title("Interpretable 1 h and 3 h adjusted ramp extrema")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "v4r_physical_ramps.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evidence = load_verified_evidence()
    build_v1_closeout()
    build_six_market_design()
    build_protocol_progression(evidence)
    build_validation_test(evidence)
    build_per_market_test(evidence)
    build_cost_ramp_and_robustness(evidence)
    build_behavior_and_physical_ramps(evidence)
    figure_names = [
        "protocol_progression.png",
        "six_market_study_design.png",
        "v1_validation_closeout.png",
        "v4r_behavior.png",
        "v4r_cost_ramp_relationship.png",
        "v4r_per_market_test.png",
        "v4r_physical_ramps.png",
        "v4r_robustness.png",
        "v4r_validation_and_test_separate.png",
    ]
    manifest = {
        "schema_version": "ramp-v6-v4r-thesis-figures-v1",
        "canonical_evidence_sha256": evidence["_verified"]["canonical_sha256"],
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
    print(f"Wrote figures under {OUTPUT}")


if __name__ == "__main__":
    main()
