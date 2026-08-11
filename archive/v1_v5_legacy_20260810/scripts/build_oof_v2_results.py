"""Build canonical tables and figures from the frozen v2 OOF summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent.parent
OOF_ROOT = ROOT / "output" / "oof_v2_2025"
SUMMARY_PATH = OOF_ROOT / "summary.json"

FOLDS = {
    "ad_to_eh": "a-d to e-h",
    "eh_to_ad": "e-h to a-d",
}
CONFIGS = {
    "us_spatial": "US spatial",
    "us_batch": "US joint",
    "global_spatial": "Global spatial",
    "global_batch": "Global joint",
}
STATUS_QUO = "Status Quo (local, no deferral)"


def fold_row(
    fold: str,
    config: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    summary = entry["summary"]
    status_quo = entry["baselines"][STATUS_QUO]
    seeds = list(entry["ppo_seeds"].values())
    qp = entry["qp"]["optimum_total"]
    qp_headroom = (
        100.0 * (status_quo["total_cost"] - qp) / status_quo["total_cost"]
    )
    savings = np.asarray(
        [seed["savings_vs_status_quo_pct"] for seed in seeds],
        dtype=float,
    )
    demand_charge = np.asarray(
        [seed["demand_charge_ref"] for seed in seeds],
        dtype=float,
    )
    peak = np.asarray([seed["peak_grid_mw"] for seed in seeds], dtype=float)
    t_margin = float(
        student_t.ppf(0.975, len(savings) - 1)
        * savings.std(ddof=1)
        / np.sqrt(len(savings))
    )

    ramp: dict[str, Any] = {}
    for horizon in ("1h", "3h"):
        baseline = status_quo["physical_ramp_metrics"][horizon]
        ppo_max = np.asarray(
            [
                seed["physical_ramp_metrics"][horizon]["max_up_delta_mw"]
                for seed in seeds
            ],
            dtype=float,
        )
        ppo_p95 = np.asarray(
            [
                seed["physical_ramp_metrics"][horizon]["p95_up_delta_mw"]
                for seed in seeds
            ],
            dtype=float,
        )
        ramp[horizon] = {
            "status_quo_max_delta_mw": baseline["max_up_delta_mw"],
            "ppo_mean_max_delta_mw": float(ppo_max.mean()),
            "ppo_minus_status_quo_max_delta_mw": float(
                ppo_max.mean() - baseline["max_up_delta_mw"]
            ),
            "status_quo_p95_delta_mw": baseline["p95_up_delta_mw"],
            "ppo_mean_p95_delta_mw": float(ppo_p95.mean()),
            "ppo_minus_status_quo_p95_delta_mw": float(
                ppo_p95.mean() - baseline["p95_up_delta_mw"]
            ),
        }

    batch_completion = np.asarray(
        [seed.get("batch_completion_fraction", 1.0) for seed in seeds],
        dtype=float,
    )
    return {
        "fold": fold,
        "fold_label": FOLDS[fold],
        "config": config,
        "config_label": CONFIGS[config],
        "status_quo_cost": status_quo["total_cost"],
        "ppo_mean_cost": summary["ppo_mean_cost"],
        "ppo_std_cost": summary["ppo_std_cost"],
        "ppo_mean_savings_pct": summary["ppo_mean_savings_pct"],
        "ppo_std_savings_pct": summary["ppo_std_savings_pct"],
        "ppo_savings_ci95": summary["ppo_mean_savings_ci95_optimizer"],
        "ppo_savings_t_ci95": [
            float(savings.mean() - t_margin),
            float(savings.mean() + t_margin),
        ],
        "ppo_min_savings_pct": summary["ppo_min_savings_pct"],
        "ppo_max_savings_pct": summary["ppo_max_savings_pct"],
        "positive_seed_count": int((savings > 0.0).sum()),
        "feasible_seed_count": int(
            sum(seed["feasible"] for seed in seeds)
        ),
        "service_completion_min": summary["minimum_service_completion"],
        "batch_completion_min": summary["minimum_batch_completion"],
        "batch_completion_mean": float(batch_completion.mean()),
        "batch_expired_total": float(
            sum(seed.get("total_batch_expired", 0.0) for seed in seeds)
        ),
        "terminal_batch_pool_max": float(
            max(seed.get("terminal_batch_pool", 0.0) for seed in seeds)
        ),
        "best_feasible_baseline": summary["best_feasible_baseline"],
        "best_feasible_baseline_cost": summary[
            "best_feasible_baseline_cost"
        ],
        "qp_optimum": qp,
        "qp_headroom_pct": qp_headroom,
        "ppo_gap_to_qp_pct": summary["ppo_gap_to_qp_pct"],
        "qp_savings_captured_pct": (
            100.0 * summary["ppo_mean_savings_pct"] / qp_headroom
            if qp_headroom > 0.0
            else 0.0
        ),
        "demand_charge_ref_status_quo": status_quo["demand_charge_ref"],
        "ppo_mean_demand_charge_ref": float(demand_charge.mean()),
        "demand_charge_change_pct": float(
            100.0
            * (demand_charge.mean() - status_quo["demand_charge_ref"])
            / status_quo["demand_charge_ref"]
        ),
        "fleet_peak_status_quo_mw": status_quo["peak_grid_mw"],
        "ppo_mean_fleet_peak_mw": float(peak.mean()),
        "fleet_peak_change_mw": float(
            peak.mean() - status_quo["peak_grid_mw"]
        ),
        "ramp": ramp,
        "seed_savings_pct": {
            seed: value["savings_vs_status_quo_pct"]
            for seed, value in entry["ppo_seeds"].items()
        },
    }


def build() -> dict[str, Any]:
    raw = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    rows = [
        fold_row(fold, config, raw["folds"][fold][config])
        for fold in FOLDS
        for config in CONFIGS
    ]

    temporal_comparison = {}
    for fold in FOLDS:
        temporal_comparison[fold] = {}
        for region in ("us", "global"):
            spatial = raw["folds"][fold][f"{region}_spatial"]["summary"][
                "ppo_mean_cost"
            ]
            joint = raw["folds"][fold][f"{region}_batch"]["summary"][
                "ppo_mean_cost"
            ]
            temporal_comparison[fold][region] = {
                "joint_minus_spatial_cost_pct": (
                    100.0 * (joint - spatial) / spatial
                ),
                "spatial_savings_pct": raw["folds"][fold][
                    f"{region}_spatial"
                ]["summary"]["ppo_mean_savings_pct"],
                "joint_savings_pct": raw["folds"][fold][
                    f"{region}_batch"
                ]["summary"]["ppo_mean_savings_pct"],
            }

    result = {
        "protocol": raw["protocol"],
        "headline_joint_shaping_supported": raw[
            "headline_joint_shaping_supported"
        ],
        "aggregate_configs": raw["configs"],
        "fold_rows": rows,
        "temporal_comparison": temporal_comparison,
        "primary_finding": (
            "Only Global spatial PPO satisfies the frozen positive-CI and "
            "feasibility criteria in both folds. The joint-shaping headline "
            "criterion fails."
        ),
    }
    (OOF_ROOT / "canonical_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    write_report(result)
    plot_savings(result)
    plot_qp(result)
    plot_secondary(result)
    plot_pipeline()
    return result


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# Frozen Energy-Model v2 OOF Results",
        "",
        "> 80 PPO models; two symmetric held-out workload folds; 10 optimizer "
        "seeds per configuration; no validation selection or post-hoc tuning.",
        "",
        "## Frozen verdict",
        "",
        f"**Joint-shaping headline supported: "
        f"{str(result['headline_joint_shaping_supported']).upper()}.**",
        "",
        result["primary_finding"],
        "",
        "## Held-out results",
        "",
        "| Fold | Config | Status quo | PPO mean +/- sd | Savings "
        "(95% optimizer CI) | Positive seeds | Feasible seeds | QP headroom | "
        "PPO gap to QP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["fold_rows"]:
        lines.append(
            f"| {row['fold_label']} | {row['config_label']} | "
            f"${row['status_quo_cost'] / 1e6:.3f}M | "
            f"${row['ppo_mean_cost'] / 1e6:.3f}M +/- "
            f"${row['ppo_std_cost'] / 1e6:.3f}M | "
            f"{row['ppo_mean_savings_pct']:.2f}% "
            f"[{row['ppo_savings_ci95'][0]:.2f}, "
            f"{row['ppo_savings_ci95'][1]:.2f}] | "
            f"{row['positive_seed_count']}/10 | "
            f"{row['feasible_seed_count']}/10 | "
            f"{row['qp_headroom_pct']:.2f}% | "
            f"{row['ppo_gap_to_qp_pct']:.2f}% |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **Global spatial is the only robust success:** 0.90% and 1.19% "
        "held-out mean savings, with positive optimizer CIs and complete service "
        "in both folds.",
        "- **US spatial does not establish savings:** a-d to e-h is consistent "
        "with a small loss (-0.29%, CI [-0.76, +0.15]) and its PPO mean also "
        "loses to Round Robin; e-h to a-d is indistinguishable from zero "
        "(+0.07%, CI [-0.27, +0.42]).",
        "- **Joint batch control is not established:** US is unstable and Global "
        "is fold-dependent; only 9/40 batch seeds (2 configs x 2 folds x 10 "
        "seeds) meet the frozen 99.99% completion floor, and no batch "
        "configuration reaches 4/10 feasible seeds in a fold.",
        "- **The learned policies capture little clairvoyant opportunity:** "
        "Global spatial captures 5.47% and 7.55% of QP savings; other "
        "configurations capture less or are negative.",
        "- **Joint batch control does not reliably improve over separately "
        "trained spatial PPO:** it is worse in both US folds and in Global a-d "
        "to e-h; it improves Global e-h to a-d by only 0.19%. Completion "
        "failures in 31/40 batch seeds confound any claim about pure temporal "
        "value.",
        "",
        "The frozen success rule uses a 20,000-resample percentile bootstrap "
        "over 10 optimizer seeds. As a wider small-sample sensitivity, the "
        "two Global spatial t-intervals are [0.24, 1.55]% and [0.71, 1.66]%; "
        "both remain positive, so the conclusion is unchanged.",
        "",
        "## Secondary effects",
        "",
        "| Fold | Config | Demand-charge change | Fleet-peak change | "
        "1h max-ramp change vs SQ | 3h max-ramp change vs SQ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["fold_rows"]:
        lines.append(
            f"| {row['fold_label']} | {row['config_label']} | "
            f"{row['demand_charge_change_pct']:+.2f}% | "
            f"{row['fleet_peak_change_mw']:+.2f} MW | "
            f"{row['ramp']['1h']['ppo_minus_status_quo_max_delta_mw']:+.2f} "
            f"MW | "
            f"{row['ramp']['3h']['ppo_minus_status_quo_max_delta_mw']:+.2f} "
            f"MW |"
        )
    lines += [
        "",
        "Spatial PPO generally lowers the secondary demand-charge reference "
        "(about 1.4-3.0%), while batch PPO raises it (about 5.3-8.1%). Batch "
        "PPO also worsens the rare maximum three-hour ramp across all sites in "
        "every fold, even "
        "though typical p95 three-hour ramps improve. Ramp rate was not in the "
        "training reward.",
        "",
        "## Completion audit",
        "",
        "All 80 policies complete 100% of service. Across 40 batch policies, "
        "9 meet the frozen 99.99% batch-completion floor. Global joint "
        "e-h to a-d seed 103 expires 0.889 normalized units. US joint a-d to "
        "e-h seed 101 has the largest terminal pool (7.690 units) and "
        "accumulates $2.289M of transient service-backlog cost despite clearing "
        "it by episode end.",
        "",
        "## Figures",
        "",
        "![Held-out savings](held_out_savings.png)",
        "",
        "![QP opportunity](qp_capture.png)",
        "",
        "![Secondary effects](secondary_effects.png)",
        "",
        "![End-to-end pipeline](end_to_end_pipeline.png)",
    ]
    (OOF_ROOT / "results_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def plot_savings(result: dict[str, Any]) -> None:
    rows = result["fold_rows"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, config in zip(axes.flat, CONFIGS):
        selected = [row for row in rows if row["config"] == config]
        for x, row in enumerate(selected):
            values = np.asarray(
                list(row["seed_savings_pct"].values()),
                dtype=float,
            )
            jitter = rng.uniform(-0.09, 0.09, len(values))
            ax.scatter(
                np.full(len(values), x) + jitter,
                values,
                alpha=0.65,
                s=28,
                color="#35637b",
            )
            lo, hi = row["ppo_savings_ci95"]
            ax.errorbar(
                x,
                row["ppo_mean_savings_pct"],
                yerr=[
                    [row["ppo_mean_savings_pct"] - lo],
                    [hi - row["ppo_mean_savings_pct"]],
                ],
                fmt="D",
                color="#b94d2f",
                capsize=5,
                linewidth=2,
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks([0, 1], [row["fold_label"] for row in selected])
        ax.set_title(CONFIGS[config])
        ax.set_ylabel("Savings vs held-out Status Quo (%)")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Frozen held-out PPO savings: seeds, mean, and 95% optimizer CI"
    )
    fig.tight_layout()
    fig.savefig(
        OOF_ROOT / "held_out_savings.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_qp(result: dict[str, Any]) -> None:
    rows = result["fold_rows"]
    labels = [
        f"{row['fold_label']}\n{row['config_label']}" for row in rows
    ]
    x = np.arange(len(rows))
    width = 0.38
    qp = [row["qp_headroom_pct"] for row in rows]
    ppo = [row["ppo_mean_savings_pct"] for row in rows]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x - width / 2, qp, width, label="Clairvoyant QP headroom",
           color="#b7a486")
    ax.bar(x + width / 2, ppo, width, label="PPO held-out mean savings",
           color="#35637b")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Savings vs Status Quo (%)")
    ax.set_title("Available opportunity versus learned held-out savings")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        OOF_ROOT / "qp_capture.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_secondary(result: dict[str, Any]) -> None:
    rows = result["fold_rows"]
    labels = [
        f"{row['fold_label']}\n{row['config_label']}" for row in rows
    ]
    x = np.arange(len(rows))
    series = (
        (
            "Demand-charge reference change (%)",
            [row["demand_charge_change_pct"] for row in rows],
        ),
        (
            "1h maximum ramp change vs Status Quo (MW)",
            [
                row["ramp"]["1h"][
                    "ppo_minus_status_quo_max_delta_mw"
                ]
                for row in rows
            ],
        ),
        (
            "3h maximum ramp change vs Status Quo (MW)",
            [
                row["ramp"]["3h"][
                    "ppo_minus_status_quo_max_delta_mw"
                ]
                for row in rows
            ],
        ),
    )
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, (title, values) in zip(axes, series):
        colors = ["#56775b" if value <= 0 else "#b94d2f" for value in values]
        ax.bar(x, values, color=colors)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x, labels, rotation=25, ha="right")
    fig.suptitle(
        "Secondary effects (negative is improvement versus Status Quo)"
    )
    fig.tight_layout()
    fig.savefig(
        OOF_ROOT / "secondary_effects.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")
    stages = [
        (
            0.3,
            5.2,
            2.6,
            1.65,
            "1. Measured workload",
            "ClusterData2019 instance_usage\n+ collection priority\n-> 5-minute service/batch curves",
            "#35637b",
        ),
        (
            3.2,
            5.2,
            2.6,
            1.65,
            "2. Physical calibration",
            "PowerData2019 -> per-cell idle/slope\nCAISO May-2025 -> signed net demand\nNP15 DAM -> real energy price",
            "#56775b",
        ),
        (
            6.1,
            5.2,
            2.6,
            1.65,
            "3. Controlled system",
            "4 equal 100 MW proxy DCs\nunit capacity; shifted local clocks\nmeasured arrivals + EDF pools",
            "#a66a12",
        ),
        (
            9.0,
            5.2,
            2.6,
            1.65,
            "4. PPO decision",
            "Observe service, current batch,\nqueue, price, signed demand, context\n-> route, release, place",
            "#b94d2f",
        ),
        (
            11.9,
            5.2,
            2.8,
            1.65,
            "5. Frozen training",
            "2 workload folds x 4 configs\nx 10 seeds = 80 models\n501,760 steps/model; gamma=1",
            "#7c4f65",
        ),
        (
            3.0,
            2.0,
            3.3,
            1.65,
            "6. Held-out evaluation",
            "Frozen policy on opposite cells\nStatus Quo / RR / immediate drain\nQP headroom; completion audit",
            "#35637b",
        ),
        (
            6.8,
            2.0,
            3.3,
            1.65,
            "7. Independent outcomes",
            "Cost + grid-stress components\nsecondary demand charge\n1h/3h physical ramp KPIs",
            "#56775b",
        ),
        (
            10.6,
            2.0,
            3.3,
            1.65,
            "8. Canonical thesis evidence",
            "One frozen summary and figures\nno seed/model post-selection\nclaims limited to one energy month",
            "#b94d2f",
        ),
    ]
    for x, y, w, h, title, body, color in stages:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=2,
            edgecolor=color,
            facecolor="#fffaf2",
        )
        ax.add_patch(box)
        ax.text(x + 0.15, y + h - 0.28, title, fontsize=11,
                fontweight="bold", color=color, va="top")
        ax.text(x + 0.15, y + h - 0.68, body, fontsize=9.2,
                color="#3c332d", va="top", linespacing=1.35)
    for left, right in zip(stages[:4], stages[1:5]):
        ax.annotate(
            "",
            xy=(right[0] - 0.08, right[1] + right[3] / 2),
            xytext=(left[0] + left[2] + 0.08, left[1] + left[3] / 2),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#74665c"),
        )
    ax.annotate(
        "",
        xy=(4.65, 3.75),
        xytext=(13.3, 5.1),
        arrowprops=dict(
            arrowstyle="->",
            lw=1.8,
            color="#74665c",
            connectionstyle="arc3,rad=-0.18",
        ),
    )
    for left, right in zip(stages[5:7], stages[6:8]):
        ax.annotate(
            "",
            xy=(right[0] - 0.08, right[1] + right[3] / 2),
            xytext=(left[0] + left[2] + 0.08, left[1] + left[3] / 2),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#74665c"),
        )
    ax.text(
        0.3,
        7.45,
        "From public traces to frozen held-out PPO evidence",
        fontsize=20,
        fontweight="bold",
        color="#2b211b",
    )
    ax.text(
        0.3,
        7.08,
        "Every arrow is a persisted, reproducible transformation; no v2 result "
        "is selected on the test fold.",
        fontsize=10.5,
        color="#74665c",
    )
    fig.tight_layout()
    fig.savefig(
        OOF_ROOT / "end_to_end_pipeline.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    build()
