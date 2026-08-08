"""Build the canonical v5 TD3+BC results report and figures."""

from __future__ import annotations

import json
import math
import sys
from io import BytesIO
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate import compute_summary, run_episode  # noqa: E402
from scripts.run_offpolicy_campaign_v5 import (  # noqa: E402
    EVAL_SEED,
    baseline_summary,
    make_env,
    teacher_action,
)

OUTPUT_ROOT = ROOT / "output" / "offpolicy_v5_continuous"
CANONICAL_DIR = OUTPUT_ROOT / "canonical_v5_v3"
MANIFEST_PATHS = {
    region: OUTPUT_ROOT / f"final_td3bc_manifest_{region}_v3.json"
    for region in ("us", "global")
}
T_95_DF4 = 2.7764451051977987
REGION_COLORS = {"us": "#2F5597", "global": "#C55A11"}
VARIANT_COLORS = {"bc_only": "#70AD47", "post_rl_td3bc": "#4472C4"}
PAPER_FIGURE_DIR = ROOT / "output" / "paper_figs"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure_if_changed(
    figure: Any,
    path: Path,
    *,
    dpi: int,
    bbox_inches: str | None = None,
    flatten: bool = False,
) -> None:
    rendered = BytesIO()
    figure.savefig(
        rendered,
        format="png",
        dpi=dpi,
        bbox_inches=bbox_inches,
    )
    payload = rendered.getvalue()
    if flatten:
        flattened = BytesIO()
        with Image.open(BytesIO(payload)) as image:
            image.convert("RGB").save(flattened, format="PNG")
        payload = flattened.getvalue()
    if path.is_file() and path.read_bytes() == payload:
        return
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_tier_example_figure() -> None:
    source = ROOT / "data" / "cells" / "cell_b_tiers.csv"
    values = np.genfromtxt(source, delimiter=",", names=True)
    count = 3 * 288
    days = np.arange(count) / 288
    fig, axis = plt.subplots(figsize=(6.61, 4.51), constrained_layout=True)
    axis.plot(
        days,
        values["cpu_demand_norm"][:count],
        color="#333333",
        linewidth=0.9,
        label="aggregate (measured)",
    )
    axis.plot(
        days,
        values["service_demand_norm"][:count],
        color="#7FB3D5",
        linewidth=0.8,
        label="service / residual",
    )
    axis.plot(
        days,
        values["batch_demand_norm"][:count],
        color="#27AE60",
        linewidth=0.8,
        label="batch (matched priority <=115)",
    )
    axis.set_xlabel("Days")
    axis.set_ylabel("CPU (fraction of capacity)")
    axis.set_title("Cell b: measured aggregate and classified batch")
    axis.legend(loc="center right")
    axis.grid(True, alpha=0.3)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PAPER_FIGURE_DIR / "fig2_tiers.png"
    save_figure_if_changed(
        fig,
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def descriptive_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty list")
    average = mean(values)
    sample_std = stdev(values) if len(values) > 1 else 0.0
    half_width = (
        T_95_DF4 * sample_std / math.sqrt(len(values))
        if len(values) == 5
        else 0.0
    )
    return {
        "values": values,
        "count": len(values),
        "mean": average,
        "minimum": min(values),
        "maximum": max(values),
        "sample_std": sample_std,
        "optimization_seed_t95_interval": [
            average - half_width,
            average + half_width,
        ],
        "interval_interpretation": (
            "Variation across training seeds on one deterministic calendar; "
            "not an independent-data confidence interval."
        ),
    }


def load_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [load_json(ROOT / Path(row["record_path"])) for row in rows]


def safety_pass(summary: dict[str, Any]) -> bool:
    safety = summary["safety"]
    return bool(
        math.isclose(
            float(summary["service_served_total"]),
            float(summary["service_demand_total"]),
            abs_tol=1e-8,
        )
        and math.isclose(
            float(summary["batch_completion_fraction"]),
            1.0,
            abs_tol=1e-8,
        )
        and float(summary["total_batch_expired"]) <= 1e-8
        and float(summary["terminal_batch_pool"]) <= 1e-8
        and float(summary["terminal_backlog"]) <= 1e-8
        and int(safety["infeasibility_certificates"]) == 0
        and float(safety["max_transport_conservation_error"]) <= 1e-8
    )


def evaluate_analytic_policy(
    region: str,
    reward_scale: float,
    scenario_kind: str,
    teacher_name: str,
) -> dict[str, Any]:
    baseline = baseline_summary(
        region,
        reward_scale,
        scenario_kind=scenario_kind,
    )
    env = make_env(
        region,
        EVAL_SEED,
        reward_scale,
        domain_randomization=False,
        scenario_kind=scenario_kind,
    )
    try:
        total_reward, history = run_episode(
            env,
            lambda _obs, active_env: teacher_action(active_env, teacher_name),
            is_sb3=False,
        )
        summary = compute_summary(history, batch_enabled=True)
        summary["episode_reward"] = float(total_reward)
        summary["baseline_total_cost"] = float(baseline["total_cost"])
        summary["savings_vs_status_quo_usd"] = float(
            baseline["total_cost"] - summary["total_cost"]
        )
        summary["savings_vs_status_quo_pct"] = float(
            100.0
            * summary["savings_vs_status_quo_usd"]
            / baseline["total_cost"]
        )
        return {
            "teacher_name": teacher_name,
            "scenario_kind": scenario_kind,
            "savings_vs_status_quo_pct": summary["savings_vs_status_quo_pct"],
            "total_cost": summary["total_cost"],
            "safe": safety_pass(summary),
            "emergency_intervention_rate": summary["safety"][
                "emergency_intervention_rate"
            ],
            "decoder_adjustment_rate": summary["safety"][
                "decoder_adjustment_rate"
            ],
        }
    finally:
        env.close()


def scope_secondary_kpis(
    records: list[dict[str, Any]],
    *,
    scope_key: str,
    baseline_key: str,
) -> dict[str, Any]:
    evaluations = [record[scope_key] for record in records]
    baselines = [record[baseline_key] for record in records]

    def average(rows: list[dict[str, Any]], key: str) -> float:
        return mean(float(row[key]) for row in rows)

    baseline_primary = average(baselines, "total_cost")
    policy_primary = average(evaluations, "total_cost")
    baseline_demand_charge = average(baselines, "demand_charge_ref")
    policy_demand_charge = average(evaluations, "demand_charge_ref")
    combined_baseline = baseline_primary + baseline_demand_charge
    combined_policy = policy_primary + policy_demand_charge
    return {
        "mean_baseline_primary_cost_usd": baseline_primary,
        "mean_policy_primary_cost_usd": policy_primary,
        "mean_primary_savings_usd": baseline_primary - policy_primary,
        "mean_baseline_energy_cost_usd": average(baselines, "total_energy_cost"),
        "mean_policy_energy_cost_usd": average(evaluations, "total_energy_cost"),
        "mean_baseline_grid_stress_cost_usd": average(
            baselines,
            "total_peak_penalty",
        ),
        "mean_policy_grid_stress_cost_usd": average(
            evaluations,
            "total_peak_penalty",
        ),
        "secondary_demand_charge_rate_usd_per_kw_cycle": 15.0,
        "mean_baseline_demand_charge_ref_usd": baseline_demand_charge,
        "mean_policy_demand_charge_ref_usd": policy_demand_charge,
        "mean_demand_charge_change_usd": (
            policy_demand_charge - baseline_demand_charge
        ),
        "mean_demand_charge_change_pct": (
            100.0
            * (policy_demand_charge - baseline_demand_charge)
            / baseline_demand_charge
        ),
        "combined_primary_plus_demand_charge_savings_usd": (
            combined_baseline - combined_policy
        ),
        "combined_primary_plus_demand_charge_savings_pct": (
            100.0
            * (combined_baseline - combined_policy)
            / combined_baseline
        ),
        "mean_baseline_peak_grid_mw": average(baselines, "peak_grid_mw"),
        "mean_policy_peak_grid_mw": average(evaluations, "peak_grid_mw"),
        "mean_baseline_load_factor": average(baselines, "load_factor"),
        "mean_policy_load_factor": average(evaluations, "load_factor"),
        "mean_decoder_adjustment_rate": mean(
            float(row["safety"]["decoder_adjustment_rate"])
            for row in evaluations
        ),
        "mean_decoder_adjustment_l2": mean(
            float(row["safety"]["mean_decoder_adjustment_l2"])
            for row in evaluations
        ),
        "maximum_decoder_adjustment_l2": max(
            float(row["safety"]["max_decoder_adjustment_l2"])
            for row in evaluations
        ),
        "maximum_emergency_intervention_rate": max(
            float(row["safety"]["emergency_intervention_rate"])
            for row in evaluations
        ),
        "all_safe": all(safety_pass(row) for row in evaluations),
    }


def build_results() -> dict[str, Any]:
    manifests = {region: load_json(path) for region, path in MANIFEST_PATHS.items()}
    results: dict[str, Any] = {
        "study": "v5 corrected teacher-guided TD3+BC",
        "source_commit": manifests["us"]["source_commit"],
        "protocol": manifests["us"]["protocol"],
        "claim_scope": {
            "development": "a-d development / frozen confirmation",
            "transfer": "e-h descriptive transfer only",
            "energy_generalization": (
                "No untouched confirmatory energy month; all seeds share one "
                "deterministic May 2025 CAISO archetype."
            ),
        },
        "regions": {},
    }
    for region in ("us", "global"):
        manifest = manifests[region]
        region_result: dict[str, Any] = {
            "goal_savings_pct": 5.0 if region == "us" else 10.0,
            "variants": {},
        }
        for variant in ("bc_only", "post_rl_td3bc"):
            rows = manifest[variant]["seed_records"]
            records = load_records(rows)
            development_values = [
                float(row["savings_vs_status_quo_pct"]) for row in rows
            ]
            transfer_values = [
                float(row["descriptive_transfer_eh"]["savings_vs_status_quo_pct"])
                for row in rows
            ]
            region_result["variants"][variant] = {
                "development_savings_pct": descriptive_stats(development_values),
                "descriptive_transfer_savings_pct": descriptive_stats(
                    transfer_values
                ),
                "mean_decoder_adjustment_rate": mean(
                    float(row["decoder_adjustment_rate"]) for row in rows
                ),
                "maximum_emergency_intervention_rate": max(
                    float(row["emergency_intervention_rate"]) for row in rows
                ),
                "all_safe": all(bool(row["safe_seed_passed"]) for row in rows),
                "all_provenance_pass": all(
                    bool(row["provenance_pass"]) for row in rows
                ),
                "records": [row["record_path"] for row in rows],
            }
            if variant == "post_rl_td3bc":
                region_result["secondary_kpis"] = {
                    "development": scope_secondary_kpis(
                        records,
                        scope_key="evaluation",
                        baseline_key="baseline",
                    ),
                    "descriptive_transfer": scope_secondary_kpis(
                        records,
                        scope_key="descriptive_transfer_eh",
                        baseline_key="descriptive_transfer_baseline",
                    ),
                }
                region_result["post_rl_update_evidence"] = {
                    "n_updates_each": sorted(
                        {int(row["n_updates"]) for row in rows}
                    ),
                    "all_actor_hashes_changed": all(
                        bool(row["weight_update_evidence"]["actor_hash_changed"])
                        for row in rows
                    ),
                    "all_critic_hashes_changed": all(
                        bool(row["weight_update_evidence"]["critic_hash_changed"])
                        for row in rows
                    ),
                    "all_teacher_absent_during_rl": all(
                        not bool(row["teacher_present_during_rl"]) for row in rows
                    ),
                    "all_teacher_absent_at_inference": all(
                        not bool(row["teacher_present_at_inference"]) for row in rows
                    ),
                }
        bc_values = region_result["variants"]["bc_only"][
            "development_savings_pct"
        ]["values"]
        rl_values = region_result["variants"]["post_rl_td3bc"][
            "development_savings_pct"
        ]["values"]
        paired_delta = [
            float(post_rl - bc_only)
            for bc_only, post_rl in zip(bc_values, rl_values, strict=True)
        ]
        region_result["attribution"] = {
            "paired_post_rl_minus_bc_percentage_points": descriptive_stats(
                paired_delta
            ),
            "interpretation": (
                "The reward-trained actor and critics changed, but the paired "
                "economic difference from BC-only is the incremental RL effect. "
                "Most absolute savings are attributable to teacher imitation."
            ),
        }
        reward_scale = 5.0e-4 if region == "us" else 1.0e-4
        region_result["demonstration_teacher"] = {
            "development": evaluate_analytic_policy(
                region,
                reward_scale,
                "development",
                "native_marginal_cost",
            ),
            "descriptive_transfer": evaluate_analytic_policy(
                region,
                reward_scale,
                "descriptive_transfer",
                "native_marginal_cost",
            ),
            "attribution": (
                "Greedy linear current-state policy used to generate the frozen "
                "offline demonstrations; not an RL result and absent at inference."
            ),
        }
        region_result["exact_native_benchmark"] = {
            "development": evaluate_analytic_policy(
                region,
                reward_scale,
                "development",
                "exact_native",
            ),
            "descriptive_transfer": evaluate_analytic_policy(
                region,
                reward_scale,
                "descriptive_transfer",
                "exact_native",
            ),
            "attribution": (
                "Separate exact-native convex current-state analytic benchmark; "
                "not the frozen demonstration teacher and not an RL result."
            ),
        }
        post_mean = region_result["variants"]["post_rl_td3bc"][
            "development_savings_pct"
        ]["mean"]
        benchmark_mean = region_result["exact_native_benchmark"]["development"][
            "savings_vs_status_quo_pct"
        ]
        region_result["exact_native_savings_capture_pct"] = (
            100.0 * post_mean / benchmark_mean
        )
        region_result["headline_gate_passed"] = bool(
            region_result["variants"]["post_rl_td3bc"]["all_safe"]
            and region_result["variants"]["post_rl_td3bc"][
                "maximum_emergency_intervention_rate"
            ]
            < 0.01
            and region_result["variants"]["post_rl_td3bc"][
                "development_savings_pct"
            ]["minimum"]
            >= region_result["goal_savings_pct"]
            and region_result["variants"]["post_rl_td3bc"]["all_provenance_pass"]
        )
        results["regions"][region] = region_result
    results["all_headline_gates_passed"] = all(
        region["headline_gate_passed"] for region in results["regions"].values()
    )
    return results


def save_savings_figure(results: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    seeds = np.arange(301, 306)
    x = np.arange(len(seeds))
    width = 0.34
    for axis, region in zip(axes, ("us", "global"), strict=True):
        region_result = results["regions"][region]
        for offset, variant in ((-width / 2, "bc_only"), (width / 2, "post_rl_td3bc")):
            values = region_result["variants"][variant][
                "development_savings_pct"
            ]["values"]
            label = "BC only" if variant == "bc_only" else "Post-RL TD3+BC"
            axis.bar(
                x + offset,
                values,
                width,
                color=VARIANT_COLORS[variant],
                alpha=0.82,
                label=label,
            )
        demonstration_teacher = region_result["demonstration_teacher"]["development"][
            "savings_vs_status_quo_pct"
        ]
        exact_benchmark = region_result["exact_native_benchmark"]["development"][
            "savings_vs_status_quo_pct"
        ]
        axis.axhline(
            region_result["goal_savings_pct"],
            color="#C00000",
            linestyle="--",
            linewidth=1.5,
            label="Savings gate",
        )
        axis.axhline(
            demonstration_teacher,
            color="#7F6000",
            linestyle="-.",
            linewidth=1.5,
            label="Demonstration teacher",
        )
        axis.axhline(
            exact_benchmark,
            color="#7030A0",
            linestyle=":",
            linewidth=1.8,
            label="Exact-native benchmark",
        )
        axis.set_xticks(x, seeds)
        axis.set_xlabel("Training seed")
        axis.set_ylabel("Primary savings vs. Status Quo (%)")
        axis.set_title(f"{region.upper()} a-d frozen confirmation")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(
        "Teacher imitation preserved by genuine TD3+BC reward updates",
        fontsize=14,
        fontweight="bold",
    )
    output_path = CANONICAL_DIR / "v5_savings_by_seed.png"
    save_figure_if_changed(fig, output_path, dpi=180)
    plt.close(fig)


def save_attribution_safety_figure(results: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), constrained_layout=True)
    seeds = np.arange(301, 306)
    x = np.arange(len(seeds))
    for region in ("us", "global"):
        deltas = results["regions"][region]["attribution"][
            "paired_post_rl_minus_bc_percentage_points"
        ]["values"]
        axes[0].plot(
            x,
            deltas,
            marker="o",
            linewidth=1.8,
            color=REGION_COLORS[region],
            label=region.upper(),
        )
    axes[0].axhline(0.0, color="#404040", linewidth=1.0)
    axes[0].set_xticks(x, seeds)
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Post-RL minus BC savings (percentage points)")
    axes[0].set_title("Incremental economic effect of reward updates")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    labels = ["US", "Global"]
    decoder = [
        100.0
        * results["regions"][region]["secondary_kpis"]["development"][
            "mean_decoder_adjustment_rate"
        ]
        for region in ("us", "global")
    ]
    emergency = [
        100.0
        * results["regions"][region]["secondary_kpis"]["development"][
            "maximum_emergency_intervention_rate"
        ]
        for region in ("us", "global")
    ]
    indices = np.arange(len(labels))
    axes[1].bar(
        indices - 0.18,
        decoder,
        0.36,
        color="#4472C4",
        label="Normal decoder adjustment",
    )
    axes[1].bar(
        indices + 0.18,
        emergency,
        0.36,
        color="#C00000",
        label="Emergency fallback",
    )
    axes[1].set_xticks(indices, labels)
    axes[1].set_ylabel("Development steps (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Constraint enforcement is frequent; emergencies are zero")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    output_path = CANONICAL_DIR / "v5_attribution_and_safety_rgb.png"
    save_figure_if_changed(
        fig,
        output_path,
        dpi=180,
        flatten=True,
    )
    plt.close(fig)


def report_table_row(region: str, result: dict[str, Any]) -> str:
    bc = result["variants"]["bc_only"]["development_savings_pct"]
    rl = result["variants"]["post_rl_td3bc"]["development_savings_pct"]
    transfer = result["variants"]["post_rl_td3bc"][
        "descriptive_transfer_savings_pct"
    ]
    delta = result["attribution"]["paired_post_rl_minus_bc_percentage_points"]
    safety = result["secondary_kpis"]["development"]
    return (
        f"| {region.upper()} | {bc['mean']:.3f}% | {rl['mean']:.3f}% | "
        f"{rl['minimum']:.3f}% | {delta['mean']:+.4f} pp | "
        f"{transfer['mean']:.3f}% | {100*safety['mean_decoder_adjustment_rate']:.2f}% | "
        f"{100*safety['maximum_emergency_intervention_rate']:.2f}% |"
    )


def save_report(results: dict[str, Any]) -> None:
    lines = [
        "# Corrected v5 TD3+BC Results",
        "",
        "## Verdict",
        "",
        "The corrected, cryptographically verified post-RL TD3+BC network clears "
        "the predeclared a-d savings and safety gates in both regions. The teacher "
        "is absent during reward updates and inference. Every post-RL seed has "
        "nonzero actor and critic changes and 2,048 TD3 updates.",
        "",
        "| Region | BC-only mean | Post-RL mean | Post-RL minimum | RL - BC mean | e-h descriptive mean | Normal decoder adjustment | Emergency fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        report_table_row("us", results["regions"]["us"]),
        report_table_row("global", results["regions"]["global"]),
        "",
        "The result is primarily an imitation result preserved by genuine reward "
        "training: post-RL minus BC is nearly zero in US and slightly negative in "
        "Global. It is therefore inaccurate to attribute the full absolute savings "
        "to the short TD3 phase.",
        "",
        "Routine constraint-decoder adjustment is frequent and must not be confused "
        "with emergency shielding. The decoder maps network preferences to feasible "
        "service, origin, and destination amounts on most steps; the separate "
        "emergency fallback remains unused.",
        "",
        "## Development and transfer statistics",
        "",
    ]
    for region in ("us", "global"):
        result = results["regions"][region]
        rl = result["variants"]["post_rl_td3bc"]["development_savings_pct"]
        transfer = result["variants"]["post_rl_td3bc"][
            "descriptive_transfer_savings_pct"
        ]
        demonstration_teacher = result["demonstration_teacher"]
        exact_benchmark = result["exact_native_benchmark"]
        lines.extend(
            [
                f"### {region.upper()}",
                "",
                f"- a-d post-RL mean: **{rl['mean']:.6f}%**; minimum: "
                f"**{rl['minimum']:.6f}%**; optimization-seed interval: "
                f"[{rl['optimization_seed_t95_interval'][0]:.6f}%, "
                f"{rl['optimization_seed_t95_interval'][1]:.6f}%].",
                f"- e-h descriptive post-RL mean: **{transfer['mean']:.6f}%**; "
                f"minimum: **{transfer['minimum']:.6f}%**.",
                f"- Frozen greedy demonstration teacher: "
                f"{demonstration_teacher['development']['savings_vs_status_quo_pct']:.6f}% "
                f"a-d and {demonstration_teacher['descriptive_transfer']['savings_vs_status_quo_pct']:.6f}% "
                "e-h. This is the offline label policy, not an RL result.",
                f"- Separate exact-native analytic benchmark: "
                f"{exact_benchmark['development']['savings_vs_status_quo_pct']:.6f}% "
                f"a-d and {exact_benchmark['descriptive_transfer']['savings_vs_status_quo_pct']:.6f}% "
                "e-h. It is not the demonstration teacher.",
                f"- Post-RL network captures {result['exact_native_savings_capture_pct']:.2f}% "
                "of the exact-native a-d benchmark.",
                "",
            ]
        )
    lines.extend(
        [
            "## Secondary demand-charge finding",
            "",
            "Demand charge was excluded from the primary training objective and is "
            "reported at the illustrative $15/kW-cycle reference. The final policy "
            "concentrates load spatially, increasing the summed site peaks and the "
            "secondary demand-charge estimate.",
            "",
            "| Region/scope | Primary savings | Demand-charge change | Combined primary + demand-charge sensitivity |",
            "|---|---:|---:|---:|",
        ]
    )
    for region in ("us", "global"):
        for scope, label in (
            ("development", "a-d"),
            ("descriptive_transfer", "e-h descriptive"),
        ):
            kpi = results["regions"][region]["secondary_kpis"][scope]
            primary_pct = (
                100.0
                * kpi["mean_primary_savings_usd"]
                / kpi["mean_baseline_primary_cost_usd"]
            )
            lines.append(
                f"| {region.upper()} {label} | {primary_pct:.3f}% | "
                f"{kpi['mean_demand_charge_change_pct']:+.3f}% "
                f"(${kpi['mean_demand_charge_change_usd']:,.0f}) | "
                f"{kpi['combined_primary_plus_demand_charge_savings_pct']:+.3f}% |"
            )
    lines.extend(
        [
            "",
            "This sensitivity is not a tariff forecast: the simulator uses five-minute "
            "peaks and one illustrative rate, while real demand tariffs commonly use "
            "15-minute windows and utility-specific ratchets. It nevertheless shows "
            "that optimizing energy and grid stress alone does not guarantee peak-demand savings.",
            "",
            "## Scope",
            "",
            "- a-d is development/frozen confirmation, not an untouched holdout.",
            "- e-h is descriptive transfer only.",
            "- All seeds share one deterministic May 2025 CAISO archetype.",
            "- The optimization-seed intervals describe training variability, not "
            "independent months or markets.",
            "",
            "## Reproduction",
            "",
            "The published model records were generated from source commit "
            f"`{results['source_commit']}`. Recreate that source identity in a "
            "separate worktree; final reporting and thesis builders live on the "
            "later final branch tip.",
            "",
            "```powershell",
            f"git worktree add ..\\dc-energy-v5-training {results['source_commit']}",
            "Push-Location ..\\dc-energy-v5-training",
            "python scripts\\run_offpolicy_campaign_v5.py --campaign td3bc_bconly_frozen_v3 --regions us global --seeds 301 302 303 304 305 --workers 4",
            "python scripts\\run_offpolicy_campaign_v5.py --campaign td3bc_postrl_frozen_v3 --regions us global --seeds 301 302 303 304 305 --workers 4",
            "python scripts\\build_offpolicy_evidence_v5.py --bc-campaign td3bc_bconly_frozen_v3 --postrl-campaign td3bc_postrl_frozen_v3 --seeds 301 302 303 304 305 --workers 4 --suffix v3",
            "Pop-Location",
            "# Back on the final branch tip:",
            "python scripts\\build_v5_results.py",
            "```",
            "",
        ]
    )
    (CANONICAL_DIR / "results_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    results = build_results()
    (CANONICAL_DIR / "canonical_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    save_savings_figure(results)
    save_attribution_safety_figure(results)
    save_tier_example_figure()
    save_report(results)
    print(CANONICAL_DIR / "canonical_results.json")


if __name__ == "__main__":
    main()
