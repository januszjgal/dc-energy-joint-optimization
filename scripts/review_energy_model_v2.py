"""Generate the no-training review gate for energy model v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines import (  # noqa: E402
    DrainImmediatelyPolicy,
    RoundRobinPolicy,
    StatusQuoPolicy,
)
from env.data_loader import load_scenario  # noqa: E402
from env.multi_dc_env import MultiDCEnv  # noqa: E402
from env.protocol import load_protocol  # noqa: E402
from evaluate import compute_summary, run_episode  # noqa: E402
from scripts.compute_qp_optimum import build_inputs, solve_clarabel  # noqa: E402

PROTOCOL = load_protocol()
ALPHA = float(PROTOCOL["objective"]["peak_penalty_weight"])
COMPLETION_PENALTY = float(
    PROTOCOL["objective"]["completion_penalty_weight"]
)


def make_env(
    path: Path,
    batch: bool,
    peak_penalty_weight: float = ALPHA,
) -> MultiDCEnv:
    sites, power_model, batch_config = load_scenario(
        path, batch_enabled=batch, seed=42
    )
    return MultiDCEnv(
        sites,
        power_model,
        batch_enabled=batch,
        peak_penalty_weight=peak_penalty_weight,
        flexibility_factor=batch_config.get("flexibility_factor", 1.0),
        deadline_penalty_weight=batch_config.get(
            "deadline_penalty_weight", 2.0
        ),
        urgency_horizon_steps=batch_config.get("urgency_horizon_steps", 12),
        enforce_batch_completion=batch,
        completion_penalty_weight=(
            COMPLETION_PENALTY if batch else None
        ),
    )


def evaluate_baselines(
    path: Path,
    batch: bool,
    peak_penalty_weight: float = ALPHA,
) -> dict[str, Any]:
    policies = [StatusQuoPolicy, RoundRobinPolicy]
    if batch:
        policies.append(DrainImmediatelyPolicy)
    result = {}
    for policy_type in policies:
        policy = policy_type()
        _, history = run_episode(
            make_env(path, batch, peak_penalty_weight),
            policy.predict,
            is_sb3=False,
        )
        result[policy.name] = compute_summary(
            history, batch_enabled=batch
        )
    return result


def format_money(value: float) -> str:
    return f"${value / 1e6:.3f}M"


def build(year: int) -> dict[str, Any]:
    output = ROOT / "output" / "energy_model_v2" / str(year)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (
            ROOT
            / "data"
            / "energy_model_v2"
            / str(year)
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    scenarios = [
        ("US a-d", f"us_model_v2_{year}.yaml"),
        ("US e-h", f"us_model_eh_v2_{year}.yaml"),
        ("Global a-d", f"global_model_v2_{year}.yaml"),
        ("Global e-h", f"global_model_eh_v2_{year}.yaml"),
    ]
    result: dict[str, Any] = {
        "year": year,
        "energy_manifest": manifest,
        "scenarios": {},
    }
    rows = []
    for label, filename in scenarios:
        scenario = ROOT / "env" / "scenarios" / filename
        spatial_qp = solve_clarabel(
            build_inputs(str(scenario), False), False
        )
        batch_qp = solve_clarabel(
            build_inputs(
                str(scenario),
                True,
                enforce_batch_completion=True,
                completion_penalty_weight=COMPLETION_PENALTY,
            ),
            True,
        )
        spatial_baselines = evaluate_baselines(scenario, False)
        batch_baselines = evaluate_baselines(scenario, True)
        spatial_sq = spatial_baselines[
            "Status Quo (local, no deferral)"
        ]["total_cost"]
        batch_sq = batch_baselines[
            "Status Quo (local, no deferral)"
        ]["total_cost"]
        status_quo_delta = abs(batch_sq - spatial_sq)
        if status_quo_delta > 0.01:
            raise RuntimeError(
                f"{label}: Status Quo spatial/batch costs differ by "
                f"${status_quo_delta:,.2f}. Resolve workload/capacity "
                "normalization before publishing a headroom gate."
            )
        entry = {
            "scenario": str(scenario.relative_to(ROOT)).replace("\\", "/"),
            "spatial_qp": spatial_qp,
            "batch_qp": batch_qp,
            "spatial_baselines": spatial_baselines,
            "batch_baselines": batch_baselines,
            "spatial_headroom_vs_status_quo_pct": (
                100.0 * (spatial_sq - spatial_qp["optimum_total"])
                / spatial_sq
            ),
            "joint_headroom_vs_status_quo_pct": (
                100.0 * (batch_sq - batch_qp["optimum_total"])
                / batch_sq
            ),
            "incremental_temporal_headroom_pct": (
                100.0
                * (
                    spatial_qp["optimum_total"]
                    - batch_qp["optimum_total"]
                )
                / spatial_qp["optimum_total"]
            ),
        }
        result["scenarios"][label] = entry
        rows.append((label, entry, spatial_sq, batch_sq))

    (output / "gate_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    diagnostics = manifest["diagnostics"]
    report = [
        f"# Energy Model v2 Review Gate — May {year}",
        "",
        "> **No PPO retraining has been launched.** This report validates the "
        "energy system and theoretical headroom first.",
        "",
        "## Experimental energy model",
        "",
        "The primary model is one co-timestamped, real CAISO archetype:",
        "",
        "- CAISO Today's Outlook native five-minute net demand;",
        "- CAISO OASIS NP15 hourly day-ahead total LMP, expanded stepwise;",
        "- negative prices preserved;",
        "- one Pacific civil-time calendar with IANA/DST conversion;",
        "- price and net demand shifted together across market slots;",
        "- no regional price re-averaging.",
        "- net demand scaled by the maximum absolute Pacific May value, "
        "preserving its sign in [-1, 1];",
        "- every site is an equal 100 MW proxy with normalized capacity 1.0;",
        "- current measured batch arrival is observable before the action that "
        "may release it.",
        "",
        "US slots: Pacific, Mountain, Central, Eastern. Global slots: "
        "Pacific, Central, Amsterdam, Singapore.",
        "",
        "Google's May-2019 workload shapes are anchored to the May-2025 "
        "energy calendar as an explicit cross-year counterfactual.",
        "",
        "## Finite-window boundary handling",
        "",
        "The time-zone transformation is continuous rather than circular. Every "
        "slot still contains exactly 8,928 five-minute intervals (744 hours), "
        "but shifted slots can use adjacent real CAISO hours at the month "
        "boundary instead of wrapping May 31 back to May 1. Singapore is 15 "
        "hours ahead of Pacific time, so its reference window replaces CAISO's "
        "first 15 hours of May (mean $21.54/MWh) with the first 15 hours of "
        "June (mean $29.16/MWh). Its monthly mean is therefore $26.09/MWh "
        "instead of Pacific's $25.93/MWh: a $0.154/MWh (0.59%) boundary effect, "
        "not extra simulated time or a Singapore price premium.",
        "",
        "The continuous shift is retained because a circular within-May shift "
        "would create an artificial May 31-to-May 1 discontinuity. Baselines, "
        "QP, and future learned policies are compared on the same slot data "
        "within each scenario; regional monthly means are not forced to match.",
        "",
        "## Reference diagnostics",
        "",
        "| Slot | Mean price | Price std | Price/net-demand r | "
        "Price peak UTC | Net peak UTC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in diagnostics.items():
        report.append(
            f"| {name} | ${values['price_mean_usd_mwh']:.2f}/MWh | "
            f"${values['price_std_usd_mwh']:.2f}/MWh | "
            f"{values['price_net_demand_correlation']:.3f} | "
            f"{values['average_price_peak_utc_hour']:02d}:00 | "
            f"{values['average_net_demand_peak_utc_hour']:02d}:00 |"
        )

    joint_headroom = [
        entry["joint_headroom_vs_status_quo_pct"]
        for _, entry, _, _ in rows
    ]
    temporal_headroom = [
        entry["incremental_temporal_headroom_pct"]
        for _, entry, _, _ in rows
    ]
    status_quo_name = "Status Quo (local, no deferral)"
    component_shares = {}
    demand_charge_ref = None
    for label in ("US a-d", "Global a-d"):
        summary = result["scenarios"][label]["spatial_baselines"][
            status_quo_name
        ]
        component_shares[label] = (
            100.0
            * summary["total_peak_penalty"]
            / summary["total_energy_cost"]
        )
        demand_charge_ref = summary["demand_charge_ref"]

    flexibility_ranges = {}
    flexibility_path = output / "flexibility_sweep.json"
    if flexibility_path.exists():
        flexibility = json.loads(
            flexibility_path.read_text(encoding="utf-8")
        )
        for factor in ("0.0", "1.0", "2.0"):
            values = [
                scenario[factor][
                    "incremental_temporal_headroom_pct"
                ]
                for scenario in flexibility["scenarios"].values()
            ]
            flexibility_ranges[factor] = (min(values), max(values))
    structure = None
    rated_power_ranges = {}
    structure_path = output / "structure_ablation.json"
    if structure_path.exists():
        structure = json.loads(structure_path.read_text(encoding="utf-8"))
        for label, entries in structure["rated_power_sensitivity"].items():
            values = [entry["headroom_pct"] for entry in entries.values()]
            rated_power_ranges[label] = (min(values), max(values))
    status_quo_ramps = [
        result["scenarios"][label]["spatial_baselines"][
            status_quo_name
        ]["physical_ramp_metrics"]["1h"]
        for label in result["scenarios"]
    ]
    report += [
        "",
        "Reference CAISO facts:",
        "",
        "- mean DAM price: about $25.93/MWh;",
        "- negative-price intervals are retained;",
        "- average local price and net-demand trough: ~12:00 PDT;",
        "- average local price and net-demand peak: ~20:00 PDT;",
        "- price/net-demand correlation: ~0.895.",
        "- negative-net-demand intervals average about $2.86/MWh versus "
        "$29.88/MWh otherwise; the deepest 5% average about $0.11/MWh.",
        "",
        "## QP headroom gate — unrestricted-routing upper bound",
        "",
        "| Scenario | Status quo | Spatial QP | Joint QP | "
        "Spatial headroom | Joint headroom | Incremental temporal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, entry, spatial_sq, batch_sq in rows:
        report.append(
            f"| {label} | {format_money(spatial_sq)} | "
            f"{format_money(entry['spatial_qp']['optimum_total'])} | "
            f"{format_money(entry['batch_qp']['optimum_total'])} | "
            f"{entry['spatial_headroom_vs_status_quo_pct']:.2f}% | "
            f"{entry['joint_headroom_vs_status_quo_pct']:.2f}% | "
            f"{entry['incremental_temporal_headroom_pct']:.2f}% |"
        )

    report += [
        "",
        "## Objective coefficient and demand-charge treatment",
        "",
        f"The frozen primary objective keeps `alpha={ALPHA:g}`. On a-d calibration "
        "cells, the Status Quo grid-stress component is "
        f"{component_shares['US a-d']:.1f}% of energy cost in US and "
        f"{component_shares['Global a-d']:.1f}% in Global: material but not "
        "dominant. The alpha sweep is generated from a-d only so held-out e-h "
        "does not select the coefficient.",
        "",
        "The standardized demand charge is excluded from the primary reward. "
        f"At $15/kW-cycle it is about {format_money(demand_charge_ref)} for "
        "Status Quo and would dominate the controlled objective; it is also "
        "not a real Dutch or Singapore tariff. It remains a secondary reported "
        "sensitivity and optional later extension.",
        "",
        "![Objective sensitivity](objective_sensitivity.png)",
    ]
    if flexibility_ranges:
        report += [
            "",
            "## Experimental deadline sensitivity",
            "",
            "ClusterData 2019 does not publish deadlines. The primary "
            "`flexibility_factor=1` sets `H=2x` fitted mean duration. "
            "QP robustness uses factor 0 (`H=1x`) and factor 2 (`H=3x`).",
            "",
            "- factor 0 incremental temporal headroom: "
            f"{flexibility_ranges['0.0'][0]:.1f}–"
            f"{flexibility_ranges['0.0'][1]:.1f}%;",
            "- factor 1 incremental temporal headroom: "
            f"{flexibility_ranges['1.0'][0]:.1f}–"
            f"{flexibility_ranges['1.0'][1]:.1f}%;",
            "- factor 2 incremental temporal headroom: "
            f"{flexibility_ranges['2.0'][0]:.1f}–"
            f"{flexibility_ranges['2.0'][1]:.1f}%.",
            "",
            "Full table: [flexibility_sweep.md](flexibility_sweep.md).",
        ]
    if structure:
        report += [
            "",
            "## Structural attribution and rated-power sensitivity",
            "",
            "V2 uses one price level shifted in time, so the v1 unequal-mean "
            "synthetic-price criticism no longer applies. The following "
            "a-d-only QP ablations diagnose—not additively decompose—the "
            "sources of spatial headroom:",
            "",
            "| Condition | US headroom | Global headroom |",
            "|---|---:|---:|",
        ]
        for condition in (
            "primary",
            "energy_only",
            "pooled_power",
            "synchronous_market",
            "shifted_market_only",
            "power_heterogeneity_only",
            "degenerate_control",
        ):
            us_value = structure["scenarios"]["US a-d"][condition][
                "headroom_pct"
            ]
            global_value = structure["scenarios"]["Global a-d"][condition][
                "headroom_pct"
            ]
            report.append(
                f"| {condition} | {us_value:.2f}% | {global_value:.2f}% |"
            )
        report += [
            "",
            "Per-cell power heterogeneity makes routing non-degenerate but is "
            "not a novel RL mechanism. Shifted market phase is the dominant "
            "Global lever; both mechanisms interact. The fully equal, "
            "energy-only control leaves about 0.05% headroom.",
            "",
            "With fixed alpha, R sensitivity is modest in US "
            f"({rated_power_ranges['US a-d'][0]:.2f}–"
            f"{rated_power_ranges['US a-d'][1]:.2f}%) "
            "and larger in Global "
            f"({rated_power_ranges['Global a-d'][0]:.2f}–"
            f"{rated_power_ranges['Global a-d'][1]:.2f}%) "
            "over R=50–200 MW.",
            "",
            "![Structural attribution](structure_ablation.png)",
            "",
            "Full table: [structure_ablation.md](structure_ablation.md).",
        ]
    report += [
        "",
        "## Ramp-rate treatment",
        "",
        "Ramp rate is evaluated, not optimized. The primary reward contains no "
        "net-demand derivative. Price correlates about 0.206 with the one-hour "
        "CAISO ramp and 0.429 with the three-hour ramp, so it is only a partial "
        "ramp proxy.",
        "",
        "Evaluation reports per-region maximum and p95 upward ramps for one "
        "hour and three hours, comparing raw net demand with net demand plus "
        "DC load. As an instrumentation check, Status Quo changes the maximum "
        "one-hour ramp by "
        f"{min(x['max_up_delta_mw'] for x in status_quo_ramps):.1f} to "
        f"{max(x['max_up_delta_mw'] for x in status_quo_ramps):.1f} MW "
        f"against a {status_quo_ramps[0]['base_max_up_mw']:,.0f} MW base ramp. "
        "A ramp penalty is future work only if trained v2 policies worsen these "
        "physical KPIs.",
    ]

    report += [
        "",
        "## Figures",
        "",
        "![Reference month](reference_month.png)",
        "",
        "![US shifted profiles](us_shifted_daily_profiles.png)",
        "",
        "![Global shifted profiles](global_shifted_daily_profiles.png)",
        "",
        "## Review interpretation",
        "",
        "The aligned real energy model creates meaningful but optimistic "
        "unrestricted-routing optimization "
        f"headroom ({min(joint_headroom):.1f}–{max(joint_headroom):.1f}% "
        "depending on scenario). Spatial diversity remains the dominant lever. "
        "Incremental temporal headroom is positive but modest "
        f"({min(temporal_headroom):.1f}–{max(temporal_headroom):.1f}%), so a "
        "defensible thesis should not promise a large temporal gain.",
        "",
        "## Explicit limitations / future work",
        "",
        "- The primary experiment is a controlled CAISO archetype, not a real "
        "multi-market replay.",
        "- Regional price levels are intentionally not re-averaged.",
        "- Time-zone shifts use adjacent real boundary hours rather than a "
        "circular within-May wrap, so shifted monthly means can differ slightly.",
        "- Five-minute RTM price is future robustness work.",
        "- Historical 2019/2024 duck-curve comparison and extrapolation are "
        "future work, not additional training scenarios.",
        "- Workload shapes are from May 2019 while the energy calendar is May "
        f"{year}; this is an explicit counterfactual.",
        "- The controlled study reuses the same May-2025 CAISO calendar across "
        "OOF workload folds; other months/years are future work.",
        "- Service and batch routing are unrestricted across all four slots; "
        "latency, residency, and movement constraints are future work.",
        "- The primary deadline uses fitted mean duration with flexibility "
        "factor 1.0; factors 0 and 2 are planned robustness cases.",
        "- A causal MPC comparator is future work; the QP remains a "
        "clairvoyant diagnostic only.",
        "- The primary reward penalizes high net-demand exposure, not ramp "
        "rate. Evaluation reports per-region 1h/3h maximum and p95 upward "
        "ramps; a ramp penalty is future work only if those KPIs worsen.",
        "",
        "## Downstream frozen campaign",
        "",
        "**The energy gate was consumed by the completed 80-model frozen OOF "
        "campaign. The joint-shaping headline criterion failed; only Global "
        "spatial PPO produced positive held-out optimizer CIs in both folds.**",
        "",
        "See [the canonical OOF results](../../oof_v2_2025/results_report.md).",
    ]
    (output / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    build(args.year)


if __name__ == "__main__":
    main()
