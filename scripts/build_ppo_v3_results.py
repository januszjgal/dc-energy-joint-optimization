"""Build the PPO v3 canonical analysis package from frozen artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_v3 import DEFAULT_DEADLINE_BUCKET_EDGES, make_recovery_env

V3_ROOT = ROOT / "output" / "ppo_v3_reward_sweep"
V2_ROOT = ROOT / "output" / "oof_v2_2025"
MODELS_ROOT = ROOT / "models" / "ppo_v3_reward_sweep"

PRIMARY_BASELINE = "Status Quo (local, no deferral)"
SAFETY_FLOOR = 0.9999
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 0
ACTION_LOGIT_BOUND = 3.0

FIGURE_PATHS = {
    "successive_halving": V3_ROOT / "ppo_v3_successive_halving.png",
    "budget_curve": V3_ROOT / "ppo_v3_budget_curve.png",
    "selected_ad": V3_ROOT / "ppo_v3_selected_ad_safety_cost.png",
    "eh_transfer": V3_ROOT / "ppo_v3_eh_transfer.png",
    "negative_probe": V3_ROOT / "ppo_v3_negative_net_demand_probe.png",
}

V3_INPUTS = [
    "protocol.json",
    "preflight.json",
    "round1_results.json",
    "round1_selection.json",
    "round2_results.json",
    "round2_selection.json",
    "full_results.json",
    "full_gate.json",
    "budget_curve_results.json",
    "budget_selection.json",
    "budget_selected_results.json",
    "budget_selected_gate.json",
    "budget_selected_eh_results.json",
    "final_eh_transfer.json",
]

V2_INPUTS = [
    "canonical_results.json",
    "results_report.md",
    "summary.json",
    "protocol.json",
]

SELECTED_EXPECTED = {
    "us": {"combo": "R3_P1", "budget_stage": "budget_151552", "timesteps": 151552},
    "global": {
        "combo": "R0_P3",
        "budget_stage": "budget_1003520",
        "timesteps": 1003520,
    },
}

ROUND_TIMESTEPS = {"round1": 151552, "round2": 301056, "full": 501760}

LIMITATION_NOTES = {
    "v2_reward_scale_confound": (
        "Frozen v2 spatial-vs-joint comparisons were also affected by a reward-scale "
        "confound, so v3 does not use any new spatial-only-versus-joint comparison as "
        "evidence."
    ),
    "tests": (
        "Existing exact demand-charge and batch-accounting telescope tests had already "
        "passed before this frozen reward-sweep analysis."
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_canonical_json(path: Path, payload: Any) -> str:
    text = canonical_json_text(payload)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return sha256_file(path)


def write_text(path: Path, text: str) -> str:
    if not text.endswith("\n"):
        text += "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return sha256_file(path)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def bootstrap_ci(values: list[float], *, resamples: int = BOOTSTRAP_RESAMPLES) -> list[float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return [math.nan, math.nan]
    if array.size == 1:
        only = float(array[0])
        return [only, only]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def parse_seed_key(seed_key: str) -> int:
    return int(str(seed_key).lstrip("s"))


def safe_seed_fraction(summary: dict[str, Any]) -> float:
    return float(summary["safe_seed_count"]) / float(summary["seed_count"])


def savings_pct_from_usd(interval_usd: list[float], baseline_cost: float) -> list[float]:
    return [100.0 * value / baseline_cost for value in interval_usd]


def compact_usd(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}${value:,.0f}"


def compact_pct(value: float, digits: int = 3) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def compact_completion(value: float) -> str:
    return f"{value:.6f}"


def compact_millions(value: float) -> str:
    return f"${value / 1e6:.3f}M"


def model_group_key(model_path: str) -> tuple[str, str, str, int]:
    parts = PurePosixPath(model_path).parts
    _, _, stage, region, combo, seed_dir, _ = parts
    return stage, region, combo, parse_seed_key(seed_dir)


def load_inputs() -> dict[str, Any]:
    v3 = {name: read_json(V3_ROOT / name) for name in V3_INPUTS}
    v2 = {name: read_json(V2_ROOT / name) if name.endswith(".json") else (V2_ROOT / name).read_text(encoding="utf-8") for name in V2_INPUTS}
    manifest = read_json(MODELS_ROOT / "manifest.json")
    complete_records = []
    for path in sorted(MODELS_ROOT.glob("**/complete.json")):
        record = read_json(path)
        record["_relative_path"] = path.relative_to(ROOT).as_posix()
        record["_record_sha256"] = sha256_file(path)
        complete_records.append(record)
    return {"v3": v3, "v2": v2, "manifest": manifest, "complete_records": complete_records}


def build_input_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in V3_INPUTS:
        path = V3_ROOT / name
        hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for name in V2_INPUTS:
        path = V2_ROOT / name
        hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    hashes[(MODELS_ROOT / "manifest.json").relative_to(ROOT).as_posix()] = sha256_file(
        MODELS_ROOT / "manifest.json"
    )
    return dict(sorted(hashes.items()))


def build_model_provenance(manifest: dict[str, Any], complete_records: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_map = {row["path"]: row for row in manifest["models"]}
    if len(manifest_map) != manifest["model_count"]:
        raise ValueError("Model manifest count mismatch.")
    complete_map: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    stage_counts: Counter[str] = Counter()
    grouped: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in complete_records:
        key = (
            record["stage"],
            record["region"],
            f"{record['candidate']}_{record['variant']}" if record["variant"] else record["candidate"],
            int(record["seed"]),
        )
        complete_map[key] = record
        stage_counts[record["stage"]] += 1
        grouped[key[:3]].append(record)
        manifest_row = manifest_map.get(record["model_path"])
        if manifest_row is None:
            raise ValueError(f"Manifest missing model path {record['model_path']}")
        if manifest_row["sha256"] != record["model_sha256"]:
            raise ValueError(f"Model hash mismatch for {record['model_path']}")
        if manifest_row["job_fingerprint"] != record["job_fingerprint"]:
            raise ValueError(f"Job fingerprint mismatch for {record['model_path']}")
    if len(complete_records) != manifest["model_count"]:
        raise ValueError("Completion record count does not match model count.")
    groups = []
    for (stage, region, combo), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["seed"]))
        groups.append(
            {
                "stage": stage,
                "region": region,
                "combo": combo,
                "seed_count": len(rows),
                "seeds": [int(row["seed"]) for row in rows],
                "model_sha256s": [row["model_sha256"] for row in rows],
                "job_fingerprints": [row["job_fingerprint"] for row in rows],
                "complete_record_sha256s": [row["_record_sha256"] for row in rows],
            }
        )
    return {
        "protocol_sha256": manifest["protocol_sha256"],
        "manifest_sha256": sha256_file(MODELS_ROOT / "manifest.json"),
        "model_count": manifest["model_count"],
        "complete_record_count": len(complete_records),
        "stage_counts": dict(sorted(stage_counts.items())),
        "manifest_entries": manifest["models"],
        "stage_groups": groups,
        "_complete_map": complete_map,
    }


def baseline_summaries(region_results: dict[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, baseline in region_results["baselines"].items():
        result[name] = {
            "total_cost": float(baseline["total_cost"]),
            "demand_charge_ref": float(baseline["demand_charge_ref"]),
            "peak_grid_mw": float(baseline["peak_grid_mw"]),
        }
    return result


def summarize_seed(seed_key: str, seed: dict[str, Any], baseline_cost: float) -> dict[str, Any]:
    return {
        "seed": parse_seed_key(seed_key),
        "total_cost_usd": float(seed["total_cost"]),
        "improvement_usd_vs_status_quo": float(baseline_cost - seed["total_cost"]),
        "savings_vs_status_quo_pct": float(seed["savings_vs_status_quo_pct"]),
        "batch_completion_fraction": float(seed["batch_completion_fraction"]),
        "terminal_batch_pool": float(seed["terminal_batch_pool"]),
        "maximum_service_backlog": float(seed["max_backlog"]),
        "total_batch_expired": float(seed.get("total_batch_expired", 0.0)),
        "action_saturation_fraction": float(seed.get("action_saturation_fraction", 0.0)),
        "safe": bool(seed["safe"]),
        "job": seed["job"],
        "model_sha256": seed["model_sha256"],
    }


def summarize_candidate(
    combo: str,
    candidate: dict[str, Any],
    baseline_cost: float,
    *,
    override_optimizer_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seeds = [
        summarize_seed(seed_key, candidate["seeds"][seed_key], baseline_cost)
        for seed_key in sorted(candidate["seeds"], key=parse_seed_key)
    ]
    improvements = [seed["improvement_usd_vs_status_quo"] for seed in seeds]
    savings = [seed["savings_vs_status_quo_pct"] for seed in seeds]
    comparison = override_optimizer_comparison or {
        "interpretation": (
            "Positive USD means joint PPO costs less than deterministic Status Quo. "
            "The interval describes optimizer-seed variability only."
        ),
        "mean_improvement_usd": float(np.mean(improvements)),
        "optimizer_bootstrap_ci95_usd": bootstrap_ci(improvements),
        "optimizer_bootstrap_ci95_savings_pct": bootstrap_ci(savings),
        "all_seed_improvements_positive": bool(all(value > 0.0 for value in improvements)),
    }
    if "optimizer_bootstrap_ci95_savings_pct" not in comparison:
        comparison["optimizer_bootstrap_ci95_savings_pct"] = savings_pct_from_usd(
            comparison["optimizer_bootstrap_ci95_usd"], baseline_cost
        )
    return {
        "combo": combo,
        "candidate": candidate["candidate"],
        "variant": candidate["variant"],
        "timesteps": int(next(iter(candidate["seeds"].values()))["job"]["timesteps"]),
        "summary": candidate["summary"],
        "optimizer_comparison": comparison,
        "seeds": seeds,
    }


def build_selection_stage(
    stage_name: str,
    results: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    stage = {
        "stage": stage_name,
        "results_sha256": selection["results_sha256"],
        "selection_rule": selection["selection_rule"],
        "regions": {},
    }
    for region in ("us", "global"):
        region_results = results["regions"][region]
        baseline_cost = float(region_results["baselines"][PRIMARY_BASELINE]["total_cost"])
        candidates = []
        for combo in region_results["ranking"]:
            candidates.append(
                summarize_candidate(combo, region_results["candidates"][combo], baseline_cost)
            )
        stage["regions"][region] = {
            "scenario": region_results["scenario"],
            "baseline_cost_usd": baseline_cost,
            "diagnostic_baselines": baseline_summaries(region_results),
            "ranking": [candidate["combo"] for candidate in candidates],
            "selected_combo": selection["selected"][region][0]["combo"],
            "candidates": candidates,
        }
    return stage


def build_budget_curve(
    budget_curve_results: dict[str, Any],
    budget_selection: dict[str, Any],
    full_results: dict[str, Any],
    full_gate: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "results_sha256": budget_selection["results_sha256"],
        "selection_rule": budget_selection["selection_rule"],
        "matched_five_seed_curve": {},
        "full_501760_ten_seed": {},
        "selected_budget_by_region": {},
    }
    for region in ("us", "global"):
        region_results = budget_curve_results["regions"][region]
        baseline_cost = float(region_results["baselines"][PRIMARY_BASELINE]["total_cost"])
        budgets = []
        for budget_key in sorted(region_results["budgets"], key=int):
            budget = region_results["budgets"][budget_key]
            combo = f"{budget['candidate']}_{budget['variant']}"
            budgets.append(
                summarize_candidate(combo, budget, baseline_cost)
            )
        selected = budget_selection["selected"][region][0]
        result["matched_five_seed_curve"][region] = {
            "scenario": region_results["scenario"],
            "baseline_cost_usd": baseline_cost,
            "ranking": [int(budget["timesteps"]) for budget in budgets],
            "budgets": budgets,
        }
        result["selected_budget_by_region"][region] = {
            "combo": f"{selected['candidate']}_{selected['variant']}",
            "timesteps": int(selected["timesteps"]),
            "summary": selected["summary"],
        }
        full_region_results = full_results["regions"][region]
        full_combo = full_region_results["ranking"][0]
        gate_region = full_gate["regions"][region]
        result["full_501760_ten_seed"][region] = {
            "scenario": full_region_results["scenario"],
            "baseline_cost_usd": float(full_region_results["baselines"][PRIMARY_BASELINE]["total_cost"]),
            "selected_combo": full_combo,
            "candidate": summarize_candidate(
                full_combo,
                full_region_results["candidates"][full_combo],
                float(full_region_results["baselines"][PRIMARY_BASELINE]["total_cost"]),
                override_optimizer_comparison=gate_region["optimizer_comparison"],
            ),
            "passed": bool(gate_region["passed"]),
        }
    return result


def build_selected_ad(
    budget_selected_results: dict[str, Any],
    budget_selected_gate: dict[str, Any],
) -> dict[str, Any]:
    result = {"passed": bool(budget_selected_gate["passed"]), "regions": {}}
    for region in ("us", "global"):
        region_results = budget_selected_results["regions"][region]
        gate_region = budget_selected_gate["regions"][region]
        combo = region_results["ranking"][0]
        candidate = summarize_candidate(
            combo,
            region_results["candidates"][combo],
            float(region_results["baselines"][PRIMARY_BASELINE]["total_cost"]),
            override_optimizer_comparison=gate_region["optimizer_comparison"],
        )
        result["regions"][region] = {
            "scenario": region_results["scenario"],
            "combo": gate_region["combo"],
            "baseline": gate_region["primary_baseline"],
            "candidate": candidate,
            "candidate_summary": gate_region["candidate_summary"],
            "passed": bool(gate_region["passed"]),
        }
    return result


def build_eh_transfer(
    final_eh_transfer: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "interpretation": final_eh_transfer["interpretation"],
        "headline_eligible": bool(final_eh_transfer["headline_eligible"]),
        "results_sha256": final_eh_transfer["results_sha256"],
        "regions": {},
    }
    for region in ("us", "global"):
        region_results = final_eh_transfer["regions"][region]
        combo = region_results["ranking"][0]
        baseline_cost = float(region_results["primary_baseline"]["total_cost"])
        candidate = summarize_candidate(
            combo,
            region_results["candidates"][combo],
            baseline_cost,
        )
        result["regions"][region] = {
            "combo": combo,
            "baseline": region_results["primary_baseline"],
            "candidate": candidate,
        }
    return result


def build_v2_context(v2_canonical: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    joint_rows = [row for row in v2_canonical["fold_rows"] if row["config"].endswith("batch")]
    best_joint = max(joint_rows, key=lambda row: row["ppo_mean_savings_pct"])
    return {
        "headline_joint_shaping_supported": bool(v2_canonical["headline_joint_shaping_supported"]),
        "primary_finding": v2_canonical["primary_finding"],
        "best_joint_fold": {
            "fold": best_joint["fold"],
            "fold_label": best_joint["fold_label"],
            "config_label": best_joint["config_label"],
            "ppo_mean_savings_pct": best_joint["ppo_mean_savings_pct"],
            "optimizer_ci95_pct": best_joint["ppo_savings_ci95"],
            "feasible_seed_count": best_joint["feasible_seed_count"],
            "batch_completion_min": best_joint["batch_completion_min"],
        },
        "state_repair": protocol["state_repair"],
        "v2_reward_scale_confound_limitation": LIMITATION_NOTES["v2_reward_scale_confound"],
        "proxy_limitation": protocol["proxy_limitation"],
    }


def build_action_cap_diagnostic(selected_ad: dict[str, Any]) -> dict[str, Any]:
    max_drain_fraction = sigmoid(ACTION_LOGIT_BOUND)
    four_step_completion_ceiling = 1.0 - (1.0 - max_drain_fraction) ** 4
    return {
        "action_logit_bound": ACTION_LOGIT_BOUND,
        "max_drain_fraction": max_drain_fraction,
        "four_step_completion_ceiling": four_step_completion_ceiling,
        "selected_mean_action_saturation_fraction": {
            region: selected_ad["regions"][region]["candidate_summary"]["mean_action_saturation_fraction"]
            for region in ("us", "global")
        },
        "interpretation": (
            "The +3 action-logit bound caps any single-step learned drain at sigmoid(3) = "
            f"{100.0 * max_drain_fraction:.3f}%. Repeated control still permits "
            f">{100.0 * four_step_completion_ceiling:.4f}% theoretical clearance over four "
            "steps, and observed mean saturation was zero in the selected aggregates, so the "
            "cap can contribute but does not by itself explain the failures. Future exact "
            "0–100% drain behavior would require a hard decoder or explicit override; merely "
            "widening a sigmoid is not exact."
        ),
    }


def build_deadline_boundary_diagnostic(
    model_provenance: dict[str, Any],
    selected_ad: dict[str, Any],
    eh_transfer: dict[str, Any],
) -> dict[str, Any]:
    """Document the remaining pre-action deadline-boundary observability defect."""
    regions: dict[str, Any] = {}
    for region, records in selected_complete_records(model_provenance).items():
        training_rows = []
        for record in records:
            diagnostics_path = (ROOT / record["model_path"]).parent / "training.json"
            if sha256_file(diagnostics_path) != record["diagnostics_sha256"]:
                raise ValueError(
                    f"Training diagnostics hash mismatch for {diagnostics_path}"
                )
            diagnostics = read_json(diagnostics_path)["diagnostics"]
            training_rows.append(
                {
                    "seed": int(record["seed"]),
                    "batch_expired_during_training": float(
                        diagnostics["batch_expired_during_training"]
                    ),
                    "diagnostics_path": diagnostics_path.relative_to(ROOT).as_posix(),
                    "diagnostics_sha256": record["diagnostics_sha256"],
                }
            )
        ad_seeds = selected_ad["regions"][region]["candidate"]["seeds"]
        eh_seeds = eh_transfer["regions"][region]["candidate"]["seeds"]
        regions[region] = {
            "selected_training_seed_count": len(training_rows),
            "training_seeds_with_expiry": sum(
                row["batch_expired_during_training"] > 1e-12
                for row in training_rows
            ),
            "training_expired_total_across_all_episodes": float(
                sum(row["batch_expired_during_training"] for row in training_rows)
            ),
            "training_expired_max_per_seed_across_all_episodes": float(
                max(row["batch_expired_during_training"] for row in training_rows)
            ),
            "training_per_seed": training_rows,
            "selected_a_d_evaluation_seeds_with_expiry": sum(
                row["total_batch_expired"] > 1e-12 for row in ad_seeds
            ),
            "selected_a_d_evaluation_expired_total": float(
                sum(row["total_batch_expired"] for row in ad_seeds)
            ),
            "descriptive_e_h_evaluation_seeds_with_expiry": sum(
                row["total_batch_expired"] > 1e-12 for row in eh_seeds
            ),
            "descriptive_e_h_evaluation_expired_total": float(
                sum(row["total_batch_expired"] for row in eh_seeds)
            ),
        }
    return {
        "defect": (
            "The pre-action observation at step t can include pool entries with "
            "deadline_step <= t even though the current transition expires those "
            "entries before service. That due-now mass is therefore observable but "
            "not actionable."
        ),
        "scope": (
            "This does not change the recorded full-dollar evaluation totals. No "
            "selected a-d or descriptive e-h evaluation seed expired work, so the "
            "stale due-now bucket was absent from those selected trajectories. "
            "However, selected Global policies were trained through randomized "
            "episodes containing expiry, so the v3 state repair is incomplete and "
            "the exploratory learning result remains conditional on this boundary "
            "semantics."
        ),
        "next_protocol_fix": (
            "Preserve the current deadline and expiry timing, but compute actionable "
            "pool size, urgency, and deadline buckets from entries with "
            "deadline_step > t. Keep due-now unavoidable mass out of those actionable "
            "features; optionally expose it separately as an audit/value feature. "
            "This avoids an extra service step and makes every advertised actionable "
            "pool unit serviceable. The fix requires a new protocol and retraining; "
            "frozen v3 source/results must not be silently rewritten."
        ),
        "regions": regions,
    }


def selected_complete_records(
    model_provenance: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    complete_map = model_provenance["_complete_map"]
    output: dict[str, list[dict[str, Any]]] = {}
    for region, expected in SELECTED_EXPECTED.items():
        rows = []
        for seed in range(201, 211):
            key = (
                expected["budget_stage"],
                region,
                expected["combo"],
                seed,
            )
            if key not in complete_map:
                raise ValueError(f"Missing selected completion record for {key}")
            rows.append(complete_map[key])
        output[region] = rows
    return output


def _probe_totals_template() -> dict[str, float]:
    return {
        "step_count": 0.0,
        "negative_step_count": 0.0,
        "nonnegative_step_count": 0.0,
        "sum_mean_policy_drain_negative": 0.0,
        "sum_mean_policy_drain_nonnegative": 0.0,
        "sum_actual_pool_clearance_negative": 0.0,
        "sum_actual_pool_clearance_nonnegative": 0.0,
        "sum_negative_route_share": 0.0,
        "sum_fraction_destinations_negative": 0.0,
        "negative_high_drain_step_count": 0.0,
    }


def finalize_probe_totals(totals: dict[str, float]) -> dict[str, Any]:
    negative = int(totals["negative_step_count"])
    nonnegative = int(totals["nonnegative_step_count"])
    steps = int(totals["step_count"])
    return {
        "step_count": steps,
        "negative_step_count": negative,
        "negative_step_share": (
            totals["negative_step_count"] / totals["step_count"]
            if totals["step_count"] > 0
            else math.nan
        ),
        "mean_policy_drain_when_any_destination_negative": (
            totals["sum_mean_policy_drain_negative"] / totals["negative_step_count"]
            if totals["negative_step_count"] > 0
            else math.nan
        ),
        "mean_policy_drain_when_no_destination_negative": (
            totals["sum_mean_policy_drain_nonnegative"] / totals["nonnegative_step_count"]
            if totals["nonnegative_step_count"] > 0
            else math.nan
        ),
        "mean_actual_pool_clearance_fraction_when_any_destination_negative": (
            totals["sum_actual_pool_clearance_negative"] / totals["negative_step_count"]
            if totals["negative_step_count"] > 0
            else math.nan
        ),
        "mean_actual_pool_clearance_fraction_when_no_destination_negative": (
            totals["sum_actual_pool_clearance_nonnegative"] / totals["nonnegative_step_count"]
            if totals["nonnegative_step_count"] > 0
            else math.nan
        ),
        "mean_batch_routing_share_to_negative_demand_destinations_when_available": (
            totals["sum_negative_route_share"] / totals["negative_step_count"]
            if totals["negative_step_count"] > 0
            else math.nan
        ),
        "mean_fraction_destinations_negative_when_available": (
            totals["sum_fraction_destinations_negative"] / totals["negative_step_count"]
            if totals["negative_step_count"] > 0
            else math.nan
        ),
        "excess_batch_routing_share_to_negative_destinations_vs_uniform_when_available": (
            (
                totals["sum_negative_route_share"]
                - totals["sum_fraction_destinations_negative"]
            )
            / totals["negative_step_count"]
            if totals["negative_step_count"] > 0
            else math.nan
        ),
        "fraction_negative_steps_with_mean_drain_gt_90pct": (
            totals["negative_high_drain_step_count"] / totals["negative_step_count"]
            if totals["negative_step_count"] > 0
            else math.nan
        ),
        "negative_step_count_with_mean_drain_gt_90pct": int(
            totals["negative_high_drain_step_count"]
        ),
        "nonnegative_step_count": nonnegative,
    }


def probe_record(
    record: dict[str, Any],
    *,
    deadline_bucket_edges: tuple[int, ...],
    peak_penalty_weight: float,
    reward_scale: float,
) -> dict[str, Any]:
    env = make_recovery_env(
        ROOT / record["train_scenario"],
        seed=42,
        peak_penalty_weight=peak_penalty_weight,
        service_backlog_weight=float(record["service_backlog_weight"]),
        batch_completion_weight=float(record["batch_completion_weight"]),
        reward_scale=reward_scale,
        subtract_idle_cost=bool(record["subtract_idle_cost"]),
        urgency_potential_weight=float(record["urgency_potential_weight"]),
        domain_randomization=False,
        deadline_bucket_edges=deadline_bucket_edges,
    )
    model = PPO.load(ROOT / record["model_path"])
    totals = _probe_totals_template()
    vec_env: DummyVecEnv | VecNormalize | None = None
    normalize_path = (
        ROOT
        / record["model_path"]
    ).parent / "vecnormalize.pkl"
    try:
        if bool(record["normalize_observations"]):
            if not normalize_path.exists():
                raise ValueError(f"Missing VecNormalize file for {record['model_path']}")
            vec_env = DummyVecEnv([lambda env=env: env])
            vec_env = VecNormalize.load(normalize_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            obs = vec_env.reset()
        else:
            obs, _ = env.reset()
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            if vec_env is not None:
                obs, _, dones, infos = vec_env.step(action)
                done = bool(dones[0])
                info = infos[0]
            else:
                obs, _, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
            negative_mask = np.asarray(
                [dc["net_demand"] < 0.0 for dc in info["per_dc"]],
                dtype=bool,
            )
            any_negative = bool(negative_mask.any())
            mean_drain = float(np.mean(info["drain_rates"]))
            total_drained = float(
                sum(dc["batch_drained"] for dc in info["per_dc"])
            )
            total_pre_drain_pool = float(
                total_drained + sum(dc["batch_pool_size"] for dc in info["per_dc"])
            )
            actual_clearance = (
                total_drained / total_pre_drain_pool
                if total_pre_drain_pool > 1e-12
                else 0.0
            )
            totals["step_count"] += 1.0
            if any_negative:
                if info["batch_fractions"] is None:
                    raise ValueError(
                        "Selected joint probe expected batch_fractions to be present"
                    )
                negative_destination_fraction = float(np.mean(negative_mask))
                negative_route_share = float(
                    sum(
                        fraction
                        for fraction, is_negative in zip(
                            info["batch_fractions"],
                            negative_mask,
                            strict=True,
                        )
                        if is_negative
                    )
                )
                totals["negative_step_count"] += 1.0
                totals["sum_mean_policy_drain_negative"] += mean_drain
                totals["sum_actual_pool_clearance_negative"] += actual_clearance
                totals["sum_negative_route_share"] += negative_route_share
                totals["sum_fraction_destinations_negative"] += (
                    negative_destination_fraction
                )
                totals["negative_high_drain_step_count"] += float(mean_drain > 0.9)
            else:
                totals["nonnegative_step_count"] += 1.0
                totals["sum_mean_policy_drain_nonnegative"] += mean_drain
                totals["sum_actual_pool_clearance_nonnegative"] += actual_clearance
            if done:
                break
        if int(totals["step_count"]) != env.max_steps:
            raise RuntimeError(
                f"probe length mismatch for {record['model_path']}: "
                f"{int(totals['step_count'])} != {env.max_steps}"
            )
    finally:
        if vec_env is not None:
            vec_env.close()
        else:
            env.close()
    return {
        "seed": int(record["seed"]),
        "stage": record["stage"],
        "combo": f"{record['candidate']}_{record['variant']}",
        "scenario": record["train_scenario"],
        "model_path": record["model_path"],
        "model_sha256": record["model_sha256"],
        "job_fingerprint": record["job_fingerprint"],
        "metrics": finalize_probe_totals(totals),
    }


def build_negative_net_demand_probe(
    protocol: dict[str, Any],
    model_provenance: dict[str, Any],
) -> dict[str, Any]:
    deadline_bucket_edges = tuple(
        int(edge) for edge in protocol["environment"]["deadline_bucket_edges"]
    )
    if deadline_bucket_edges != DEFAULT_DEADLINE_BUCKET_EDGES:
        raise ValueError("Unexpected deadline bucket edges for negative-demand probe.")
    peak_penalty_weight = float(protocol["environment"]["peak_penalty_weight"])
    reward_scale = float(protocol["environment"]["reward_scale"])
    region_records = selected_complete_records(model_provenance)
    probe = {
        "scope": (
            "Deterministic descriptive probe over the ten selected a-d models per region. "
            "Each model is re-evaluated on its a-d scenario with domain randomization disabled."
        ),
        "step_condition": (
            "A step is classified as negative-demand if any destination has signed net_demand < 0."
        ),
        "actual_pool_clearance_definition": (
            "total batch drained during the step / total pre-drain batch pool, with "
            "pre-drain pool = post-step batch_pool_size + batch_drained."
        ),
        "mean_policy_drain_definition": "Mean of decoded drain_rates across destinations at the step.",
        "batch_routing_definition": (
            "Sum of batch_fractions assigned to destinations whose signed net_demand is negative."
        ),
        "future_control_note": (
            "If future work needs exact 0–100% drain behavior, it requires a hard decoder "
            "or explicit override; widening a sigmoid is not exact."
        ),
        "interpretation": (
            "Descriptive only. US routing is effectively uniform with respect to "
            "negative-site availability; Global has a modest positive spatial tilt. "
            "Neither selected policy increases drain or pool clearance in negative-"
            "demand windows, and neither blasts through the pool."
        ),
        "regions": {},
    }
    for region in ("us", "global"):
        per_seed = []
        region_totals = _probe_totals_template()
        for record in region_records[region]:
            seed_result = probe_record(
                record,
                deadline_bucket_edges=deadline_bucket_edges,
                peak_penalty_weight=peak_penalty_weight,
                reward_scale=reward_scale,
            )
            per_seed.append(seed_result)
            metrics = seed_result["metrics"]
            region_totals["step_count"] += metrics["step_count"]
            region_totals["negative_step_count"] += metrics["negative_step_count"]
            region_totals["nonnegative_step_count"] += metrics["nonnegative_step_count"]
            region_totals["sum_mean_policy_drain_negative"] += (
                metrics["mean_policy_drain_when_any_destination_negative"]
                * metrics["negative_step_count"]
            )
            region_totals["sum_mean_policy_drain_nonnegative"] += (
                metrics["mean_policy_drain_when_no_destination_negative"]
                * metrics["nonnegative_step_count"]
            )
            region_totals["sum_actual_pool_clearance_negative"] += (
                metrics["mean_actual_pool_clearance_fraction_when_any_destination_negative"]
                * metrics["negative_step_count"]
            )
            region_totals["sum_actual_pool_clearance_nonnegative"] += (
                metrics["mean_actual_pool_clearance_fraction_when_no_destination_negative"]
                * metrics["nonnegative_step_count"]
            )
            region_totals["sum_negative_route_share"] += (
                metrics["mean_batch_routing_share_to_negative_demand_destinations_when_available"]
                * metrics["negative_step_count"]
            )
            region_totals["sum_fraction_destinations_negative"] += (
                metrics["mean_fraction_destinations_negative_when_available"]
                * metrics["negative_step_count"]
            )
            region_totals["negative_high_drain_step_count"] += metrics[
                "negative_step_count_with_mean_drain_gt_90pct"
            ]
        region_metrics = finalize_probe_totals(region_totals)
        probe["regions"][region] = {
            "combo": SELECTED_EXPECTED[region]["combo"],
            "timesteps": SELECTED_EXPECTED[region]["timesteps"],
            "seed_count": len(per_seed),
            "scenario": per_seed[0]["scenario"],
            "model_records": [
                {
                    "seed": row["seed"],
                    "stage": row["stage"],
                    "model_path": row["model_path"],
                    "model_sha256": row["model_sha256"],
                    "job_fingerprint": row["job_fingerprint"],
                }
                for row in per_seed
            ],
            "aggregate": region_metrics,
            "per_seed": per_seed,
        }
    probe["conclusion"] = (
        "Neither selected policy increases drain or realized pool clearance in negative-demand "
        "windows. US routing to negative-demand destinations is essentially uniform with respect "
        "to how many destinations are negative on those steps, while Global shows only a modest "
        "positive spatial tilt. This probe is descriptive only and supports no causal or "
        "confirmatory claim."
    )
    return probe


def attach_selected_model_provenance(
    canonical: dict[str, Any],
    model_provenance: dict[str, Any],
) -> None:
    complete_map = model_provenance["_complete_map"]
    selected_records: dict[str, list[dict[str, Any]]] = {}
    eh_records: dict[str, list[dict[str, Any]]] = {}
    for region, expected in SELECTED_EXPECTED.items():
        rows = []
        for seed_row in canonical["selected_a_d"]["regions"][region]["candidate"]["seeds"]:
            key = (
                expected["budget_stage"],
                region,
                expected["combo"],
                seed_row["seed"],
            )
            record = complete_map[key]
            rows.append(
                {
                    "seed": seed_row["seed"],
                    "stage": record["stage"],
                    "model_path": record["model_path"],
                    "complete_record_path": record["_relative_path"],
                    "complete_record_sha256": record["_record_sha256"],
                    "model_sha256": record["model_sha256"],
                    "job_fingerprint": record["job_fingerprint"],
                    "diagnostics_sha256": record["diagnostics_sha256"],
                    "log_sha256": record["log_sha256"],
                }
            )
        selected_records[region] = rows
        eh_records[region] = list(rows)
    canonical["model_provenance"]["selected_a_d_model_records"] = selected_records
    canonical["model_provenance"]["selected_e_h_reused_model_records"] = eh_records
    del canonical["model_provenance"]["_complete_map"]


def validate_expected_design(protocol: dict[str, Any], canonical: dict[str, Any]) -> None:
    if not protocol["controller_scope"]["active"] == "joint_temporal_and_spatial":
        raise ValueError("Protocol is not joint-only.")
    if protocol["controller_scope"]["primary_comparator"] != "status_quo_local_no_deferral":
        raise ValueError("Unexpected primary comparator.")
    if protocol["environment"]["reward_weight_sweep_changes_evaluation_objective"]:
        raise ValueError("Evaluation objective unexpectedly changed with reward sweep.")
    if protocol["development_scope"]["fresh_confirmatory_data_available"]:
        raise ValueError("Unexpected fresh confirmatory data flag.")
    if protocol["round_1"]["timesteps"] != ROUND_TIMESTEPS["round1"]:
        raise ValueError("Unexpected round1 timesteps.")
    if protocol["round_2"]["timesteps"] != ROUND_TIMESTEPS["round2"]:
        raise ValueError("Unexpected round2 timesteps.")
    if protocol["full_development"]["timesteps"] != ROUND_TIMESTEPS["full"]:
        raise ValueError("Unexpected full-development timesteps.")
    for region, expected in SELECTED_EXPECTED.items():
        region_result = canonical["selected_a_d"]["regions"][region]
        if region_result["combo"] != expected["combo"]:
            raise ValueError(f"Unexpected selected combo for {region}.")
        if region_result["candidate"]["timesteps"] != expected["timesteps"]:
            raise ValueError(f"Unexpected selected budget for {region}.")


def build_canonical() -> dict[str, Any]:
    inputs = load_inputs()
    protocol = inputs["v3"]["protocol.json"]["protocol"]
    round1 = build_selection_stage(
        "round1",
        inputs["v3"]["round1_results.json"],
        inputs["v3"]["round1_selection.json"],
    )
    round2 = build_selection_stage(
        "round2",
        inputs["v3"]["round2_results.json"],
        inputs["v3"]["round2_selection.json"],
    )
    budget_curve = build_budget_curve(
        inputs["v3"]["budget_curve_results.json"],
        inputs["v3"]["budget_selection.json"],
        inputs["v3"]["full_results.json"],
        inputs["v3"]["full_gate.json"],
    )
    selected_ad = build_selected_ad(
        inputs["v3"]["budget_selected_results.json"],
        inputs["v3"]["budget_selected_gate.json"],
    )
    eh_transfer = build_eh_transfer(inputs["v3"]["final_eh_transfer.json"])
    model_provenance = build_model_provenance(inputs["manifest"], inputs["complete_records"])
    negative_net_demand_probe = build_negative_net_demand_probe(
        protocol,
        model_provenance,
    )
    deadline_boundary_diagnostic = build_deadline_boundary_diagnostic(
        model_provenance,
        selected_ad,
        eh_transfer,
    )
    canonical = {
        "analysis_package": "ppo-v3-canonical-analysis",
        "protocol": {
            "name": protocol["name"],
            "status": protocol["status"],
            "parent_protocol": protocol["parent_protocol"],
            "controller_scope": protocol["controller_scope"],
            "state_repair": protocol["state_repair"],
            "proxy_limitation": protocol["proxy_limitation"],
            "development_scope": protocol["development_scope"],
            "environment": protocol["environment"],
            "round_1": protocol["round_1"],
            "round_2": protocol["round_2"],
            "full_development": protocol["full_development"],
            "budget_scaling": protocol["budget_scaling"],
            "selection": protocol["selection"],
            "final_gate": protocol["final_gate"],
        },
        "frozen_input_hashes": build_input_hashes(),
        "model_provenance": model_provenance,
        "required_interpretation": {
            "joint_only": True,
            "primary_comparator": "deterministic Status Quo",
            "diagnostic_comparators": ["Round Robin", "Drain Immediately"],
            "full_dollar_evaluation_fixed_at_1000_1000": True,
            "selected": {region: SELECTED_EXPECTED[region] for region in ("us", "global")},
            "no_new_spatial_only_temporal_only_mpc_runs": True,
        },
        "successive_halving": {"round1": round1, "round2": round2},
        "budget_curve": budget_curve,
        "selected_a_d": selected_ad,
        "negative_net_demand_probe": negative_net_demand_probe,
        "descriptive_e_h_transfer": eh_transfer,
        "v2_context": build_v2_context(inputs["v2"]["canonical_results.json"], protocol),
        "action_cap_diagnostic": build_action_cap_diagnostic(selected_ad),
        "deadline_boundary_diagnostic": deadline_boundary_diagnostic,
        "tests_and_accounting_context": LIMITATION_NOTES["tests"],
    }
    validate_expected_design(protocol, canonical)
    attach_selected_model_provenance(canonical, model_provenance)
    return canonical


def plot_successive_halving(canonical: dict[str, Any]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("PPO v3 successive-halving search on a-d development cells", fontsize=16, fontweight="bold")
    stage_map = [
        ("round1", "Round 1 (3 seeds, 151,552 steps)"),
        ("round2", "Round 2 (5 seeds, 301,056 steps)"),
    ]
    region_titles = {"us": "US", "global": "Global"}
    for row_index, region in enumerate(("us", "global")):
        for col_index, (stage_name, stage_title) in enumerate(stage_map):
            ax = axes[row_index, col_index]
            stage = canonical["successive_halving"][stage_name]["regions"][region]
            labels = [candidate["combo"] for candidate in stage["candidates"]]
            savings = [candidate["summary"]["mean_savings_vs_status_quo_pct"] for candidate in stage["candidates"]]
            safe_counts = [candidate["summary"]["safe_seed_count"] for candidate in stage["candidates"]]
            completions = [candidate["summary"]["minimum_batch_completion"] for candidate in stage["candidates"]]
            colors = []
            for candidate in stage["candidates"]:
                if candidate["combo"] == stage["selected_combo"]:
                    colors.append("#2a6fdb")
                else:
                    colors.append("#b8c4d6")
            bars = ax.bar(labels, savings, color=colors, edgecolor="#2f3e4e", linewidth=0.8)
            ax.axhline(0.0, color="#2f3e4e", linewidth=1.0)
            ax.set_title(f"{region_titles[region]} — {stage_title}", fontsize=12, fontweight="bold")
            ax.set_ylabel("Mean savings vs Status Quo (%)")
            ax.set_ylim(min(min(savings) - 0.5, -0.6), max(max(savings) + 0.5, 2.3))
            for bar, safe_count, completion in zip(bars, safe_counts, completions, strict=True):
                y = bar.get_height()
                va = "bottom" if y >= 0 else "top"
                offset = 0.05 if y >= 0 else -0.05
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y + offset,
                    f"safe {safe_count}/{stage['candidates'][0]['summary']['seed_count']}\nηmin={completion:.6f}",
                    ha="center",
                    va=va,
                    fontsize=8,
                )
            ax.tick_params(axis="x", rotation=0)
    fig.text(
        0.5,
        0.01,
        "Selection order: all seeds safe → safe-seed count → worst-seed cost → mean cost. "
        "All round-1 and round-2 candidates remained below the 0.9999 batch-completion floor.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(
        FIGURE_PATHS["successive_halving"],
        dpi=300,
        metadata={"Software": "build_ppo_v3_results.py", "Title": "PPO v3 successive halving"},
    )
    plt.close(fig)


def plot_budget_curve(canonical: dict[str, Any]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    fig.suptitle("Matched five-seed budget curve for the selected joint configurations", fontsize=16, fontweight="bold")
    colors = {"us": "#c43d3d", "global": "#1f7a6b"}
    labels = ["151,552", "501,760", "1,003,520"]
    x = np.arange(3)
    for region in ("us", "global"):
        budgets = canonical["budget_curve"]["matched_five_seed_curve"][region]["budgets"]
        savings = [budget["summary"]["mean_savings_vs_status_quo_pct"] for budget in budgets]
        completion = [budget["summary"]["minimum_batch_completion"] for budget in budgets]
        safe_counts = [budget["summary"]["safe_seed_count"] for budget in budgets]
        expired = [budget["summary"]["total_expired"] for budget in budgets]
        axes[0].plot(x, savings, marker="o", color=colors[region], linewidth=2.5, label=region.upper())
        axes[1].plot(x, completion, marker="o", color=colors[region], linewidth=2.5, label=region.upper())
        for xi, sv, sc in zip(x, savings, safe_counts, strict=True):
            axes[0].text(xi, sv + 0.06, f"{sc}/5 safe", ha="center", fontsize=8, color=colors[region])
        for xi, comp, exp in zip(x, completion, expired, strict=True):
            label = f"expired {exp:.5f}" if exp > 0 else "expired 0"
            axes[1].text(xi, comp - 0.00006, label, ha="center", fontsize=8, color=colors[region])
    axes[0].axhline(0.0, color="#2f3e4e", linewidth=1.0)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean savings vs Status Quo (%)")
    axes[0].set_title("Cost response")
    axes[0].legend(frameon=True)
    axes[1].axhline(SAFETY_FLOOR, color="#2f3e4e", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.995, 1.00002)
    axes[1].set_ylabel("Minimum batch completion")
    axes[1].set_title("Safety response")
    axes[1].legend(frameon=True)
    fig.text(
        0.5,
        0.01,
        "US degrades with more compute: the 1,003,520-step point falls to -0.241% and 0.995415 completion with 16.69475 expired units. "
        "Global cost improves with compute (+1.764% at 1,003,520) but still never clears the safety floor.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(
        FIGURE_PATHS["budget_curve"],
        dpi=300,
        metadata={"Software": "build_ppo_v3_results.py", "Title": "PPO v3 budget curve"},
    )
    plt.close(fig)


def scatter_panel(
    ax: plt.Axes,
    title: str,
    candidate: dict[str, Any],
    summary: dict[str, Any],
    baseline_cost: float,
    note: str,
) -> None:
    savings = np.asarray([seed["savings_vs_status_quo_pct"] for seed in candidate["seeds"]], dtype=float)
    completion = np.asarray([seed["batch_completion_fraction"] for seed in candidate["seeds"]], dtype=float)
    safe = np.asarray([seed["safe"] for seed in candidate["seeds"]], dtype=bool)
    ax.scatter(savings[~safe], completion[~safe], s=70, color="#d1495b", edgecolor="#5c1f29", linewidth=0.8, label="unsafe")
    ax.scatter(savings[safe], completion[safe], s=80, color="#2a9d8f", edgecolor="#1d5c54", linewidth=0.8, label="safe")
    ci = candidate["optimizer_comparison"]["optimizer_bootstrap_ci95_savings_pct"]
    mean_savings = summary["mean_savings_vs_status_quo_pct"]
    mean_completion = float(np.mean(completion))
    ax.errorbar(
        mean_savings,
        mean_completion,
        xerr=[[mean_savings - ci[0]], [ci[1] - mean_savings]],
        fmt="*",
        markersize=14,
        color="#1f2933",
        capsize=4,
        label="mean ± optimizer CI",
    )
    ax.axvline(0.0, color="#2f3e4e", linewidth=1.0)
    ax.axhline(SAFETY_FLOOR, color="#2f3e4e", linewidth=1.0, linestyle="--")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Savings vs Status Quo (%)")
    ax.set_ylabel("Batch completion fraction")
    ax.set_ylim(min(completion.min() - 0.00025, SAFETY_FLOOR - 0.00035), 1.00003)
    x_pad = max(0.2, (savings.max() - savings.min()) * 0.18)
    ax.set_xlim(savings.min() - x_pad, savings.max() + x_pad)
    ax.text(
        0.02,
        0.04,
        note,
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#94a3b8", "boxstyle": "round,pad=0.35"},
    )
    ax.legend(loc="best", fontsize=8)


def plot_selected_ad(canonical: dict[str, Any]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    fig.suptitle("Selected ten-seed a-d safety/cost results (development cells)", fontsize=16, fontweight="bold")
    region_titles = {"us": "US — R3_P1 @ 151,552", "global": "Global — R0_P3 @ 1,003,520"}
    for ax, region in zip(axes, ("us", "global"), strict=True):
        region_data = canonical["selected_a_d"]["regions"][region]
        summary = region_data["candidate_summary"]
        note = (
            f"{summary['safe_seed_count']}/10 safe\n"
            f"mean {compact_pct(summary['mean_savings_vs_status_quo_pct'])}\n"
            f"ηmin={compact_completion(summary['minimum_batch_completion'])}\n"
            f"expired={summary['total_expired']:.5f}"
        )
        scatter_panel(
            ax,
            region_titles[region],
            region_data["candidate"],
            summary,
            float(region_data["baseline"]["summary"]["total_cost"]),
            note,
        )
    fig.text(
        0.5,
        0.01,
        "Horizontal error bars denote optimizer-seed variability only. Both selected policies fail the 0.9999 completion safety gate despite zero aggregate expiry/backlog in the selected summaries.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(
        FIGURE_PATHS["selected_ad"],
        dpi=300,
        metadata={"Software": "build_ppo_v3_results.py", "Title": "PPO v3 selected a-d results"},
    )
    plt.close(fig)


def plot_negative_net_demand_probe(canonical: dict[str, Any]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    probe = canonical["negative_net_demand_probe"]["regions"]
    regions = ("us", "global")
    labels = ["US", "Global"]
    x = np.arange(len(regions))
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(
        "Negative-net-demand behavioral probe on the selected a-d PPO models",
        fontsize=16,
        fontweight="bold",
    )

    negative_share = [
        probe[region]["aggregate"]["negative_step_share"] for region in regions
    ]
    axes[0, 0].bar(x, negative_share, color=["#3567b7", "#2b8a78"], width=0.55)
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylim(0.0, 0.55)
    axes[0, 0].set_ylabel("Step share")
    axes[0, 0].set_title("Any destination negative-demand step share")
    for xi, value in zip(x, negative_share, strict=True):
        axes[0, 0].text(xi, value + 0.015, f"{value:.6f}", ha="center", fontsize=9)

    drain_negative = [
        probe[region]["aggregate"]["mean_policy_drain_when_any_destination_negative"]
        for region in regions
    ]
    drain_nonnegative = [
        probe[region]["aggregate"]["mean_policy_drain_when_no_destination_negative"]
        for region in regions
    ]
    axes[0, 1].bar(
        x - width / 2,
        drain_negative,
        width,
        color="#5b8def",
        label="any destination negative",
    )
    axes[0, 1].bar(
        x + width / 2,
        drain_nonnegative,
        width,
        color="#b4c9f7",
        label="no destination negative",
    )
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylim(0.45, 0.55)
    axes[0, 1].set_ylabel("Mean decoded drain")
    axes[0, 1].set_title("Policy drain stays near 50%")
    axes[0, 1].legend(loc="best", fontsize=8)

    clearance_negative = [
        probe[region]["aggregate"][
            "mean_actual_pool_clearance_fraction_when_any_destination_negative"
        ]
        for region in regions
    ]
    clearance_nonnegative = [
        probe[region]["aggregate"][
            "mean_actual_pool_clearance_fraction_when_no_destination_negative"
        ]
        for region in regions
    ]
    axes[1, 0].bar(
        x - width / 2,
        clearance_negative,
        width,
        color="#2a9d8f",
        label="any destination negative",
    )
    axes[1, 0].bar(
        x + width / 2,
        clearance_nonnegative,
        width,
        color="#9ad8d0",
        label="no destination negative",
    )
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylim(0.45, 0.55)
    axes[1, 0].set_ylabel("Mean actual pool-clearance fraction")
    axes[1, 0].set_title("Realized pool clearance also stays near 50%")
    axes[1, 0].legend(loc="best", fontsize=8)

    route_negative = [
        probe[region]["aggregate"][
            "mean_batch_routing_share_to_negative_demand_destinations_when_available"
        ]
        for region in regions
    ]
    route_uniform = [
        probe[region]["aggregate"][
            "mean_fraction_destinations_negative_when_available"
        ]
        for region in regions
    ]
    route_excess = [
        probe[region]["aggregate"][
            "excess_batch_routing_share_to_negative_destinations_vs_uniform_when_available"
        ]
        for region in regions
    ]
    high_drain_fraction = [
        probe[region]["aggregate"]["fraction_negative_steps_with_mean_drain_gt_90pct"]
        for region in regions
    ]
    axes[1, 1].bar(
        x - width / 2,
        route_negative,
        width,
        color=["#d17a22", "#8f5cc3"],
        label="actual routed share",
    )
    axes[1, 1].bar(
        x + width / 2,
        route_uniform,
        width,
        color=["#f1c27d", "#c8b3e6"],
        label="uniform benchmark",
    )
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylim(0.0, 0.7)
    axes[1, 1].set_ylabel("Mean routed batch share")
    axes[1, 1].set_title("Routing versus conditioned uniform benchmark")
    axes[1, 1].legend(loc="best", fontsize=8)
    for xi, route_value, uniform_value, excess_value, drain_value in zip(
        x,
        route_negative,
        route_uniform,
        route_excess,
        high_drain_fraction,
        strict=True,
    ):
        axes[1, 1].text(
            xi,
            max(route_value, uniform_value) + 0.03,
            f"actual={route_value:.6f}\nuniform={uniform_value:.6f}\nΔ={100.0 * excess_value:+.2f} pp\n>90% drain={drain_value:.0%}",
            ha="center",
            fontsize=9,
        )

    fig.text(
        0.5,
        0.01,
        "Descriptive only: neither policy increases drain or clearance in negative-demand windows. "
        "US routing is essentially uniform with respect to negative destinations; Global shows only a modest tilt. "
        "Future exact 0–100% drain control requires a hard decoder or explicit override; widening a sigmoid is not exact.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(
        FIGURE_PATHS["negative_probe"],
        dpi=300,
        metadata={
            "Software": "build_ppo_v3_results.py",
            "Title": "PPO v3 negative-net-demand probe",
        },
    )
    plt.close(fig)


def plot_eh_transfer(canonical: dict[str, Any]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    fig.suptitle("Post-selection descriptive e-h transfer check", fontsize=16, fontweight="bold")
    region_titles = {"us": "US — R3_P1 reuse", "global": "Global — R0_P3 reuse"}
    for ax, region in zip(axes, ("us", "global"), strict=True):
        region_data = canonical["descriptive_e_h_transfer"]["regions"][region]
        candidate = region_data["candidate"]
        summary = candidate["summary"]
        note = (
            f"{summary['safe_seed_count']}/10 safe\n"
            f"mean {compact_pct(summary['mean_savings_vs_status_quo_pct'])}\n"
            f"ηmin={compact_completion(summary['minimum_batch_completion'])}\n"
            f"descriptive only"
        )
        scatter_panel(
            ax,
            region_titles[region],
            candidate,
            summary,
            float(region_data["baseline"]["total_cost"]),
            note,
        )
    fig.text(
        0.5,
        0.01,
        "Cells e-h were already exposed in frozen v2. These panels are descriptive transfer checks only and are not confirmatory evidence or a success gate.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(
        FIGURE_PATHS["eh_transfer"],
        dpi=300,
        metadata={"Software": "build_ppo_v3_results.py", "Title": "PPO v3 descriptive e-h transfer"},
    )
    plt.close(fig)


def build_report(canonical: dict[str, Any]) -> str:
    lines = [
        "# PPO v3 canonical analysis package",
        "",
        "> 126 frozen PPO model artifacts; joint temporal+spatial PPO only; no new spatial-only, temporal-only, or MPC runs.",
        "",
        "All intervals below describe optimizer-seed variability only. The a-d results are development-set diagnostics, and the e-h results are descriptive transfer checks only.",
        "",
        "## Frozen design",
        "",
        "- Primary comparator: deterministic **Status Quo**. **Round Robin** and **Drain Immediately** remain diagnostics only.",
        "- Full-dollar evaluation stayed fixed at service/batch weights **1000 / 1000** even when training reward weights changed.",
        "- Successive-halving budgets: **36 models at 151,552**, **40 models at 301,056**, **20 ten-seed full models at 501,760**, plus matched budget reuses/separate replicas for the selected configurations.",
        "- Selected configurations: **US R3_P1 @ 151,552** and **Global R0_P3 @ 1,003,520**.",
        "- No fresh confirmatory data exist in this protocol: a-d drove selection, and e-h was already exposed in frozen v2.",
        "",
        "## Frozen verdict",
        "",
        "**Both selected policies fail the frozen safety gate on a-d.** The protocol's required outcome therefore remains: report the joint PPO recovery as unsuccessful and do not activate a spatial-only, temporal-only, or MPC controller under this protocol.",
        "",
        f"- **US:** 1/10 safe, mean savings {compact_pct(canonical['selected_a_d']['regions']['us']['candidate_summary']['mean_savings_vs_status_quo_pct'])}, optimizer CI {compact_usd(canonical['selected_a_d']['regions']['us']['candidate']['optimizer_comparison']['optimizer_bootstrap_ci95_usd'][0])} to {compact_usd(canonical['selected_a_d']['regions']['us']['candidate']['optimizer_comparison']['optimizer_bootstrap_ci95_usd'][1])}, minimum completion {compact_completion(canonical['selected_a_d']['regions']['us']['candidate_summary']['minimum_batch_completion'])}, zero aggregate expiry/backlog.",
        f"- **Global:** 1/10 safe, mean savings {compact_pct(canonical['selected_a_d']['regions']['global']['candidate_summary']['mean_savings_vs_status_quo_pct'])}, optimizer CI {compact_usd(canonical['selected_a_d']['regions']['global']['candidate']['optimizer_comparison']['optimizer_bootstrap_ci95_usd'][0])} to {compact_usd(canonical['selected_a_d']['regions']['global']['candidate']['optimizer_comparison']['optimizer_bootstrap_ci95_usd'][1])}, minimum completion {compact_completion(canonical['selected_a_d']['regions']['global']['candidate_summary']['minimum_batch_completion'])}, zero aggregate expiry/backlog.",
        "",
        "## Successive-halving search",
        "",
        "| Region | Round | Combo | Mean savings | Worst savings | Safe seeds | Min completion | Selected |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for stage_name, label in (("round1", "151,552"), ("round2", "301,056")):
        stage = canonical["successive_halving"][stage_name]
        for region in ("us", "global"):
            selected_combo = stage["regions"][region]["selected_combo"]
            for candidate in stage["regions"][region]["candidates"]:
                summary = candidate["summary"]
                lines.append(
                    f"| {region.upper()} | {label} | {candidate['combo']} | "
                    f"{compact_pct(summary['mean_savings_vs_status_quo_pct'])} | "
                    f"{compact_pct(summary['worst_savings_vs_status_quo_pct'])} | "
                    f"{summary['safe_seed_count']}/{summary['seed_count']} | "
                    f"{summary['minimum_batch_completion']:.6f} | "
                    f"{'yes' if candidate['combo'] == selected_combo else ''} |"
                )
    lines += [
        "",
        "Round 1 promoted **R3** for US and **R0** for Global. Round 2 then selected **R3_P1** and **R0_P3**. No round-1 or round-2 candidate achieved all-seeds-safe status.",
        "",
        "## Matched budget curve",
        "",
        "| Region | Steps | Mean savings | Safe seeds | Min completion | Total expired |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for region in ("us", "global"):
        for budget in canonical["budget_curve"]["matched_five_seed_curve"][region]["budgets"]:
            summary = budget["summary"]
            lines.append(
                f"| {region.upper()} | {budget['timesteps']:,} | "
                f"{compact_pct(summary['mean_savings_vs_status_quo_pct'])} | "
                f"{summary['safe_seed_count']}/{summary['seed_count']} | "
                f"{summary['minimum_batch_completion']:.6f} | "
                f"{summary['total_expired']:.5f} |"
            )
    lines += [
        "",
        "- **US worsens with more compute:** the matched five-seed 1,003,520-step point falls to **-0.241%**, minimum completion **0.995415**, and **16.69475** expired units.",
        "- **Global cost improves with more compute:** the matched five-seed 1,003,520-step point reaches **+1.764%**, but still fails the safety floor in **5/5** seeds.",
        "- The separate 501,760-step ten-seed replication also remains unsafe: US **0/10** safe, Global **1/10** safe.",
        "",
        "## Final selected a-d results",
        "",
        "| Region | Selected config | Mean cost | Mean savings | Optimizer CI (USD) | Safe seeds | Min completion | Max terminal pool | Passed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for region in ("us", "global"):
        region_data = canonical["selected_a_d"]["regions"][region]
        summary = region_data["candidate_summary"]
        comparison = region_data["candidate"]["optimizer_comparison"]
        lines.append(
            f"| {region.upper()} | {region_data['combo']} @ {region_data['candidate']['timesteps']:,} | "
            f"{compact_millions(summary['mean_total_cost'])} | "
            f"{compact_pct(summary['mean_savings_vs_status_quo_pct'])} | "
            f"{compact_usd(comparison['optimizer_bootstrap_ci95_usd'][0])} to {compact_usd(comparison['optimizer_bootstrap_ci95_usd'][1])} | "
            f"{summary['safe_seed_count']}/{summary['seed_count']} | "
            f"{summary['minimum_batch_completion']:.6f} | "
            f"{summary['maximum_terminal_batch_pool']:.6f} | "
            f"{'yes' if region_data['passed'] else 'no'} |"
        )
    lines += [
        "",
        "## Negative-net-demand behavioral probe",
        "",
        "This deterministic diagnostic reconstructs the a-d evaluation environment and reruns all ten selected models per region with domain randomization disabled. It is descriptive only and supports no causal or confirmatory claim.",
        "",
        "| Region | Negative-step share | Mean drain if any destination negative | Mean drain otherwise | Mean actual clearance if any destination negative | Mean actual clearance otherwise | Actual routed batch share to negative destinations | Conditioned uniform benchmark | Actual - benchmark | Negative steps with mean drain >90% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in ("us", "global"):
        aggregate = canonical["negative_net_demand_probe"]["regions"][region]["aggregate"]
        lines.append(
            f"| {region.upper()} | "
            f"{aggregate['negative_step_share']:.6f} | "
            f"{aggregate['mean_policy_drain_when_any_destination_negative']:.6f} | "
            f"{aggregate['mean_policy_drain_when_no_destination_negative']:.6f} | "
            f"{aggregate['mean_actual_pool_clearance_fraction_when_any_destination_negative']:.6f} | "
            f"{aggregate['mean_actual_pool_clearance_fraction_when_no_destination_negative']:.6f} | "
            f"{aggregate['mean_batch_routing_share_to_negative_demand_destinations_when_available']:.6f} | "
            f"{aggregate['mean_fraction_destinations_negative_when_available']:.6f} | "
            f"{100.0 * aggregate['excess_batch_routing_share_to_negative_destinations_vs_uniform_when_available']:+.2f} pp | "
            f"{aggregate['fraction_negative_steps_with_mean_drain_gt_90pct']:.6f} |"
        )
    us_probe = canonical["negative_net_demand_probe"]["regions"]["us"]["aggregate"]
    global_probe = canonical["negative_net_demand_probe"]["regions"]["global"]["aggregate"]
    lines += [
        "",
        "Neither policy increases drain or realized pool-clearance in negative-demand windows. "
        "On conditioned any-negative steps, US actual routing to negative-demand destinations "
        f"(**{us_probe['mean_batch_routing_share_to_negative_demand_destinations_when_available']:.6f}**) is essentially uniform relative to the "
        f"negative-destination benchmark (**{us_probe['mean_fraction_destinations_negative_when_available']:.6f}**, "
        f"{100.0 * us_probe['excess_batch_routing_share_to_negative_destinations_vs_uniform_when_available']:+.2f} percentage points). "
        "Global shows only a modest positive spatial tilt "
        f"(**{global_probe['mean_batch_routing_share_to_negative_demand_destinations_when_available']:.6f}** versus "
        f"**{global_probe['mean_fraction_destinations_negative_when_available']:.6f}**, "
        f"{100.0 * global_probe['excess_batch_routing_share_to_negative_destinations_vs_uniform_when_available']:+.2f} percentage points). "
        "In both regions, **no** negative-demand step has mean drain above **90%**.",
        "",
        "Actual pool-clearance fraction here means `total batch drained / pre-drain batch pool`, with `pre-drain batch pool = post-step batch_pool_size + batch_drained`.",
        "",
        "## Descriptive e-h transfer check",
        "",
        "**Do not treat this section as confirmatory.** Cells e-h were already exposed in frozen v2, so these results are descriptive transfer only.",
        "",
        "| Region | Reused config | Mean savings | Optimizer CI (USD) | Safe seeds | Min completion |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for region in ("us", "global"):
        region_data = canonical["descriptive_e_h_transfer"]["regions"][region]
        candidate = region_data["candidate"]
        summary = candidate["summary"]
        comparison = candidate["optimizer_comparison"]
        lines.append(
            f"| {region.upper()} | {region_data['combo']} | "
            f"{compact_pct(summary['mean_savings_vs_status_quo_pct'])} | "
            f"{compact_usd(comparison['optimizer_bootstrap_ci95_usd'][0])} to {compact_usd(comparison['optimizer_bootstrap_ci95_usd'][1])} | "
            f"{summary['safe_seed_count']}/{summary['seed_count']} | "
            f"{summary['minimum_batch_completion']:.6f} |"
        )
    lines += [
        "",
        "The exact descriptive transfer means are **-0.155% for US** and **+2.826% for Global**; these numbers are reported honestly but are not a gate or headline.",
        "",
        "## Context versus frozen v2",
        "",
        f"- Frozen v2 headline support: **{str(canonical['v2_context']['headline_joint_shaping_supported']).upper()}**. {canonical['v2_context']['primary_finding']}",
        "- The v2 joint result was conditional on a state representation that omitted episode position whenever demand charge was off. V3 adds episode progress and deadline buckets, but the deadline-boundary audit below shows that this repair is incomplete. This does **not** prove causality.",
        f"- {canonical['v2_context']['v2_reward_scale_confound_limitation']}",
        "- Equal 100 MW / unit-capacity proxy sites remove real fleet-size heterogeneity and may reduce US opportunity.",
        "",
        "## Remaining deadline-boundary observability limitation",
        "",
        canonical["deadline_boundary_diagnostic"]["defect"],
        "",
        "| Region | Selected training seeds with expiry | Training expiry across all episodes | a-d eval seeds with expiry | e-h eval seeds with expiry |",
        "|---|---:|---:|---:|---:|",
    ]
    for region in ("us", "global"):
        boundary = canonical["deadline_boundary_diagnostic"]["regions"][region]
        lines.append(
            f"| {region.upper()} | "
            f"{boundary['training_seeds_with_expiry']}/{boundary['selected_training_seed_count']} | "
            f"{boundary['training_expired_total_across_all_episodes']:.6f} | "
            f"{boundary['selected_a_d_evaluation_seeds_with_expiry']}/10 | "
            f"{boundary['descriptive_e_h_evaluation_seeds_with_expiry']}/10 |"
        )
    lines += [
        "",
        canonical["deadline_boundary_diagnostic"]["scope"],
        "",
        f"**Next-protocol fix:** {canonical['deadline_boundary_diagnostic']['next_protocol_fix']}",
        "",
        "## Diagnostic notes",
        "",
        f"- {canonical['tests_and_accounting_context']}",
        f"- Action cap diagnostic: max single-step learned drain = {100.0 * canonical['action_cap_diagnostic']['max_drain_fraction']:.3f}% (sigmoid(+3)); four repeated steps still permit {100.0 * canonical['action_cap_diagnostic']['four_step_completion_ceiling']:.4f}% theoretical clearance, and selected mean saturation remained zero.",
        f"- {canonical['negative_net_demand_probe']['future_control_note']}",
        "",
        "## Provenance",
        "",
        f"- Protocol hash: `{canonical['model_provenance']['protocol_sha256']}`",
        f"- Model manifest hash: `{canonical['model_provenance']['manifest_sha256']}`",
        f"- Model count: **{canonical['model_provenance']['model_count']}**",
        f"- Completion-record count: **{canonical['model_provenance']['complete_record_count']}**",
        f"- Stage counts: `{json.dumps(canonical['model_provenance']['stage_counts'], sort_keys=True)}`",
        "- Exact manifest entries, stage-group hashes, and selected model completion-record hashes are preserved in `canonical_results.json`.",
        "",
        "## Figures",
        "",
        "![Successive halving](ppo_v3_successive_halving.png)",
        "",
        "![Budget curve](ppo_v3_budget_curve.png)",
        "",
        "![Selected a-d results](ppo_v3_selected_ad_safety_cost.png)",
        "",
        "![Negative-net-demand probe](ppo_v3_negative_net_demand_probe.png)",
        "",
        "![Descriptive e-h transfer](ppo_v3_eh_transfer.png)",
    ]
    return "\n".join(lines)


def build_outputs() -> dict[str, str]:
    canonical = build_canonical()
    canonical_hash = write_canonical_json(V3_ROOT / "canonical_results.json", canonical)
    report_hash = write_text(V3_ROOT / "results_report.md", build_report(canonical))
    plot_successive_halving(canonical)
    plot_budget_curve(canonical)
    plot_selected_ad(canonical)
    plot_negative_net_demand_probe(canonical)
    plot_eh_transfer(canonical)
    figure_hashes = {path.name: sha256_file(path) for path in FIGURE_PATHS.values()}
    return {
        "canonical_results_sha256": canonical_hash,
        "results_report_sha256": report_hash,
        **dict(sorted(figure_hashes.items())),
    }


def main() -> None:
    hashes = build_outputs()
    print(json.dumps(hashes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
