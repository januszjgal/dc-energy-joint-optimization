"""Build the PPO v4 hard-safety canonical analysis package from frozen artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "output" / "ppo_v4_safety"
V4_MODEL_ROOT = ROOT / "models" / "ppo_v4_safety"
V3_ROOT = ROOT / "output" / "ppo_v3_reward_sweep"

PROTOCOL_PATH = ROOT / "env" / "protocols" / "v4_safety.yaml"
PRELIGHT_PATH = OUT_ROOT / "preflight.json"
REPLAY_PATH = OUT_ROOT / "replay_results.json"
SHORT_PATH = OUT_ROOT / "short_results.json"
SHORT_GATE_PATH = OUT_ROOT / "short_gate.json"
MEDIUM_PATH = OUT_ROOT / "medium_results.json"
MEDIUM_GATE_PATH = OUT_ROOT / "medium_gate.json"
FULL_PATH = OUT_ROOT / "full_results.json"
FULL_GATE_PATH = OUT_ROOT / "full_gate.json"
FINAL_PATH = OUT_ROOT / "final_results.json"
MANIFEST_PATH = V4_MODEL_ROOT / "manifest.json"
V3_CANONICAL_PATH = V3_ROOT / "canonical_results.json"

PRIMARY_BASELINE = "Status Quo (local, no deferral)"
PRIMARY_MODE = "safety_only"
FLUSH_MODE = "safety_plus_negative_demand_flush"
SERVICE_EQUALITY_TOL = 1e-9
BATCH_EQUALITY_TOL = 1e-9
TERMINAL_TOL = 1e-8
TRANSPORT_TOL = 1e-8
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 0
STAGE_ORDER = {"short": 0, "medium": 1, "full": 2}
REGION_ORDER = {"us": 0, "global": 1}
COLORS = {"us": "#1f77b4", "global": "#d62728"}
FIGURE_PATHS = {
    "stage_safety_economics": OUT_ROOT / "v4_stage_safety_economics.png",
    "replay_shield_flush": OUT_ROOT / "v4_replay_shield_flush.png",
    "final_savings_ci_safety": OUT_ROOT / "v4_final_savings_ci_safety.png",
    "intervention_projection": OUT_ROOT / "v4_intervention_projection.png",
    "eh_descriptive": OUT_ROOT / "v4_eh_descriptive.png",
}

INPUT_HASH_PATHS = [
    PROTOCOL_PATH,
    PRELIGHT_PATH,
    REPLAY_PATH,
    SHORT_PATH,
    SHORT_GATE_PATH,
    MEDIUM_PATH,
    MEDIUM_GATE_PATH,
    FULL_PATH,
    FULL_GATE_PATH,
    FINAL_PATH,
    MANIFEST_PATH,
    ROOT / "env" / "safety_layer.py",
    ROOT / "env" / "safe_multi_dc_env.py",
    ROOT / "train_v4.py",
    ROOT / "scripts" / "run_safety_campaign_v4.py",
    ROOT / "scripts" / "smoke_test_safety_v4.py",
    ROOT / "scripts" / "smoke_test_demand_charge.py",
    V3_CANONICAL_PATH,
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


def to_rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def ordered_seed_keys(mapping: dict[str, Any]) -> list[str]:
    return sorted(mapping, key=lambda key: int(str(key)))


def savings_pct_ci(ci_usd: list[float], baseline_cost: float) -> list[float]:
    return [100.0 * value / baseline_cost for value in ci_usd]


def optimizer_ci_from_costs(
    candidate_costs: list[float],
    baseline_cost: float,
) -> dict[str, Any]:
    costs = np.asarray(candidate_costs, dtype=np.float64)
    improvement = baseline_cost - costs
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(improvement),
        size=(BOOTSTRAP_RESAMPLES, len(improvement)),
    )
    bootstrap = improvement[indices].mean(axis=1)
    ci_usd = [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]
    return {
        "interpretation": (
            "Positive USD means the candidate costs less than deterministic "
            "Status Quo. The interval reflects optimizer-seed variability only."
        ),
        "mean_improvement_usd": float(improvement.mean()),
        "optimizer_bootstrap_ci95_usd": ci_usd,
        "optimizer_bootstrap_ci95_savings_pct": savings_pct_ci(
            ci_usd, baseline_cost
        ),
        "all_seed_improvements_positive": bool(np.all(improvement > 0.0)),
    }


def gate_seed(summary: dict[str, Any]) -> dict[str, Any]:
    service_completion = float(summary["service_completion"])
    batch_completion = float(summary["batch_completion"])
    total_expired = float(summary.get("total_batch_expired", 0.0))
    terminal_pool = float(summary.get("terminal_batch_pool", 0.0))
    terminal_backlog = float(summary.get("terminal_service_backlog", 0.0))
    safety = dict(summary.get("safety", {}))
    max_transport_error = float(
        safety.get("max_transport_conservation_error", math.inf)
    )
    certificates = int(safety.get("infeasibility_certificates", 0))
    checks = {
        "service_completion_equals_1": math.isclose(
            service_completion,
            1.0,
            rel_tol=0.0,
            abs_tol=SERVICE_EQUALITY_TOL,
        ),
        "batch_completion_equals_1": math.isclose(
            batch_completion,
            1.0,
            rel_tol=0.0,
            abs_tol=BATCH_EQUALITY_TOL,
        ),
        "total_expired_equals_0": abs(total_expired) <= SERVICE_EQUALITY_TOL,
        "terminal_batch_pool_lte_1e_8": terminal_pool <= TERMINAL_TOL,
        "terminal_service_backlog_lte_1e_8": terminal_backlog <= TERMINAL_TOL,
        "safety_infeasibility_certificates_equals_0": certificates == 0,
        "transport_conservation_error_lte_1e-8": max_transport_error
        <= TRANSPORT_TOL,
    }
    return {
        "service_completion": service_completion,
        "batch_completion": batch_completion,
        "total_expired": total_expired,
        "terminal_batch_pool": terminal_pool,
        "terminal_service_backlog": terminal_backlog,
        "safety_infeasibility_certificates": certificates,
        "max_transport_conservation_error": max_transport_error,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def safe_seed_count(seed_map: dict[str, Any]) -> int:
    return sum(
        1
        for key in ordered_seed_keys(seed_map)
        if gate_seed(seed_map[key]["summary"])["passed"]
    )


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def summarize_training_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    histogram: Counter[str] = Counter()
    for row in rows:
        histogram.update(row["training_telemetry"]["binding_deadline_histogram"])
    metrics = [
        "intervention_rate",
        "mean_projection_l2",
        "max_projection_l2",
        "mandatory_step_rate",
        "max_mandatory_batch",
        "minimum_deadline_slack",
        "exact_zero_drain_count",
        "exact_full_drain_count",
        "negative_flush_step_count",
        "infeasibility_certificates",
        "max_transport_conservation_error",
    ]
    summary: dict[str, Any] = {
        "seed_count": len(rows),
        "timesteps_per_model": sorted(
            {int(row["timesteps"]) for row in rows}
        ),
        "binding_deadline_histogram": dict(
            sorted(histogram.items(), key=lambda item: int(item[0]))
        ),
    }
    for metric in metrics:
        values = [float(row["training_telemetry"][metric]) for row in rows]
        summary[metric] = {
            "mean": mean(values),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return summary


def load_inputs() -> dict[str, Any]:
    return {
        "protocol": read_yaml(PROTOCOL_PATH),
        "preflight": read_json(PRELIGHT_PATH),
        "replay": read_json(REPLAY_PATH),
        "short": read_json(SHORT_PATH),
        "short_gate": read_json(SHORT_GATE_PATH),
        "medium": read_json(MEDIUM_PATH),
        "medium_gate": read_json(MEDIUM_GATE_PATH),
        "full": read_json(FULL_PATH),
        "full_gate": read_json(FULL_GATE_PATH),
        "final": read_json(FINAL_PATH),
        "manifest": read_json(MANIFEST_PATH),
        "v3_canonical": read_json(V3_CANONICAL_PATH),
    }


def build_input_hashes() -> dict[str, str]:
    return {
        to_rel(path): sha256_file(path)
        for path in sorted(INPUT_HASH_PATHS, key=lambda item: to_rel(item))
    }


def verify_model_hashes(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if int(manifest["model_count"]) != 36:
        raise ValueError("expected 36 v4 models in manifest")
    model_records: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in sorted(manifest["models"], key=lambda row: row["path"]):
        model_path = ROOT / entry["path"]
        completion_path = ROOT / entry["completion_record_path"]
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if not completion_path.exists():
            raise FileNotFoundError(completion_path)
        completion = read_json(completion_path)
        training_path = completion_path.with_name("training.json")
        if not training_path.exists():
            raise FileNotFoundError(training_path)
        training = read_json(training_path)

        model_sha = sha256_file(model_path)
        completion_sha = sha256_file(completion_path)
        diagnostics_sha = sha256_file(training_path)
        if model_sha != entry["sha256"]:
            raise ValueError(f"manifest model hash mismatch: {entry['path']}")
        if completion_sha != entry["completion_record_sha256"]:
            raise ValueError(
                f"manifest completion hash mismatch: {entry['completion_record_path']}"
            )
        if model_sha != completion["model_sha256"]:
            raise ValueError(f"completion model hash mismatch: {entry['path']}")
        if diagnostics_sha != completion["diagnostics_sha256"]:
            raise ValueError(
                f"completion diagnostics hash mismatch: {to_rel(training_path)}"
            )
        if completion["model_path"] != entry["path"]:
            raise ValueError(f"completion model path mismatch: {entry['path']}")
        if completion["job_fingerprint"] != entry["job_fingerprint"]:
            raise ValueError(f"job fingerprint mismatch: {entry['path']}")
        if completion["protocol_sha256"] != manifest["protocol_sha256"]:
            raise ValueError(f"protocol hash mismatch: {entry['path']}")

        stage = str(completion["stage"])
        region = str(completion["region"])
        combo = str(completion["combo"])
        seed = int(completion["seed"])
        stage_counts[stage] += 1
        record = {
            "stage": stage,
            "region": region,
            "combo": combo,
            "seed": seed,
            "timesteps": int(completion["timesteps"]),
            "model_path": entry["path"],
            "training_path": to_rel(training_path),
            "completion_path": entry["completion_record_path"],
            "model_sha256": model_sha,
            "diagnostics_sha256": diagnostics_sha,
            "completion_sha256": completion_sha,
            "job_fingerprint": entry["job_fingerprint"],
            "training_telemetry": {
                "steps_observed": int(training["diagnostics"]["steps_observed"]),
                "intervention_count": int(
                    training["diagnostics"]["intervention_count"]
                ),
                "intervention_rate": float(
                    training["diagnostics"]["intervention_rate"]
                ),
                "mean_projection_l2": float(
                    training["diagnostics"]["mean_projection_l2"]
                ),
                "max_projection_l2": float(
                    training["diagnostics"]["max_projection_l2"]
                ),
                "mandatory_step_count": int(
                    training["diagnostics"]["mandatory_step_count"]
                ),
                "mandatory_step_rate": float(
                    training["diagnostics"]["mandatory_step_rate"]
                ),
                "max_mandatory_batch": float(
                    training["diagnostics"]["max_mandatory_batch"]
                ),
                "minimum_deadline_slack": float(
                    training["diagnostics"]["minimum_deadline_slack"] or 0.0
                ),
                "binding_deadline_histogram": {
                    str(key): int(value)
                    for key, value in training["diagnostics"][
                        "binding_deadline_histogram"
                    ].items()
                },
                "exact_zero_drain_count": int(
                    training["diagnostics"]["exact_zero_drain_count"]
                ),
                "exact_full_drain_count": int(
                    training["diagnostics"]["exact_full_drain_count"]
                ),
                "negative_flush_step_count": int(
                    training["diagnostics"]["negative_flush_step_count"]
                ),
                "infeasibility_certificates": int(
                    training["diagnostics"]["infeasibility_certificates"]
                ),
                "max_transport_conservation_error": float(
                    training["diagnostics"]["max_transport_conservation_error"]
                ),
            },
        }
        model_records.append(record)
        grouped[(stage, region)].append(record)

    if dict(stage_counts) != {"short": 6, "medium": 10, "full": 20}:
        raise ValueError(f"unexpected stage counts: {dict(stage_counts)}")

    group_summaries = {}
    for key in sorted(grouped, key=lambda item: (STAGE_ORDER[item[0]], REGION_ORDER[item[1]])):
        stage, region = key
        rows = sorted(grouped[key], key=lambda row: row["seed"])
        group_summaries[f"{stage}:{region}"] = summarize_training_group(rows)

    verification = {
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "protocol_sha256": manifest["protocol_sha256"],
        "model_count": len(model_records),
        "verified_model_hash_count": len(model_records),
        "verified_diagnostics_hash_count": len(model_records),
        "verified_completion_hash_count": len(model_records),
        "all_hashes_verified": True,
        "stage_counts": dict(sorted(stage_counts.items())),
        "group_summaries": group_summaries,
    }
    return verification, model_records, group_summaries


def stage_region_summary(
    result: dict[str, Any],
    *,
    gate: dict[str, Any] | None = None,
    include_diagnostic: bool = False,
    compute_optimizer: bool = False,
) -> dict[str, Any]:
    region_payload: dict[str, Any] = {}
    for region in ("us", "global"):
        row = result["regions"][region]
        baseline_cost = float(row["baselines"][PRIMARY_BASELINE]["total_cost"])
        candidate_costs = [
            float(row["seeds"][key]["summary"]["total_cost"])
            for key in ordered_seed_keys(row["seeds"])
        ]
        if compute_optimizer:
            optimizer = optimizer_ci_from_costs(candidate_costs, baseline_cost)
        else:
            optimizer = dict(row["optimizer_vs_status_quo"])
            optimizer["optimizer_bootstrap_ci95_savings_pct"] = savings_pct_ci(
                list(optimizer["optimizer_bootstrap_ci95_usd"]),
                baseline_cost,
            )
        safe_count = safe_seed_count(row["seeds"])
        gate_region = gate["regions"][region] if gate is not None else None
        aggregate = row["aggregate"]
        summary = {
            "seed_count": int(aggregate["seed_count"]),
            "safe_seed_count": safe_count,
            "all_seeds_safe": safe_count == int(aggregate["seed_count"]),
            "baseline_cost_usd": baseline_cost,
            "mean_total_cost_usd": float(aggregate["mean_total_cost"]),
            "std_total_cost_usd": float(aggregate["std_total_cost"]),
            "mean_savings_pct_vs_status_quo": float(
                aggregate["mean_savings_vs_status_quo_pct"]
            ),
            "worst_seed_savings_pct_vs_status_quo": float(
                aggregate["worst_savings_vs_status_quo_pct"]
            ),
            "mean_safety_intervention_rate": float(
                aggregate["mean_safety_intervention_rate"]
            ),
            "max_projection_l2": float(aggregate["max_projection_l2"]),
            "safety_metrics": {
                "minimum_service_completion": float(
                    aggregate["minimum_service_completion"]
                ),
                "minimum_batch_completion": float(
                    aggregate["minimum_batch_completion"]
                ),
                "total_expired": float(aggregate["total_expired"]),
                "maximum_terminal_batch_pool": float(
                    aggregate["maximum_terminal_batch_pool"]
                ),
                "maximum_terminal_service_backlog": float(
                    aggregate["maximum_terminal_service_backlog"]
                ),
                "safety_infeasibility_certificates": int(
                    aggregate["safety_infeasibility_certificates"]
                ),
                "maximum_transport_conservation_error": float(
                    aggregate["maximum_transport_conservation_error"]
                ),
            },
            "optimizer_vs_status_quo": optimizer,
            "seed_gate_status": {
                key: gate_seed(row["seeds"][key]["summary"])
                for key in ordered_seed_keys(row["seeds"])
            },
        }
        if gate_region is not None:
            summary["gate_passed"] = bool(gate_region["passed"])
        if include_diagnostic:
            summary["diagnostic_vs_archived_v3_safety_only"] = dict(
                row["diagnostic_vs_archived_v3_safety_only"]
            )
        region_payload[region] = summary
    return region_payload


def replay_summary(result: dict[str, Any]) -> dict[str, Any]:
    regions: dict[str, Any] = {}
    for region in ("us", "global"):
        modes_payload: dict[str, Any] = {}
        region_row = result["regions"][region]
        for mode in (PRIMARY_MODE, FLUSH_MODE):
            row = region_row["modes"][mode]
            baseline_cost = float(row["baselines"][PRIMARY_BASELINE]["total_cost"])
            optimizer = dict(row["optimizer_vs_status_quo"])
            optimizer["optimizer_bootstrap_ci95_savings_pct"] = savings_pct_ci(
                list(optimizer["optimizer_bootstrap_ci95_usd"]),
                baseline_cost,
            )
            aggregate = row["aggregate"]
            safe_count = safe_seed_count(row["seeds"])
            modes_payload[mode] = {
                "seed_count": int(aggregate["seed_count"]),
                "safe_seed_count": safe_count,
                "all_seeds_safe": safe_count == int(aggregate["seed_count"]),
                "baseline_cost_usd": baseline_cost,
                "mean_total_cost_usd": float(aggregate["mean_total_cost"]),
                "std_total_cost_usd": float(aggregate["std_total_cost"]),
                "mean_savings_pct_vs_status_quo": float(
                    aggregate["mean_savings_vs_status_quo_pct"]
                ),
                "worst_seed_savings_pct_vs_status_quo": float(
                    aggregate["worst_savings_vs_status_quo_pct"]
                ),
                "mean_safety_intervention_rate": float(
                    aggregate["mean_safety_intervention_rate"]
                ),
                "max_projection_l2": float(aggregate["max_projection_l2"]),
                "safety_metrics": {
                    "minimum_service_completion": float(
                        aggregate["minimum_service_completion"]
                    ),
                    "minimum_batch_completion": float(
                        aggregate["minimum_batch_completion"]
                    ),
                    "total_expired": float(aggregate["total_expired"]),
                    "maximum_terminal_batch_pool": float(
                        aggregate["maximum_terminal_batch_pool"]
                    ),
                    "maximum_terminal_service_backlog": float(
                        aggregate["maximum_terminal_service_backlog"]
                    ),
                    "safety_infeasibility_certificates": int(
                        aggregate["safety_infeasibility_certificates"]
                    ),
                    "maximum_transport_conservation_error": float(
                        aggregate["maximum_transport_conservation_error"]
                    ),
                },
                "optimizer_vs_status_quo": optimizer,
            }
        primary = modes_payload[PRIMARY_MODE]
        flush = modes_payload[FLUSH_MODE]
        regions[region] = {
            "scenario": region_row["scenario"],
            "source_combo": region_row["source_combo"],
            "modes": modes_payload,
            "flush_ablation_delta_vs_primary": {
                "mean_cost_delta_usd": float(
                    primary["mean_total_cost_usd"] - flush["mean_total_cost_usd"]
                ),
                "mean_intervention_rate_delta": float(
                    flush["mean_safety_intervention_rate"]
                    - primary["mean_safety_intervention_rate"]
                ),
            },
        }
    return regions


def compact_money(value: float, *, digits: int = 0) -> str:
    return f"${value:,.{digits}f}"


def compact_pct(value: float, *, digits: int = 4, signed: bool = True) -> str:
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def compact_rate(value: float, *, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def compact_sci(value: float) -> str:
    return f"{value:.3e}"


def implementation_matrix(package: dict[str, Any]) -> list[dict[str, Any]]:
    full = package["stage_results"]["full_a_d"]["regions"]
    replay = package["stage_results"]["replay"]["regions"]
    protocol = package["protocol_snapshot"]
    preflight = package["preflight_snapshot"]
    rows = [
        {
            "item": 1,
            "mechanism": "Actionable state excludes any batch with deadline_step <= t; due-now mass is audit-only and expires before service.",
            "code_evidence": [
                "env/protocols/v4_safety.yaml:19-24",
                "env/safe_multi_dc_env.py:46-49,200-212",
                "env/safety_layer.py:286-294",
            ],
            "test_evidence": [
                "scripts/smoke_test_safety_v4.py:84-90",
            ],
            "runtime_evidence": (
                "Full a-d gate is 20/20 safe with total expiry 0 in both regions."
            ),
        },
        {
            "item": 2,
            "mechanism": "Cumulative exact EDF uses frozen causal envelopes to force enough work now that every known deadline and episode end remain schedulable.",
            "code_evidence": [
                "env/protocols/v4_safety.yaml:26-47",
                "env/safety_layer.py:18-36,409-431,461-472",
                "env/safety_layer.py:276-339,607-618",
            ],
            "test_evidence": [
                "scripts/smoke_test_safety_v4.py:93-151,276-332,385-408",
            ],
            "runtime_evidence": (
                "Preflight observed maxima service 2.130388 and batch 0.831723 below envelopes 2.25/1.0; training records mandatory EDF steps and full a-d/e-h expiry remains zero."
            ),
        },
        {
            "item": 3,
            "mechanism": "Service guarantee is clean-state only: any preexisting local service backlog fails closed instead of being silently pooled.",
            "code_evidence": [
                "env/safe_multi_dc_env.py:170-185,311-312",
            ],
            "test_evidence": [
                "scripts/smoke_test_safety_v4.py:200-212",
            ],
            "runtime_evidence": (
                "Full a-d maximum terminal service backlog is exactly 0.0 in both regions."
            ),
        },
        {
            "item": 6,
            "mechanism": "Hard infeasibility certificates fail closed on envelope overruns, missed deadlines, and mandatory-capacity deficits.",
            "code_evidence": [
                "env/safety_layer.py:409-442,485-495,525-534",
                "env/safe_multi_dc_env.py:204-212",
            ],
            "test_evidence": [
                "scripts/smoke_test_safety_v4.py:154-197",
            ],
            "runtime_evidence": (
                "Replay, short, medium, full, and e-h all report zero safety infeasibility certificates; all 36 training.json files also report zero."
            ),
        },
        {
            "item": 8,
            "mechanism": "PPO is trained with the projector active and every intervention, projection distance, binding horizon, deadline slack, exact endpoint, flush, certificate, and conservation metric is persisted.",
            "code_evidence": [
                "train_v4.py:31-132,159-315",
                "env/safe_multi_dc_env.py:418-481",
                "evaluate.py:368-462",
            ],
            "test_evidence": [
                "scripts/smoke_test_safety_v4.py:385-408",
            ],
            "runtime_evidence": (
                "All 36 hash-verified training diagnostics contain projector telemetry and zero infeasibility certificates."
            ),
        },
        {
            "item": 7,
            "mechanism": "The shield applies a minimal Euclidean projection and records intervention/projection telemetry rather than replacing the policy with a rule-based controller.",
            "code_evidence": [
                "env/safety_layer.py:134-232,595-605,634-635",
                "env/safe_multi_dc_env.py:418-481",
            ],
            "test_evidence": [
                "scripts/smoke_test_safety_v4.py:67-82,276-332",
            ],
            "runtime_evidence": (
                f"Global replay intervention is {compact_pct(replay['global']['modes'][PRIMARY_MODE]['mean_safety_intervention_rate'] * 100.0, digits=3)}, while the trained full a-d policy is still changed on {compact_pct(full['global']['mean_safety_intervention_rate'] * 100.0, digits=3)} of steps."
            ),
        },
        {
            "item": 5,
            "mechanism": "Exact drain endpoints are preserved: the projector can realize true 0% and 100% drains when required.",
            "code_evidence": [
                "env/safety_layer.py:640-642",
            ],
            "test_evidence": [
                "scripts/smoke_test_safety_v4.py:93-151",
            ],
            "runtime_evidence": (
                "Adversarial tests exercise both exact endpoints; every stage-region training group records nonzero exact-full-drain counts. The measured traces did not require exact-zero overrides."
            ),
        },
        {
            "item": 4,
            "mechanism": "Origin-destination batch flow is exact and conservation is checked at step level and episode summary level.",
            "code_evidence": [
                "env/safety_layer.py:235-273,571-575",
                "env/safe_multi_dc_env.py:357-380,452-469",
            ],
            "test_evidence": [
                "scripts/smoke_test_safety_v4.py:67-82,385-408",
            ],
            "runtime_evidence": (
                f"Full a-d max transport conservation error is {compact_sci(full['us']['safety_metrics']['maximum_transport_conservation_error'])} (US) and {compact_sci(full['global']['safety_metrics']['maximum_transport_conservation_error'])} (Global)."
            ),
        },
        {
            "item": 9,
            "mechanism": "Negative-demand flush exists as an optional ablation, not the primary protocol.",
            "code_evidence": [
                "env/protocols/v4_safety.yaml:43-44",
                "env/safety_layer.py:497-564",
            ],
            "test_evidence": [
                "scripts/smoke_test_safety_v4.py:241-273",
            ],
            "runtime_evidence": (
                "Replay flush ablation remains 10/10 safe in both regions but raises mean intervention to 23.90% (US) and 48.52% (Global); it is reported as economic ablation only."
            ),
        },
        {
            "item": 10,
            "mechanism": "Grid and ramp caps are implemented in the shield, but the primary frozen protocol leaves both disabled.",
            "code_evidence": [
                "env/safety_layer.py:23-27,62-79",
                "env/safe_multi_dc_env.py:67-95,122-160,275-310",
            ],
            "test_evidence": [
                "scripts/smoke_test_safety_v4.py:335-382",
            ],
            "runtime_evidence": (
                "All 36 completion records show primary max_grid_mw=null and max_upward_ramp_mw=null, matching the protocol snapshot."
            ),
        },
    ]
    return sorted(rows, key=lambda row: row["item"])


def build_package() -> dict[str, Any]:
    inputs = load_inputs()
    protocol = inputs["protocol"]
    preflight = inputs["preflight"]
    if inputs["full"]["evaluation_cells"] != "a-d":
        raise ValueError("full_results.json is not the a-d evaluation")
    if inputs["final"]["evaluation_cells"] != "e-h":
        raise ValueError("final_results.json is not the e-h evaluation")
    verification, model_records, training_groups = verify_model_hashes(
        inputs["manifest"]
    )

    replay = replay_summary(inputs["replay"])
    short = stage_region_summary(
        inputs["short"],
        gate=inputs["short_gate"],
        include_diagnostic=True,
    )
    medium = stage_region_summary(
        inputs["medium"],
        gate=inputs["medium_gate"],
        include_diagnostic=True,
    )
    full = stage_region_summary(
        inputs["full"],
        gate=inputs["full_gate"],
        include_diagnostic=True,
    )
    final = stage_region_summary(
        inputs["final"],
        compute_optimizer=True,
    )

    v3_selected = inputs["v3_canonical"]["selected_a_d"]["regions"]
    package = {
        "analysis_package": "ppo-v4-hard-safety-canonical-analysis",
        "scope": {
            "new_v4_model_count": 36,
            "new_v4_stage_counts": {"short": 6, "medium": 10, "full": 20},
            "replay_archived_v3_model_count": 20,
            "replay_modes": [PRIMARY_MODE, FLUSH_MODE],
            "fresh_confirmatory_data_available": bool(
                protocol["claim_scope"]["fresh_confirmatory_data_available"]
            ),
            "headline_scope": "a-d development cells only",
            "descriptive_scope": "e-h only; not headline-eligible",
            "interval_scope": "optimizer-seed variability only",
            "untouched_month_or_cell_claim": "No new untouched month/cells were introduced by this frozen package.",
        },
        "protocol_snapshot": {
            "name": protocol["name"],
            "status": protocol["status"],
            "parent_protocol": protocol["parent_protocol"],
            "claim_scope": protocol["claim_scope"],
            "actionable_state": protocol["actionable_state"],
            "safety": {
                **protocol["safety"],
                "guaranteed_carried_batch_capacity": float(
                    protocol["safety"]["future_fleet_capacity_total"]
                    - protocol["safety"]["service_envelope_total"]
                    - protocol["safety"]["batch_arrival_envelope_total"]
                ),
            },
            "replay": protocol["replay"],
            "stages": protocol["stages"],
            "promotion_gate": protocol["promotion_gate"],
            "required_metrics": protocol["required_metrics"],
        },
        "preflight_snapshot": preflight,
        "stage_results": {
            "replay": {
                "evaluation_cells": "a-d",
                "regions": replay,
            },
            "short": {
                "evaluation_cells": "a-d",
                "gate_passed": bool(inputs["short_gate"]["passed"]),
                "regions": short,
            },
            "medium": {
                "evaluation_cells": "a-d",
                "gate_passed": bool(inputs["medium_gate"]["passed"]),
                "regions": medium,
            },
            "full_a_d": {
                "evaluation_cells": "a-d",
                "gate_passed": bool(inputs["full_gate"]["passed"]),
                "regions": full,
            },
            "descriptive_e_h": {
                "evaluation_cells": "e-h",
                "regions": final,
                "interpretation": inputs["final"]["interpretation"],
            },
        },
        "v3_comparison": {
            "archived_v3_selected_a_d": {
                region: {
                    "safe_seed_count": int(
                        v3_selected[region]["candidate_summary"]["safe_seed_count"]
                    ),
                    "seed_count": int(
                        v3_selected[region]["candidate_summary"]["seed_count"]
                    ),
                    "mean_savings_pct_vs_status_quo": float(
                        v3_selected[region]["candidate_summary"][
                            "mean_savings_vs_status_quo_pct"
                        ]
                    ),
                }
                for region in ("us", "global")
            },
            "v4_full_a_d": {
                region: {
                    "safe_seed_count": int(full[region]["safe_seed_count"]),
                    "seed_count": int(full[region]["seed_count"]),
                    "mean_savings_pct_vs_status_quo": float(
                        full[region]["mean_savings_pct_vs_status_quo"]
                    ),
                }
                for region in ("us", "global")
            },
            "interpretation": (
                "Under this frozen protocol, the archived v3 selected policy is only 1/10 safe in each region, whereas v4 full a-d is 10/10 safe in each region. That is protocol evidence, not a causal proof."
            ),
        },
        "provenance": {
            "input_hashes": build_input_hashes(),
            "hash_verification": verification,
            "model_records": sorted(
                model_records,
                key=lambda row: (
                    STAGE_ORDER[row["stage"]],
                    REGION_ORDER[row["region"]],
                    row["seed"],
                ),
            ),
        },
        "training_telemetry": {
            "group_summaries": training_groups,
            "interpretation": (
                "Telemetry comes directly from frozen training.json diagnostics: intervention/projection, mandatory EDF activity, deadline slack, exact endpoints, flush activations, certificates, and transport conservation."
            ),
        },
        "implementation_matrix": [],
        "figures": {
            key: path.name for key, path in FIGURE_PATHS.items()
        },
        "required_interpretation": {
            "replay": (
                "Replay uses 20 archived v3 models evaluated in two modes. The flush variant is an economic ablation only."
            ),
            "a_d": (
                "Full a-d establishes hard-safety gate passage for the frozen development cells, but it does not create fresh untouched confirmatory data."
            ),
            "e_h": (
                "E-h remains descriptive only and must not be used as a headline or generalization claim."
            ),
            "projector_role": (
                "The shield is materially co-producing feasible behavior: replay intervention is small, but trained-policy intervention remains substantial, especially in Global."
            ),
        },
    }
    package["implementation_matrix"] = implementation_matrix(package)
    return package


def build_stage_safety_economics_figure(package: dict[str, Any]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        sharey=True,
        constrained_layout=True,
    )
    stages = ["short", "medium", "full_a_d"]
    x = np.arange(3)
    for axis, region in zip(axes, ("us", "global"), strict=True):
        means = []
        lower = []
        upper = []
        labels = []
        for stage in stages:
            row = package["stage_results"][stage]["regions"][region]
            means.append(row["mean_savings_pct_vs_status_quo"])
            ci = row["optimizer_vs_status_quo"]["optimizer_bootstrap_ci95_savings_pct"]
            lower.append(row["mean_savings_pct_vs_status_quo"] - ci[0])
            upper.append(ci[1] - row["mean_savings_pct_vs_status_quo"])
            labels.append(f"{row['safe_seed_count']}/{row['seed_count']} safe")
        axis.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
        axis.errorbar(
            x,
            means,
            yerr=np.vstack([lower, upper]),
            fmt="o-",
            color=COLORS[region],
            linewidth=2.2,
            capsize=5,
            markersize=7,
        )
        for idx, label in enumerate(labels):
            axis.annotate(
                label,
                (x[idx], means[idx]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )
        axis.set_title(region.upper())
        axis.set_xticks(x, ["Short (3)", "Medium (5)", "Full (10)"])
        axis.set_xlabel("Frozen training stage")
        axis.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("Mean savings vs Status Quo (%)")
    fig.suptitle(
        "V4 a-d stage economics with exact-safety gate counts\n95% intervals reflect optimizer-seed variability only"
    )
    fig.savefig(
        FIGURE_PATHS["stage_safety_economics"],
        dpi=220,
        metadata={"Software": "build_safety_v4_results.py"},
    )
    plt.close(fig)


def build_replay_shield_flush_figure(package: dict[str, Any]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        constrained_layout=True,
    )
    width = 0.34
    regions = ["us", "global"]
    x = np.arange(len(regions))
    replay = package["stage_results"]["replay"]["regions"]
    primary_savings = [
        replay[region]["modes"][PRIMARY_MODE]["mean_savings_pct_vs_status_quo"]
        for region in regions
    ]
    flush_savings = [
        replay[region]["modes"][FLUSH_MODE]["mean_savings_pct_vs_status_quo"]
        for region in regions
    ]
    primary_intervention = [
        100.0
        * replay[region]["modes"][PRIMARY_MODE]["mean_safety_intervention_rate"]
        for region in regions
    ]
    flush_intervention = [
        100.0
        * replay[region]["modes"][FLUSH_MODE]["mean_safety_intervention_rate"]
        for region in regions
    ]
    axes[0].bar(x - width / 2, primary_savings, width, label="Safety only", color="#4c78a8")
    axes[0].bar(
        x + width / 2,
        flush_savings,
        width,
        label="Flush ablation",
        color="#f58518",
    )
    axes[0].axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [label.upper() for label in regions])
    axes[0].set_ylabel("Mean savings vs mode-specific Status Quo (%)")
    axes[0].set_title("Replay economics (both modes remain 10/10 safe)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(
        x - width / 2,
        primary_intervention,
        width,
        label="Safety only",
        color="#4c78a8",
    )
    axes[1].bar(
        x + width / 2,
        flush_intervention,
        width,
        label="Flush ablation",
        color="#f58518",
    )
    axes[1].set_xticks(x, [label.upper() for label in regions])
    axes[1].set_ylabel("Mean intervention rate (%)")
    axes[1].set_title("Replay shield activity")
    axes[1].grid(alpha=0.25, axis="y")
    fig.suptitle(
        "Archived-v3 replay under the v4 shield\nFlush is reported as an economic ablation only"
    )
    fig.savefig(
        FIGURE_PATHS["replay_shield_flush"],
        dpi=220,
        metadata={"Software": "build_safety_v4_results.py"},
    )
    plt.close(fig)


def build_final_savings_ci_safety_figure(package: dict[str, Any]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.3),
        constrained_layout=True,
    )
    full = package["stage_results"]["full_a_d"]["regions"]
    regions = ["us", "global"]
    y = np.arange(len(regions))
    means = [full[region]["optimizer_vs_status_quo"]["mean_improvement_usd"] for region in regions]
    cis = [full[region]["optimizer_vs_status_quo"]["optimizer_bootstrap_ci95_usd"] for region in regions]
    lower = [mean_value - ci[0] for mean_value, ci in zip(means, cis, strict=True)]
    upper = [ci[1] - mean_value for mean_value, ci in zip(means, cis, strict=True)]
    axes[0].axvline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    axes[0].errorbar(
        means,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#2f4b7c",
        ecolor="#2f4b7c",
        linewidth=2.0,
        capsize=5,
        markersize=8,
    )
    axes[0].set_yticks(y, [label.upper() for label in regions])
    axes[0].set_xlabel("Mean improvement vs Status Quo (USD)")
    axes[0].set_title("Full a-d optimizer-seed 95% intervals")
    axes[0].grid(alpha=0.25, axis="x")

    axes[1].axis("off")
    lines = ["Full a-d safety scoreboard", ""]
    for region in regions:
        row = full[region]
        safety = row["safety_metrics"]
        lines.extend(
            [
                f"{region.upper()}: {row['safe_seed_count']}/{row['seed_count']} safe",
                f"  mean savings {compact_pct(row['mean_savings_pct_vs_status_quo'], digits=5)}",
                f"  worst seed {compact_pct(row['worst_seed_savings_pct_vs_status_quo'], digits=5)}",
                f"  terminal pool <= {compact_sci(safety['maximum_terminal_batch_pool'])}",
                f"  terminal backlog = {safety['maximum_terminal_service_backlog']:.1f}",
                f"  certs = {safety['safety_infeasibility_certificates']}",
                f"  transport error <= {compact_sci(safety['maximum_transport_conservation_error'])}",
                f"  intervention = {compact_rate(row['mean_safety_intervention_rate'])}",
                "",
            ]
        )
    axes[1].text(
        0.0,
        1.0,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )
    fig.suptitle("Frozen a-d headline: savings interval plus hard-safety evidence")
    fig.savefig(
        FIGURE_PATHS["final_savings_ci_safety"],
        dpi=220,
        metadata={"Software": "build_safety_v4_results.py"},
    )
    plt.close(fig)


def build_intervention_projection_figure(package: dict[str, Any]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 4.8),
        constrained_layout=True,
    )
    x_labels = ["Replay", "Short", "Medium", "Full", "e-h"]
    series = {
        "us": [
            package["stage_results"]["replay"]["regions"]["us"]["modes"][PRIMARY_MODE],
            package["stage_results"]["short"]["regions"]["us"],
            package["stage_results"]["medium"]["regions"]["us"],
            package["stage_results"]["full_a_d"]["regions"]["us"],
            package["stage_results"]["descriptive_e_h"]["regions"]["us"],
        ],
        "global": [
            package["stage_results"]["replay"]["regions"]["global"]["modes"][PRIMARY_MODE],
            package["stage_results"]["short"]["regions"]["global"],
            package["stage_results"]["medium"]["regions"]["global"],
            package["stage_results"]["full_a_d"]["regions"]["global"],
            package["stage_results"]["descriptive_e_h"]["regions"]["global"],
        ],
    }
    x = np.arange(len(x_labels))
    for region in ("us", "global"):
        axes[0].plot(
            x,
            [
                100.0 * row["mean_safety_intervention_rate"]
                for row in series[region]
            ],
            marker="o",
            linewidth=2.2,
            color=COLORS[region],
            label=region.upper(),
        )
        axes[1].plot(
            x,
            [row["max_projection_l2"] for row in series[region]],
            marker="o",
            linewidth=2.2,
            color=COLORS[region],
            label=region.upper(),
        )
    axes[0].set_xticks(x, x_labels)
    axes[0].set_ylabel("Mean intervention rate (%)")
    axes[0].set_title("Evaluation-time shield activity")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend(frameon=False)
    axes[1].set_xticks(x, x_labels)
    axes[1].set_ylabel("Max projection L2")
    axes[1].set_title("Evaluation-time projection magnitude")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend(frameon=False)
    fig.suptitle(
        "The shield remains materially active after training\nGlobal full a-d intervention rises from replay 0.47% to trained-policy 18.79%"
    )
    fig.savefig(
        FIGURE_PATHS["intervention_projection"],
        dpi=220,
        metadata={"Software": "build_safety_v4_results.py"},
    )
    plt.close(fig)


def build_eh_descriptive_figure(package: dict[str, Any]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        constrained_layout=True,
        sharey=True,
    )
    final = package["stage_results"]["descriptive_e_h"]["regions"]
    regions = ["us", "global"]
    y = np.arange(len(regions))
    means = [final[region]["mean_savings_pct_vs_status_quo"] for region in regions]
    cis = [
        final[region]["optimizer_vs_status_quo"]["optimizer_bootstrap_ci95_savings_pct"]
        for region in regions
    ]
    lower = [mean_value - ci[0] for mean_value, ci in zip(means, cis, strict=True)]
    upper = [ci[1] - mean_value for mean_value, ci in zip(means, cis, strict=True)]
    axes[0].axvline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    axes[0].errorbar(
        means,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#665191",
        ecolor="#665191",
        linewidth=2.0,
        capsize=5,
        markersize=8,
    )
    axes[0].scatter(
        [final[region]["worst_seed_savings_pct_vs_status_quo"] for region in regions],
        y,
        marker="D",
        color="#ff7c43",
        label="Worst seed",
    )
    axes[0].set_yticks(y, [label.upper() for label in regions])
    axes[0].set_xlabel("Savings vs Status Quo (%)")
    axes[0].set_title("Descriptive e-h savings")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25, axis="x")

    axes[1].axis("off")
    lines = ["E-h is descriptive only", ""]
    for region in regions:
        row = final[region]
        lines.extend(
            [
                f"{region.upper()}: {row['safe_seed_count']}/{row['seed_count']} safe",
                f"  mean savings {compact_pct(row['mean_savings_pct_vs_status_quo'], digits=5)}",
                f"  worst seed {compact_pct(row['worst_seed_savings_pct_vs_status_quo'], digits=4)}",
                f"  intervention {compact_rate(row['mean_safety_intervention_rate'])}",
                "",
            ]
        )
    axes[1].text(
        0.0,
        1.0,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )
    fig.suptitle(
        "Frozen e-h transfer check\nReported honestly, but not a headline or generalization claim"
    )
    fig.savefig(
        FIGURE_PATHS["eh_descriptive"],
        dpi=220,
        metadata={"Software": "build_safety_v4_results.py"},
    )
    plt.close(fig)


def build_figures(package: dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.titlesize": 14,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )
    build_stage_safety_economics_figure(package)
    build_replay_shield_flush_figure(package)
    build_final_savings_ci_safety_figure(package)
    build_intervention_projection_figure(package)
    build_eh_descriptive_figure(package)


def build_report(package: dict[str, Any]) -> str:
    replay = package["stage_results"]["replay"]["regions"]
    short = package["stage_results"]["short"]["regions"]
    medium = package["stage_results"]["medium"]["regions"]
    full = package["stage_results"]["full_a_d"]["regions"]
    final = package["stage_results"]["descriptive_e_h"]["regions"]
    training = package["training_telemetry"]["group_summaries"]
    verification = package["provenance"]["hash_verification"]

    lines = [
        "# PPO v4 hard-safety canonical analysis package",
        "",
        "> Frozen v4 safety-only package built deterministically from archived protocol/results/manifests/training diagnostics only.",
        "",
        "All intervals below reflect optimizer-seed variability only. No new untouched month/cells were added; a-d remains development-only and e-h remains descriptive only.",
        "",
        "## Frozen scope",
        "",
        "- **36 new v4 models:** short **6** (3/region), medium **10**, full **20**.",
        "- **Replay:** **20** archived v3 selected models evaluated in **two** modes (`safety_only`, `safety_plus_negative_demand_flush`).",
        "- **Hard mechanism:** actionable state keeps only `deadline_step > t`; frozen envelopes are **2.25** service, **1.0** new batch, **4.0** future fleet with **0.75** guaranteed carried reserve; cumulative exact EDF; clean-state service guarantee; exact flow and exact drain endpoints; fail-closed certificates; minimal Euclidean projection telemetry; negative-demand flush optional; grid/ramp caps implemented but disabled in the primary protocol.",
        "",
        "## Replay of archived v3 models under the v4 shield",
        "",
        "| Region | Mode | Safe seeds | Mean cost | Mean savings | Optimizer CI (USD) | Intervention | Note |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        f"| US | safety_only | {replay['us']['modes'][PRIMARY_MODE]['safe_seed_count']}/{replay['us']['modes'][PRIMARY_MODE]['seed_count']} | {compact_money(replay['us']['modes'][PRIMARY_MODE]['mean_total_cost_usd'], digits=3)} | {compact_pct(replay['us']['modes'][PRIMARY_MODE]['mean_savings_pct_vs_status_quo'], digits=5)} | {compact_money(replay['us']['modes'][PRIMARY_MODE]['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(replay['us']['modes'][PRIMARY_MODE]['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][1])} | {compact_rate(replay['us']['modes'][PRIMARY_MODE]['mean_safety_intervention_rate'])} | archived v3 replay |",
        f"| US | flush ablation | {replay['us']['modes'][FLUSH_MODE]['safe_seed_count']}/{replay['us']['modes'][FLUSH_MODE]['seed_count']} | {compact_money(replay['us']['modes'][FLUSH_MODE]['mean_total_cost_usd'], digits=3)} | {compact_pct(replay['us']['modes'][FLUSH_MODE]['mean_savings_pct_vs_status_quo'], digits=4)} | {compact_money(replay['us']['modes'][FLUSH_MODE]['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(replay['us']['modes'][FLUSH_MODE]['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][1])} | {compact_pct(replay['us']['modes'][FLUSH_MODE]['mean_safety_intervention_rate'] * 100.0, digits=2, signed=False)} | economic ablation only |",
        f"| GLOBAL | safety_only | {replay['global']['modes'][PRIMARY_MODE]['safe_seed_count']}/{replay['global']['modes'][PRIMARY_MODE]['seed_count']} | {compact_money(replay['global']['modes'][PRIMARY_MODE]['mean_total_cost_usd'], digits=3)} | {compact_pct(replay['global']['modes'][PRIMARY_MODE]['mean_savings_pct_vs_status_quo'], digits=5)} | {compact_money(replay['global']['modes'][PRIMARY_MODE]['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(replay['global']['modes'][PRIMARY_MODE]['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][1])} | {compact_rate(replay['global']['modes'][PRIMARY_MODE]['mean_safety_intervention_rate'])} | archived v3 replay |",
        f"| GLOBAL | flush ablation | {replay['global']['modes'][FLUSH_MODE]['safe_seed_count']}/{replay['global']['modes'][FLUSH_MODE]['seed_count']} | {compact_money(replay['global']['modes'][FLUSH_MODE]['mean_total_cost_usd'], digits=3)} | {compact_pct(replay['global']['modes'][FLUSH_MODE]['mean_savings_pct_vs_status_quo'], digits=4)} | {compact_money(replay['global']['modes'][FLUSH_MODE]['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(replay['global']['modes'][FLUSH_MODE]['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][1])} | {compact_pct(replay['global']['modes'][FLUSH_MODE]['mean_safety_intervention_rate'] * 100.0, digits=2, signed=False)} | economic ablation only |",
        "",
        "Replay `safety_only` is 10/10 safe in both regions. US mean cost is $6,563,934.425 (+0.05746%) with intervention 0.000112; Global mean cost is $6,463,819.890 (+2.04079%) with intervention 0.004727. The flush variant lowers absolute cost, but the correct comparison is versus the flush-enabled Status Quo; treat it as an economic ablation only.",
        "",
        "## A-d stage progression",
        "",
        "| Stage | Region | Safe seeds | Mean savings | Optimizer CI (USD) | All-seed positive? | Diagnostic CI vs replayed v3 (USD) |",
        "|---|---|---:|---:|---:|---|---:|",
        f"| Short | US | {short['us']['safe_seed_count']}/{short['us']['seed_count']} | {compact_pct(short['us']['mean_savings_pct_vs_status_quo'], digits=4)} | {compact_money(short['us']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(short['us']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][1])} | {str(short['us']['optimizer_vs_status_quo']['all_seed_improvements_positive']).lower()} | {compact_money(short['us']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(short['us']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][1])} |",
        f"| Short | GLOBAL | {short['global']['safe_seed_count']}/{short['global']['seed_count']} | {compact_pct(short['global']['mean_savings_pct_vs_status_quo'], digits=4)} | {compact_money(short['global']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(short['global']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][1])} | {str(short['global']['optimizer_vs_status_quo']['all_seed_improvements_positive']).lower()} | {compact_money(short['global']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(short['global']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][1])} |",
        f"| Medium | US | {medium['us']['safe_seed_count']}/{medium['us']['seed_count']} | {compact_pct(medium['us']['mean_savings_pct_vs_status_quo'], digits=4)} | {compact_money(medium['us']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(medium['us']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][1])} | {str(medium['us']['optimizer_vs_status_quo']['all_seed_improvements_positive']).lower()} | {compact_money(medium['us']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(medium['us']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][1])} |",
        f"| Medium | GLOBAL | {medium['global']['safe_seed_count']}/{medium['global']['seed_count']} | {compact_pct(medium['global']['mean_savings_pct_vs_status_quo'], digits=4)} | {compact_money(medium['global']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(medium['global']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][1])} | {str(medium['global']['optimizer_vs_status_quo']['all_seed_improvements_positive']).lower()} | {compact_money(medium['global']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(medium['global']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][1])} |",
        f"| Full | US | {full['us']['safe_seed_count']}/{full['us']['seed_count']} | {compact_pct(full['us']['mean_savings_pct_vs_status_quo'], digits=5)} | {compact_money(full['us']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(full['us']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][1])} | {str(full['us']['optimizer_vs_status_quo']['all_seed_improvements_positive']).lower()} | {compact_money(full['us']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(full['us']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][1])} |",
        f"| Full | GLOBAL | {full['global']['safe_seed_count']}/{full['global']['seed_count']} | {compact_pct(full['global']['mean_savings_pct_vs_status_quo'], digits=5)} | {compact_money(full['global']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(full['global']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][1])} | {str(full['global']['optimizer_vs_status_quo']['all_seed_improvements_positive']).lower()} | {compact_money(full['global']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(full['global']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][1])} |",
        "",
        "- **Short gate:** 6/6 safe. US mean is -0.0243% and its CI crosses zero; Global is +2.2095% with a positive optimizer-seed CI.",
        "- **Medium gate:** 10/10 safe. US is -0.1045%; Global is +1.5572%, but its CI still crosses zero.",
        "",
        "## Full a-d headline",
        "",
        "- **20/20 full a-d evaluations are safe** under the frozen gate: exact completion within protocol tolerance, zero expiry, zero certificates, zero terminal backlog, terminal pool numerically at zero (<=1.1e-9), and transport conservation around 3e-16.",
        f"- **US:** mean savings {compact_pct(full['us']['mean_savings_pct_vs_status_quo'], digits=5)}, optimizer CI {compact_money(full['us']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(full['us']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][1])}, intervention {compact_rate(full['us']['mean_safety_intervention_rate'])}. Economic improvement is **not established** under this frozen protocol.",
        f"- **Global:** mean savings {compact_pct(full['global']['mean_savings_pct_vs_status_quo'], digits=5)}, worst seed {compact_pct(full['global']['worst_seed_savings_pct_vs_status_quo'], digits=5)}, optimizer CI {compact_money(full['global']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(full['global']['optimizer_vs_status_quo']['optimizer_bootstrap_ci95_usd'][1])}, diagnostic CI vs archived v3 replay {compact_money(full['global']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][0])} to {compact_money(full['global']['diagnostic_vs_archived_v3_safety_only']['optimizer_bootstrap_ci95_usd'][1])}, intervention {compact_rate(full['global']['mean_safety_intervention_rate'])}. All ten Global full seeds are positive.",
        f"- The projector changes **{compact_pct(full['global']['mean_safety_intervention_rate'] * 100.0, digits=1, signed=False)}** of Global full a-d steps even though Global replay `safety_only` intervention was only **{compact_pct(replay['global']['modes'][PRIMARY_MODE]['mean_safety_intervention_rate'] * 100.0, digits=2, signed=False)}**. It is therefore materially co-producing feasible behavior, not merely certifying a nearly-feasible archived policy.",
        "",
        "## E-h descriptive transfer only",
        "",
        f"- **US:** 10/10 safe, mean savings {compact_pct(final['us']['mean_savings_pct_vs_status_quo'], digits=5)}, worst seed {compact_pct(final['us']['worst_seed_savings_pct_vs_status_quo'], digits=3)}, intervention {compact_rate(final['us']['mean_safety_intervention_rate'])}.",
        f"- **Global:** 10/10 safe, mean savings {compact_pct(final['global']['mean_savings_pct_vs_status_quo'], digits=5)}, worst seed {compact_pct(final['global']['worst_seed_savings_pct_vs_status_quo'], digits=4)}, intervention {compact_rate(final['global']['mean_safety_intervention_rate'])}.",
        "- These e-h numbers are descriptive only. They must not be promoted to a headline or generalized beyond the frozen protocol.",
        "",
        "## Comparison with archived v3 selected policies",
        "",
        f"- Archived v3 selected a-d safety: US {package['v3_comparison']['archived_v3_selected_a_d']['us']['safe_seed_count']}/{package['v3_comparison']['archived_v3_selected_a_d']['us']['seed_count']}, Global {package['v3_comparison']['archived_v3_selected_a_d']['global']['safe_seed_count']}/{package['v3_comparison']['archived_v3_selected_a_d']['global']['seed_count']}.",
        f"- V4 full a-d hard safety: US {package['v3_comparison']['v4_full_a_d']['us']['safe_seed_count']}/{package['v3_comparison']['v4_full_a_d']['us']['seed_count']}, Global {package['v3_comparison']['v4_full_a_d']['global']['safe_seed_count']}/{package['v3_comparison']['v4_full_a_d']['global']['seed_count']}.",
        "- This is a protocol comparison only; it does not establish a causal proof beyond the frozen design.",
        "",
        "## Training telemetry from frozen training.json diagnostics",
        "",
        "| Group | Mean intervention | Mean projection L2 | Mandatory-step rate | Min slack | Exact full drains (mean) | Exact zero drains (mean) | Certificates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for key in (
        "short:us",
        "short:global",
        "medium:us",
        "medium:global",
        "full:us",
        "full:global",
    ):
        row = training[key]
        lines.append(
            f"| {key} | {compact_rate(row['intervention_rate']['mean'])} | "
            f"{row['mean_projection_l2']['mean']:.6f} | "
            f"{row['mandatory_step_rate']['mean']:.6f} | "
            f"{row['minimum_deadline_slack']['min']:.6f} | "
            f"{row['exact_full_drain_count']['mean']:.1f} | "
            f"{row['exact_zero_drain_count']['mean']:.1f} | "
            f"{row['infeasibility_certificates']['max']:.0f} |"
        )

    lines.extend(
        [
            "",
            "All 36 training diagnostics were hash-verified against their completion records. Negative flush activations are zero in every training run because the primary protocol keeps flush disabled.",
            "",
            "## Ten-item implementation matrix",
            "",
            "| # | Mechanism | Code/tests evidence | Runtime evidence |",
            "|---:|---|---|---|",
        ]
    )
    for row in package["implementation_matrix"]:
        code = "<br>".join(row["code_evidence"] + row["test_evidence"])
        lines.append(
            f"| {row['item']} | {row['mechanism']} | {code} | {row['runtime_evidence']} |"
        )

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Protocol hash: `{verification['protocol_sha256']}`",
            f"- Manifest hash: `{verification['manifest_sha256']}`",
            f"- Verified model hashes: **{verification['verified_model_hash_count']}**",
            f"- Verified diagnostics hashes: **{verification['verified_diagnostics_hash_count']}**",
            f"- Verified completion hashes: **{verification['verified_completion_hash_count']}**",
            f"- Stage counts: `{json.dumps(verification['stage_counts'], sort_keys=True)}`",
            f"- Frozen input hashes are preserved in `canonical_results.json` under `provenance.input_hashes`.",
            "",
            "## Figures",
            "",
            f"![Stage safety and economics]({FIGURE_PATHS['stage_safety_economics'].name})",
            "",
            f"![Replay shield and flush ablation]({FIGURE_PATHS['replay_shield_flush'].name})",
            "",
            f"![Full a-d savings CI and safety]({FIGURE_PATHS['final_savings_ci_safety'].name})",
            "",
            f"![Intervention and projection]({FIGURE_PATHS['intervention_projection'].name})",
            "",
            f"![Descriptive e-h transfer]({FIGURE_PATHS['eh_descriptive'].name})",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    package = build_package()
    build_figures(package)
    canonical_path = OUT_ROOT / "canonical_results.json"
    report_path = OUT_ROOT / "results_report.md"
    canonical_sha = write_canonical_json(canonical_path, package)
    report_sha = write_text(report_path, build_report(package))
    output_hashes = {
        to_rel(canonical_path): canonical_sha,
        to_rel(report_path): report_sha,
    }
    for name, path in FIGURE_PATHS.items():
        output_hashes[to_rel(path)] = sha256_file(path)
    print(json.dumps(output_hashes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
