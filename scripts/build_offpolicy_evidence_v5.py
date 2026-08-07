"""Build and validate canonical evidence for frozen v5 TD3+BC campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_offpolicy_campaign_v5 import (  # noqa: E402
    CAMPAIGNS,
    EXPECTED_SB3_VERSION,
    OUT_ROOT,
    PROTOCOL_ID,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    TD3BehaviorCloning,
    campaign_job,
    module_sha256,
    model_dir,
)

GENERATED_PREFIXES = (
    "models/offpolicy_v5_continuous/",
    "output/offpolicy_v5_continuous/",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def saved_model_hashes(path: Path) -> dict[str, str]:
    model = TD3BehaviorCloning.load(path, device="cpu")
    return {
        "actor": module_sha256(model.actor),
        "actor_target": module_sha256(model.actor_target),
        "critic": module_sha256(model.critic),
    }


def git_status_short() -> list[str]:
    return subprocess.check_output(
        ["git", "--no-pager", "status", "--short", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    ).splitlines()


def classify_workspace_state(
    status: list[str],
    *,
    unstaged_tracked_clean: bool,
    staged_tracked_clean: bool,
) -> dict[str, Any]:
    tracked_entries = [line for line in status if not line.startswith("?? ")]
    untracked_entries = [line for line in status if line.startswith("?? ")]
    untracked_generated_only = all(
        any(
            line[3:].replace("\\", "/").startswith(prefix)
            for prefix in GENERATED_PREFIXES
        )
        for line in untracked_entries
    )
    tracked_source_clean = (
        unstaged_tracked_clean
        and staged_tracked_clean
        and not tracked_entries
    )
    return {
        "git_status_short": status,
        "tracked_source_clean": tracked_source_clean,
        "untracked_generated_only": untracked_generated_only,
        "tracked_entries": tracked_entries,
        "untracked_entries": untracked_entries,
        "generated_artifact_roots": [
            "models\\offpolicy_v5_continuous\\",
            "output\\offpolicy_v5_continuous\\",
        ],
    }


def workspace_state() -> dict[str, Any]:
    unstaged_clean = (
        subprocess.run(
            ["git", "diff", "--quiet", "--"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )
    staged_clean = (
        subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )
    state = classify_workspace_state(
        git_status_short(),
        unstaged_tracked_clean=unstaged_clean,
        staged_tracked_clean=staged_clean,
    )
    if not state["tracked_source_clean"]:
        raise RuntimeError(
            "canonical evidence requires clean staged and unstaged tracked source"
        )
    if not state["untracked_generated_only"]:
        raise RuntimeError(
            "canonical evidence allows untracked files only under the v5 model/output roots"
        )
    return state


def validate_campaign_roles(
    bc_campaign: str,
    postrl_campaign: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if bc_campaign == postrl_campaign:
        raise ValueError("BC and post-RL campaigns must be different")
    bc_config = CAMPAIGNS[bc_campaign]
    postrl_config = CAMPAIGNS[postrl_campaign]
    errors: list[str] = []
    if bc_config.get("role") != "bc_only":
        errors.append(f"{bc_campaign} is not role=bc_only")
    if postrl_config.get("role") != "post_rl":
        errors.append(f"{postrl_campaign} is not role=post_rl")
    if postrl_config.get("warm_start_campaign") != bc_campaign:
        errors.append("post-RL warm_start_campaign does not match BC campaign")
    if bc_config.get("expected_training_mode") != "teacher_bc_only":
        errors.append("BC expected_training_mode is invalid")
    if postrl_config.get("expected_training_mode") != "td3_bc_postrl":
        errors.append("post-RL expected_training_mode is invalid")
    if int(bc_config.get("timesteps", -1)) != 0:
        errors.append("BC campaign must have zero reward-training timesteps")
    if int(postrl_config.get("timesteps", 0)) <= 0:
        errors.append("post-RL campaign must have positive reward-training timesteps")
    if errors:
        raise ValueError("; ".join(errors))
    return bc_config, postrl_config


def _path_text(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("/", "\\")


def _record_contract_errors(
    record: dict[str, Any],
    *,
    campaign_name: str,
    region: str,
    seed: int,
) -> list[str]:
    config = CAMPAIGNS[campaign_name]
    role = str(config["role"])
    job = record.get("job", {})
    evaluation = record.get("evaluation", {})
    transfer = record.get("descriptive_transfer_eh", {})
    weight = record.get("weight_update_evidence", {})
    teacher = record.get("teacher", {})
    teacher_audit = teacher.get("action_audit", {})
    errors: list[str] = []

    expected_job = campaign_job(campaign_name, region=region, seed=seed)
    expected_profile = str(config["profile"])
    expected_fields = {
        "campaign": campaign_name,
        "region": region,
        "seed": seed,
        "timesteps": int(config["timesteps"]),
        "algorithm": str(config["algorithm"]),
        "stage": str(config["stage"]),
    }
    for key, expected in expected_fields.items():
        actual = job.get(key) if key in job else record.get(key)
        if actual != expected:
            errors.append(f"{key}={actual!r}, expected {expected!r}")
    if record.get("campaign") != campaign_name:
        errors.append("record campaign mismatch")
    if record.get("profile") != expected_profile:
        errors.append("record profile mismatch")
    if record.get("campaign_role") != role:
        errors.append("record campaign_role mismatch")
    if record.get("training_mode") != config["expected_training_mode"]:
        errors.append("record training_mode mismatch")
    if record.get("protocol_id") != PROTOCOL_ID:
        errors.append("record protocol_id mismatch")
    if record.get("protocol_sha256") != PROTOCOL_SHA256:
        errors.append("record protocol_sha256 mismatch")
    if record.get("protocol_path") != _path_text(PROTOCOL_PATH):
        errors.append("record protocol_path mismatch")
    if record.get("stable_baselines3_version") != EXPECTED_SB3_VERSION:
        errors.append("record Stable-Baselines3 version mismatch")
    if record.get("evaluation_label") != config["evaluation_label"]:
        errors.append("record evaluation label mismatch")
    if record.get("descriptive_transfer_label") != config["descriptive_transfer_label"]:
        errors.append("record descriptive-transfer label mismatch")
    if evaluation.get("evaluation_scope") != config["evaluation_label"]:
        errors.append("evaluation scope mismatch")
    if transfer.get("evaluation_scope") != config["descriptive_transfer_label"]:
        errors.append("descriptive-transfer scope mismatch")
    if evaluation.get("teacher_call_guard_enabled") is not True:
        errors.append("development evaluation teacher guard is not enabled")
    if transfer.get("teacher_call_guard_enabled") is not True:
        errors.append("transfer evaluation teacher guard is not enabled")
    if record.get("teacher_present_during_rl") is not False:
        errors.append("teacher_present_during_rl must be false")
    if record.get("teacher_present_at_inference") is not False:
        errors.append("teacher_present_at_inference must be false")
    if teacher_audit.get("stored_action_matches_executed") is not True:
        errors.append("teacher action stored/executed pairing is not exact")
    if int(teacher_audit.get("preclip_action_space_violations", -1)) != 0:
        errors.append("teacher emitted out-of-space actions")
    if int(teacher_audit.get("clipped_action_count", -1)) != 0:
        errors.append("teacher action clipping occurred")
    actor_hash_changed = record.get("actor_hash_before") != record.get(
        "actor_hash_after"
    )
    critic_hash_changed = record.get("critic_hash_before") != record.get(
        "critic_hash_after"
    )
    if weight.get("actor_hash_changed") is not actor_hash_changed:
        errors.append("actor hash-change flag does not match recorded hashes")
    if weight.get("critic_hash_changed") is not critic_hash_changed:
        errors.append("critic hash-change flag does not match recorded hashes")

    if role == "bc_only":
        bc = teacher.get("behavior_cloning", {})
        if int(record.get("n_updates", -1)) != 0:
            errors.append("BC-only record has reward-training updates")
        if not weight.get("actor_hash_changed"):
            errors.append("BC actor did not change")
        if weight.get("critic_hash_changed"):
            errors.append("BC critic changed without reward training")
        if float(weight.get("actor_parameter_delta_l2", 0.0)) <= 0.0:
            errors.append("BC actor parameter delta is not positive")
        if int(bc.get("steps", -1)) != int(config["teacher_bc_steps"]):
            errors.append("BC step count mismatch")
        if bc.get("synchronized") is not True:
            errors.append("BC actor_target is not synchronized")
        if float(bc.get("distance_l2_after", math.inf)) > 1e-12:
            errors.append("BC actor_target distance is nonzero")
        if bc.get("actor_hash") != bc.get("actor_target_hash"):
            errors.append("BC actor and target hashes differ")
        if "source_bc_model" in record:
            errors.append("BC-only record unexpectedly has source_bc_model")
    elif role == "post_rl":
        expected_updates = int(config["expected_n_updates"])
        if int(record.get("n_updates", 0)) != expected_updates:
            errors.append("post-RL update count mismatch")
        if not weight.get("actor_hash_changed") or not weight.get("critic_hash_changed"):
            errors.append("post-RL actor/critic hashes did not both change")
        if float(weight.get("actor_parameter_delta_l2", 0.0)) <= 0.0:
            errors.append("post-RL actor parameter delta is not positive")
        if float(weight.get("critic_parameter_delta_l2", 0.0)) <= 0.0:
            errors.append("post-RL critic parameter delta is not positive")
        sync = record.get("warm_start_actor_target_sync", {})
        if sync.get("synchronized") is not True:
            errors.append("post-RL warm-start actor_target is not synchronized")
        if float(sync.get("distance_l2_after", math.inf)) > 1e-12:
            errors.append("post-RL warm-start target distance is nonzero")
        warm_campaign = str(config["warm_start_campaign"])
        warm_job = campaign_job(warm_campaign, region=region, seed=seed)
        warm_profile = str(CAMPAIGNS[warm_campaign]["profile"])
        expected_source = model_dir(warm_job, warm_profile) / "model.zip"
        if record.get("source_bc_model") != _path_text(expected_source):
            errors.append("post-RL source_bc_model mismatch")
    else:
        errors.append(f"unsupported campaign role {role!r}")

    expected_model = model_dir(expected_job, expected_profile) / "model.zip"
    if record.get("model_path") != _path_text(expected_model):
        errors.append("record model_path mismatch")
    return errors


def post_rl_seed_provenance_pass(row: dict[str, Any]) -> bool:
    weight = row.get("weight_update_evidence", {})
    sync = row.get("warm_start_actor_target_sync", {})
    audit = row.get("teacher_action_audit", {})
    return bool(
        row.get("campaign_role") == "post_rl"
        and row.get("training_mode") == "td3_bc_postrl"
        and int(row.get("n_updates", 0)) > 0
        and weight.get("actor_hash_changed") is True
        and weight.get("critic_hash_changed") is True
        and row.get("actor_hash_before") != row.get("actor_hash_after")
        and row.get("critic_hash_before") != row.get("critic_hash_after")
        and row.get("saved_actor_hash") == row.get("actor_hash_after")
        and row.get("saved_critic_hash") == row.get("critic_hash_after")
        and row.get("source_bc_model_sha256")
        == row.get("verified_source_bc_model_sha256")
        and row.get("actor_hash_before") == row.get("source_bc_actor_hash")
        and row.get("critic_hash_before") == row.get("source_bc_critic_hash")
        and float(weight.get("actor_parameter_delta_l2", 0.0)) > 0.0
        and float(weight.get("critic_parameter_delta_l2", 0.0)) > 0.0
        and sync.get("synchronized") is True
        and float(sync.get("distance_l2_after", math.inf)) <= 1e-12
        and row.get("teacher_present_during_rl") is False
        and row.get("teacher_present_at_inference") is False
        and audit.get("stored_action_matches_executed") is True
        and row.get("stable_baselines3_version") == EXPECTED_SB3_VERSION
        and row.get("protocol_id") == PROTOCOL_ID
        and row.get("protocol_sha256") == PROTOCOL_SHA256
    )


def seed_record(campaign_name: str, region: str, seed: int) -> dict[str, Any]:
    config = CAMPAIGNS[campaign_name]
    job = campaign_job(campaign_name, region=region, seed=seed)
    profile = str(config["profile"])
    seed_dir = model_dir(job, profile)
    record_path = seed_dir / "record.json"
    model_path = seed_dir / "model.zip"
    if not record_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"missing model evidence under {seed_dir}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    saved_hashes = saved_model_hashes(model_path)
    errors = _record_contract_errors(
        record,
        campaign_name=campaign_name,
        region=region,
        seed=seed,
    )
    if record.get("actor_hash_after") != saved_hashes["actor"]:
        errors.append("saved actor hash does not match record")
    if record.get("critic_hash_after") != saved_hashes["critic"]:
        errors.append("saved critic hash does not match record")
    if config["role"] == "bc_only":
        bc = record.get("teacher", {}).get("behavior_cloning", {})
        if bc.get("actor_target_hash") != saved_hashes["actor_target"]:
            errors.append("saved BC actor_target hash does not match record")
        if saved_hashes["actor"] != saved_hashes["actor_target"]:
            errors.append("saved BC actor and actor_target differ")

    source_bc_fields: dict[str, Any] = {}
    if config["role"] == "post_rl":
        warm_campaign = str(config["warm_start_campaign"])
        warm_job = campaign_job(warm_campaign, region=region, seed=seed)
        warm_profile = str(CAMPAIGNS[warm_campaign]["profile"])
        source_model = model_dir(warm_job, warm_profile) / "model.zip"
        source_record_path = source_model.with_name("record.json")
        if not source_model.is_file() or not source_record_path.is_file():
            errors.append("source BC model or record is missing")
        else:
            source_record = json.loads(source_record_path.read_text(encoding="utf-8"))
            source_saved_hashes = saved_model_hashes(source_model)
            verified_source_sha = sha256(source_model)
            if record.get("source_bc_model_sha256") != verified_source_sha:
                errors.append("source BC model SHA-256 does not match training record")
            if source_record.get("actor_hash_after") != source_saved_hashes["actor"]:
                errors.append("source BC actor hash does not match its saved model")
            if source_record.get("critic_hash_after") != source_saved_hashes["critic"]:
                errors.append("source BC critic hash does not match its saved model")
            if record.get("actor_hash_before") != source_saved_hashes["actor"]:
                errors.append("post-RL actor-before hash does not match source BC actor")
            if record.get("critic_hash_before") != source_saved_hashes["critic"]:
                errors.append("post-RL critic-before hash does not match source BC critic")
            source_bc_fields = {
                "verified_source_bc_model_sha256": verified_source_sha,
                "source_bc_actor_hash": source_saved_hashes["actor"],
                "source_bc_critic_hash": source_saved_hashes["critic"],
            }
    if errors:
        raise ValueError(
            f"{campaign_name}/{region}/s{seed} violates evidence contract: "
            + "; ".join(errors)
        )
    row = {
        "campaign": campaign_name,
        "campaign_role": config["role"],
        "profile": profile,
        "seed": seed,
        "record_path": _path_text(record_path),
        "record_sha256": sha256(record_path),
        "model_path": _path_text(model_path),
        "model_sha256": sha256(model_path),
        "training_mode": record["training_mode"],
        "savings_vs_status_quo_pct": float(
            record["evaluation"]["savings_vs_status_quo_pct"]
        ),
        "safe_seed_passed": bool(record["safe_seed_passed"]),
        "emergency_intervention_rate": float(
            record["evaluation"]["safety"]["emergency_intervention_rate"]
        ),
        "decoder_adjustment_rate": float(
            record["evaluation"]["safety"]["decoder_adjustment_rate"]
        ),
        "mean_decoder_adjustment_l2": float(
            record["evaluation"]["safety"]["mean_decoder_adjustment_l2"]
        ),
        "max_decoder_adjustment_l2": float(
            record["evaluation"]["safety"]["max_decoder_adjustment_l2"]
        ),
        "service_completion": float(record["evaluation"]["service_served_total"])
        / float(record["evaluation"]["service_demand_total"]),
        "batch_completion_fraction": float(
            record["evaluation"]["batch_completion_fraction"]
        ),
        "terminal_backlog": float(record["evaluation"]["terminal_backlog"]),
        "terminal_batch_pool": float(record["evaluation"]["terminal_batch_pool"]),
        "total_batch_expired": float(record["evaluation"]["total_batch_expired"]),
        "actor_hash_before": record["actor_hash_before"],
        "actor_hash_after": record["actor_hash_after"],
        "critic_hash_before": record["critic_hash_before"],
        "critic_hash_after": record["critic_hash_after"],
        "weight_update_evidence": record["weight_update_evidence"],
        "saved_actor_hash": saved_hashes["actor"],
        "saved_actor_target_hash": saved_hashes["actor_target"],
        "saved_critic_hash": saved_hashes["critic"],
        "warm_start_actor_target_sync": record.get(
            "warm_start_actor_target_sync",
            {},
        ),
        "n_updates": int(record.get("n_updates", 0)),
        "teacher_present_during_rl": bool(record["teacher_present_during_rl"]),
        "teacher_present_at_inference": bool(record["teacher_present_at_inference"]),
        "teacher_action_audit": record["teacher"]["action_audit"],
        "descriptive_transfer_eh": {
            "savings_vs_status_quo_pct": float(
                record["descriptive_transfer_eh"]["savings_vs_status_quo_pct"]
            ),
            "emergency_intervention_rate": float(
                record["descriptive_transfer_eh"]["safety"][
                    "emergency_intervention_rate"
                ]
            ),
            "decoder_adjustment_rate": float(
                record["descriptive_transfer_eh"]["safety"]["decoder_adjustment_rate"]
            ),
        },
        "source_commit": record["source_commit"],
        "protocol_id": record["protocol_id"],
        "protocol_sha256": record["protocol_sha256"],
        "stable_baselines3_version": record["stable_baselines3_version"],
        "provenance_pass": True,
        **source_bc_fields,
    }
    if "source_bc_model" in record:
        row["source_bc_model"] = record["source_bc_model"]
        row["source_bc_model_sha256"] = record["source_bc_model_sha256"]
    if "rl_hyperparameters" in record:
        row["rl_hyperparameters"] = record["rl_hyperparameters"]
    if config["role"] == "post_rl":
        row["provenance_pass"] = post_rl_seed_provenance_pass(row)
        if not row["provenance_pass"]:
            raise ValueError(f"{campaign_name}/{region}/s{seed} failed post-RL provenance")
    return row


def variant_manifest(
    campaign_name: str,
    region: str,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    rows = [seed_record(campaign_name, region, seed) for seed in seeds]
    savings = [row["savings_vs_status_quo_pct"] for row in rows]
    emergency = [row["emergency_intervention_rate"] for row in rows]
    decoder_rates = [row["decoder_adjustment_rate"] for row in rows]
    transfer = [
        row["descriptive_transfer_eh"]["savings_vs_status_quo_pct"] for row in rows
    ]
    return {
        "campaign": campaign_name,
        "campaign_role": CAMPAIGNS[campaign_name]["role"],
        "profile": str(CAMPAIGNS[campaign_name]["profile"]),
        "seed_records": rows,
        "aggregate": {
            "seed_count": len(rows),
            "mean_savings_vs_status_quo_pct": mean(savings),
            "min_savings_vs_status_quo_pct": min(savings),
            "max_savings_vs_status_quo_pct": max(savings),
            "mean_emergency_intervention_rate": mean(emergency),
            "max_emergency_intervention_rate": max(emergency),
            "mean_decoder_adjustment_rate": mean(decoder_rates),
            "max_decoder_adjustment_rate": max(decoder_rates),
            "all_safe": all(row["safe_seed_passed"] for row in rows),
            "all_service_complete": all(
                math.isclose(row["service_completion"], 1.0, abs_tol=1e-8)
                for row in rows
            ),
            "all_batch_complete": all(
                math.isclose(row["batch_completion_fraction"], 1.0, abs_tol=1e-8)
                for row in rows
            ),
            "all_terminal_clear": all(
                row["terminal_backlog"] <= 1e-8
                and row["terminal_batch_pool"] <= 1e-8
                and row["total_batch_expired"] <= 1e-8
                for row in rows
            ),
            "all_provenance_pass": all(row["provenance_pass"] for row in rows),
            "descriptive_eh_mean_savings_vs_status_quo_pct": mean(transfer),
            "descriptive_eh_min_savings_vs_status_quo_pct": min(transfer),
        },
    }


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _protocol_hash_at_commit(commit: str) -> str:
    relative = PROTOCOL_PATH.relative_to(ROOT).as_posix()
    payload = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
    )
    normalized = payload.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


TRAINING_SOURCE_PATHS = (
    "requirements.txt",
    "baselines.py",
    "evaluate.py",
    "env",
    "data/cells",
    "data/jobs",
    "data/power_model_params.json",
    "data/energy_model_v2/2025/processed",
    "scripts/run_offpolicy_campaign_v5.py",
)


def _training_source_unchanged_since(commit: str) -> bool:
    return (
        subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                f"{commit}..HEAD",
                "--",
                *TRAINING_SOURCE_PATHS,
            ],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )


def source_commit_contract_pass(
    source_commits: set[str],
    *,
    source_is_ancestor: bool,
    training_source_unchanged: bool,
    committed_protocol_sha256: str,
) -> bool:
    return bool(
        len(source_commits) == 1
        and source_is_ancestor
        and training_source_unchanged
        and committed_protocol_sha256 == PROTOCOL_SHA256
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-campaign", required=True, choices=tuple(sorted(CAMPAIGNS)))
    parser.add_argument(
        "--postrl-campaign",
        required=True,
        choices=tuple(sorted(CAMPAIGNS)),
    )
    parser.add_argument("--suffix", default="v3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)

    validate_campaign_roles(args.bc_campaign, args.postrl_campaign)
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty unique list")
    state = workspace_state()

    manifests: dict[str, Any] = {}
    source_commits: set[str] = set()
    region_payloads: dict[str, dict[str, Any]] = {}
    for region in ("global", "us"):
        bc_only = variant_manifest(args.bc_campaign, region, seeds)
        post_rl = variant_manifest(args.postrl_campaign, region, seeds)
        source_commits.update(
            row["source_commit"]
            for variant in (bc_only, post_rl)
            for row in variant["seed_records"]
        )
        post_aggregate = post_rl["aggregate"]
        performance_gate = (
            post_aggregate["all_safe"]
            and post_aggregate["all_service_complete"]
            and post_aggregate["all_batch_complete"]
            and post_aggregate["all_terminal_clear"]
            and post_aggregate["max_emergency_intervention_rate"] < 0.01
            and post_aggregate["min_savings_vs_status_quo_pct"]
            >= (10.0 if region == "global" else 5.0)
        )
        manifest = {
            "region": region,
            "protocol": {
                "id": PROTOCOL_ID,
                "path": _path_text(PROTOCOL_PATH),
                "sha256": PROTOCOL_SHA256,
            },
            "workspace_state": state,
            "bc_only": bc_only,
            "post_rl_td3bc": post_rl,
            "comparison": {
                "bc_only_mean_savings_vs_status_quo_pct": bc_only["aggregate"][
                    "mean_savings_vs_status_quo_pct"
                ],
                "post_rl_mean_savings_vs_status_quo_pct": post_aggregate[
                    "mean_savings_vs_status_quo_pct"
                ],
                "bc_only_min_savings_vs_status_quo_pct": bc_only["aggregate"][
                    "min_savings_vs_status_quo_pct"
                ],
                "post_rl_min_savings_vs_status_quo_pct": post_aggregate[
                    "min_savings_vs_status_quo_pct"
                ],
                "post_rl_provenance_gate": post_aggregate["all_provenance_pass"],
                "post_rl_performance_gate": performance_gate,
                "post_rl_meets_gate": (
                    post_aggregate["all_provenance_pass"] and performance_gate
                ),
            },
        }
        region_payloads[region] = manifest

    if len(source_commits) != 1:
        raise ValueError(f"records span multiple source commits: {sorted(source_commits)}")
    source_commit = next(iter(source_commits))
    source_is_ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )
    training_source_unchanged = _training_source_unchanged_since(source_commit)
    committed_protocol_sha256 = _protocol_hash_at_commit(source_commit)
    if not source_commit_contract_pass(
        source_commits,
        source_is_ancestor=source_is_ancestor,
        training_source_unchanged=training_source_unchanged,
        committed_protocol_sha256=committed_protocol_sha256,
    ):
        raise ValueError(
            "record source commit/protocol is not an unchanged ancestor of current training source"
        )

    canonical_dir = OUT_ROOT / f"canonical_v5_{args.suffix}"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    for region, manifest in region_payloads.items():
        manifest["source_commit"] = source_commit
        manifest_path = OUT_ROOT / f"final_td3bc_manifest_{region}_{args.suffix}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests[region] = {
            "path": _path_text(manifest_path),
            "sha256": sha256(manifest_path),
        }

    seed_text = " ".join(str(seed) for seed in seeds)
    bc_cmd = (
        "python scripts\\run_offpolicy_campaign_v5.py "
        f"--campaign {args.bc_campaign} --regions us global --seeds "
        f"{seed_text} --workers {int(args.workers)}"
    )
    postrl_cmd = (
        "python scripts\\run_offpolicy_campaign_v5.py "
        f"--campaign {args.postrl_campaign} --regions us global --seeds "
        f"{seed_text} --workers {int(args.workers)}"
    )
    build_cmd = (
        "python scripts\\build_offpolicy_evidence_v5.py "
        f"--bc-campaign {args.bc_campaign} --postrl-campaign {args.postrl_campaign} "
        f"--seeds {seed_text} --workers {int(args.workers)} --suffix {args.suffix}"
    )
    reproduce = {
        "bc_command": bc_cmd,
        "postrl_command": postrl_cmd,
        "build_command": build_cmd,
        "smoke_test_command": "python scripts\\smoke_test_offpolicy_v5.py",
        "teacher_present_during_rl_semantics": (
            "false means no teacher policy calls or fresh labels during reward-update "
            "rollouts or gradients; bounded offline teacher demonstrations prefill replay."
        ),
    }
    (canonical_dir / "reproduce_config.json").write_text(
        json.dumps(reproduce, indent=2),
        encoding="utf-8",
    )
    evidence = {
        "package_version": f"v5-td3bc-{args.suffix}",
        "source_commit": source_commit,
        "protocol": {
            "id": PROTOCOL_ID,
            "path": _path_text(PROTOCOL_PATH),
            "sha256": PROTOCOL_SHA256,
        },
        "workspace_state": state,
        "final_manifests": manifests,
        "reproduce_config": _path_text(canonical_dir / "reproduce_config.json"),
        "commands": reproduce,
    }
    (canonical_dir / "evidence_manifest.json").write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# Canonical v5 TD3+BC evidence package",
        "",
        f"- Source commit: `{source_commit}`",
        f"- Protocol: `{PROTOCOL_ID}` (`{PROTOCOL_SHA256}`)",
        f"- BC-only campaign: `{args.bc_campaign}`",
        f"- Post-RL campaign: `{args.postrl_campaign}`",
        f"- Global manifest: `{manifests['global']['path']}`",
        f"- US manifest: `{manifests['us']['path']}`",
        f"- Build command: `{build_cmd}`",
        "- Teacher is absent during reward updates and inference.",
        "- Normal constraint-decoder adjustment and emergency fallback are reported separately.",
    ]
    (canonical_dir / "REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
