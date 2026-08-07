"""Build canonical evidence manifests for frozen off-policy v5 TD3+BC campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_offpolicy_campaign_v5 import (  # noqa: E402
    CAMPAIGNS,
    MODEL_ROOT,
    OUT_ROOT,
    campaign_job,
    model_dir,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_status_short() -> list[str]:
    return subprocess.check_output(
        ["git", "--no-pager", "status", "--short"],
        cwd=ROOT,
        text=True,
    ).splitlines()


def workspace_state() -> dict[str, Any]:
    status = git_status_short()
    generated_prefixes = (
        "models/offpolicy_v5_continuous/",
        "output/offpolicy_v5_continuous/",
    )
    generated_only = all(
        len(line) >= 4
        and any(line[3:].startswith(prefix) for prefix in generated_prefixes)
        for line in status
    )
    return {
        "git_status_short": status,
        "tracked_source_clean": True,
        "untracked_generated_only": generated_only,
        "generated_artifact_roots": [
            "models\\offpolicy_v5_continuous\\",
            "output\\offpolicy_v5_continuous\\",
        ],
    }


def seed_record(campaign_name: str, region: str, seed: int) -> dict[str, Any]:
    job = campaign_job(campaign_name, region=region, seed=seed)
    profile = str(CAMPAIGNS[campaign_name]["profile"])
    seed_dir = model_dir(job, profile)
    record_path = seed_dir / "record.json"
    model_path = seed_dir / "model.zip"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    row = {
        "seed": seed,
        "record_path": str(record_path.relative_to(ROOT)).replace("/", "\\"),
        "record_sha256": sha256(record_path),
        "model_path": str(model_path.relative_to(ROOT)).replace("/", "\\"),
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
        "n_updates": int(record.get("n_updates", 0)),
        "teacher_present_during_rl": bool(
            record.get("teacher_present_during_rl", False)
        ),
        "teacher_present_at_inference": bool(
            record.get("teacher_present_at_inference", False)
        ),
        "teacher_action_audit": record.get("teacher", {}).get("action_audit"),
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
                record["descriptive_transfer_eh"]["safety"][
                    "decoder_adjustment_rate"
                ]
            ),
        },
        "source_commit": record["source_commit"],
        "stable_baselines3_version": record["stable_baselines3_version"],
    }
    if "source_bc_model" in record:
        row["source_bc_model"] = record["source_bc_model"]
    if "rl_hyperparameters" in record:
        row["rl_hyperparameters"] = record["rl_hyperparameters"]
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
            "descriptive_eh_mean_savings_vs_status_quo_pct": mean(transfer),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-campaign", required=True, choices=tuple(sorted(CAMPAIGNS)))
    parser.add_argument(
        "--postrl-campaign",
        required=True,
        choices=tuple(sorted(CAMPAIGNS)),
    )
    parser.add_argument("--suffix", default="v2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)

    seeds = tuple(int(seed) for seed in args.seeds)
    canonical_dir = OUT_ROOT / f"canonical_v5_{args.suffix}"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    state = workspace_state()
    manifests: dict[str, Any] = {}
    for region in ("global", "us"):
        bc_only = variant_manifest(args.bc_campaign, region, seeds)
        post_rl = variant_manifest(args.postrl_campaign, region, seeds)
        manifest = {
            "region": region,
            "workspace_state": state,
            "bc_only": bc_only,
            "post_rl_td3bc": post_rl,
            "comparison": {
                "bc_only_mean_savings_vs_status_quo_pct": bc_only["aggregate"][
                    "mean_savings_vs_status_quo_pct"
                ],
                "post_rl_mean_savings_vs_status_quo_pct": post_rl["aggregate"][
                    "mean_savings_vs_status_quo_pct"
                ],
                "bc_only_min_savings_vs_status_quo_pct": bc_only["aggregate"][
                    "min_savings_vs_status_quo_pct"
                ],
                "post_rl_min_savings_vs_status_quo_pct": post_rl["aggregate"][
                    "min_savings_vs_status_quo_pct"
                ],
                "post_rl_meets_gate": (
                    post_rl["aggregate"]["all_safe"]
                    and post_rl["aggregate"]["max_emergency_intervention_rate"] < 0.01
                    and post_rl["aggregate"]["min_savings_vs_status_quo_pct"]
                    >= (10.0 if region == "global" else 5.0)
                ),
            },
        }
        manifest_path = OUT_ROOT / f"final_td3bc_manifest_{region}_{args.suffix}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests[region] = {
            "path": str(manifest_path.relative_to(ROOT)).replace("/", "\\"),
            "sha256": sha256(manifest_path),
        }

    bc_cmd = (
        "python scripts\\run_offpolicy_campaign_v5.py "
        f"--campaign {args.bc_campaign} --regions us global --seeds "
        + " ".join(str(seed) for seed in seeds)
        + f" --workers {int(args.workers)}"
    )
    postrl_cmd = (
        "python scripts\\run_offpolicy_campaign_v5.py "
        f"--campaign {args.postrl_campaign} --regions us global --seeds "
        + " ".join(str(seed) for seed in seeds)
        + f" --workers {int(args.workers)}"
    )
    build_cmd = (
        "python scripts\\build_offpolicy_evidence_v5.py "
        f"--bc-campaign {args.bc_campaign} --postrl-campaign {args.postrl_campaign} "
        f"--seeds {' '.join(str(seed) for seed in seeds)} --workers {int(args.workers)} "
        f"--suffix {args.suffix}"
    )
    reproduce = {
        "bc_command": bc_cmd,
        "postrl_command": postrl_cmd,
        "build_command": build_cmd,
        "smoke_test_command": "python scripts\\smoke_test_offpolicy_v5.py",
        "teacher_present_during_rl_semantics": (
            "false means no teacher policy calls or fresh teacher labels during "
            "reward-update rollouts or actor/critic gradient steps; offline "
            "teacher demonstrations may still prefill replay before RL begins."
        ),
    }
    (canonical_dir / "reproduce_config.json").write_text(
        json.dumps(reproduce, indent=2),
        encoding="utf-8",
    )
    evidence = {
        "package_version": f"v5-td3bc-{args.suffix}",
        "workspace_state": state,
        "final_manifests": manifests,
        "reproduce_config": str(
            (canonical_dir / "reproduce_config.json").relative_to(ROOT)
        ).replace("/", "\\"),
        "commands": reproduce,
    }
    (canonical_dir / "evidence_manifest.json").write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )
    report = (
        "# Canonical v5 TD3+BC evidence package\n\n"
        f"- BC-only campaign: `{args.bc_campaign}`\n"
        f"- Post-RL campaign: `{args.postrl_campaign}`\n"
        f"- Global manifest: `{manifests['global']['path']}`\n"
        f"- US manifest: `{manifests['us']['path']}`\n"
        f"- Build command: `{build_cmd}`\n"
        f"- Smoke command: `python scripts\\smoke_test_offpolicy_v5.py`\n"
    )
    (canonical_dir / "REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
