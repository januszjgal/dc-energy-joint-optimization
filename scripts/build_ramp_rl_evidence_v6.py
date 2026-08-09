"""Build a compact evidence index from completed isolated v6 pure-RL jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stable_baselines3 import PPO, SAC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ramp_rl.evidence import sha256_file, verify_pure_rl_manifest  # noqa: E402
from ramp_rl.runner import model_hashes, source_bundle_hash  # noqa: E402
from ramp_rl.schema import DEFAULT_PROTOCOL_PATH, load_protocol  # noqa: E402


def _artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_saved_artifacts(
    record: dict[str, object], protocol: dict[str, object]
) -> list[str]:
    errors: list[str] = []
    artifacts = record.get("artifacts", {})
    for name in ("model", "normalization", "replay"):
        artifact = artifacts.get(name)
        if artifact is None:
            if name == "replay" and record.get("algorithm") == "sac":
                errors.append("SAC replay artifact is missing")
            continue
        path = _artifact_path(str(artifact["path"]))
        if not path.exists():
            errors.append(f"{name} artifact is missing")
        elif sha256_file(path) != artifact["sha256"]:
            errors.append(f"{name} artifact hash mismatch")
    model_artifact = artifacts.get("model")
    if model_artifact is not None:
        model_path = _artifact_path(str(model_artifact["path"]))
        if model_path.exists():
            model_class = PPO if record["algorithm"] == "ppo" else SAC
            hashes = model_hashes(model_class.load(model_path, device="cpu"))
            if hashes["policy"] != record["final_policy_sha256"]:
                errors.append("saved policy tensor hash mismatch")
            if hashes["critic"] != record["final_critic_sha256"]:
                errors.append("saved critic tensor hash mismatch")
    if record.get("source_bundle_sha256") != source_bundle_hash(protocol):
        errors.append("training source bundle hash mismatch")
    factory = record.get("job_identity", {}).get("factory", {})
    source_path = ROOT / str(factory.get("source_path", ""))
    if not source_path.is_file() or sha256_file(source_path) != factory.get("source_sha256"):
        errors.append("environment factory source hash mismatch")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", nargs="+", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    rows = []
    errors = []
    for job in args.jobs:
        path = job / "training_manifest.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record_errors = verify_pure_rl_manifest(record)
        record_errors.extend(verify_saved_artifacts(record, protocol))
        if record["protocol_sha256"] != protocol["_sha256"]:
            record_errors.append("protocol hash mismatch")
        if (
            record.get("environment_contract", {}).get("protocol_id")
            != protocol["environment_protocol"]["id"]
        ):
            record_errors.append("environment protocol mismatch")
        expected_reward_normalization = bool(
            protocol["training"]["normalization"].get("reward", True)
        )
        if (
            bool(
                record.get("normalization", {}).get(
                    "reward_normalization_training_only"
                )
            )
            != expected_reward_normalization
        ):
            record_errors.append("reward normalization mismatch")
        rows.append(
            {
                "job": str(job),
                "manifest_sha256": sha256_file(path),
                "algorithm": record["algorithm"],
                "seed": record["seed"],
                "interactions": record["interaction_count"],
                "updates": record["update_count"],
                "initial_policy_sha256": record["initial_policy_sha256"],
                "final_policy_sha256": record["final_policy_sha256"],
                "pure_rl_pass": not record_errors,
                "errors": record_errors,
            }
        )
        errors.extend(f"{job}: {error}" for error in record_errors)
    evidence = {
        "schema_version": "ramp-pure-rl-evidence-index-v1",
        "protocol_id": protocol["protocol"]["id"],
        "protocol_sha256": protocol["_sha256"],
        "final_long_campaign_launched": False,
        "integration_blockers": protocol["campaign"]["final_campaign_blocked_until"],
        "jobs": rows,
        "passed": not errors,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("evidence verification failed")


if __name__ == "__main__":
    main()
