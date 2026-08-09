"""Verify and publish deterministic reproductions of the lost V3 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ramp_rl.evidence import sha256_file  # noqa: E402
from ramp_rl.runner import model_hashes, source_bundle_hash  # noqa: E402
from scripts.run_ramp_rl_v3 import EXPECTED_SEEDS, load_v3_protocol  # noqa: E402

FREEZE_PATH = ROOT / "output" / "ramp_rl_v6" / "live_v3" / "source_freeze.json"
FACTORY_MANIFEST_PATH = (
    ROOT / "output" / "energy_model_v3" / "ramp_v6" / "factory_manifest.json"
)
ORIGINAL_ROOT = (
    ROOT / "models" / "ramp_rl_v6" / "live_v3" / "confirmation" / "ppo"
)
RECOVERY_ROOT = ROOT / "models" / "ramp_rl_v6" / "recovery_v3"
RECOVERY_MANIFEST_PATH = (
    ROOT / "output" / "ramp_rl_v6" / "live_v3" / "recovery_manifest.json"
)
ARTIFACT_NAMES = ("model.zip", "vecnormalize.pkl")
COMPARISON_FIELDS = (
    "initial_policy_sha256",
    "initial_critic_sha256",
    "final_policy_sha256",
    "final_critic_sha256",
    "interaction_count",
    "interaction_count_this_invocation",
    "update_count",
    "actual_terminal_count_this_invocation",
    "requested_target_timesteps",
    "effective_boundary_target_timesteps",
    "checkpoint_boundary_quantum",
    "normalization",
    "pure_rl_assertions",
    "pure_rl_verification",
    "replay_provenance",
    "training_data_provenance",
    "forecast_identity",
    "job_identity",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _utc_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _model_container_diagnostics(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        model_data = json.loads(archive.read("data"))
        members = {
            info.filename: {
                "date_time": list(info.date_time),
                "payload_sha256": hashlib.sha256(
                    archive.read(info.filename)
                ).hexdigest(),
                "size": info.file_size,
            }
            for info in archive.infolist()
        }
    return {
        "sb3_runtime_start_time_ns": int(model_data["start_time"]),
        "sb3_runtime_start_time_utc": datetime.fromtimestamp(
            int(model_data["start_time"]) / 1_000_000_000,
            timezone.utc,
        ).isoformat(),
        "members": members,
        "diagnosis": (
            "SB3 model.zip is not a content-addressed policy artifact: its data member "
            "persists runtime start_time and writestr members persist wall-clock ZIP "
            "timestamps. Exact policy/critic state therefore does not guarantee an "
            "identical model.zip byte stream."
        ),
    }


def _check(
    checks: dict[str, dict[str, Any]],
    errors: list[str],
    label: str,
    actual: Any,
    expected: Any,
) -> None:
    matched = actual == expected
    if isinstance(actual, (dict, list)) and isinstance(expected, type(actual)):
        actual_json = json.dumps(actual, sort_keys=True, separators=(",", ":"))
        expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"))
        checks[label] = {
            "matched": matched,
            "actual_sha256": hashlib.sha256(actual_json.encode("utf-8")).hexdigest(),
            "expected_sha256": hashlib.sha256(expected_json.encode("utf-8")).hexdigest(),
        }
    else:
        checks[label] = {"matched": matched, "actual": actual, "expected": expected}
    if not matched:
        errors.append(label)


def _verify_identity(
    freeze: dict[str, Any],
    factory: dict[str, Any],
    checks: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    protocol = load_v3_protocol()
    _check(
        checks,
        errors,
        "identity.protocol_sha256",
        protocol["_sha256"],
        freeze["protocol_sha256"],
    )
    _check(
        checks,
        errors,
        "identity.source_bundle_sha256",
        source_bundle_hash(),
        freeze["current_source_bundle_sha256"],
    )
    _check(
        checks,
        errors,
        "identity.factory_manifest_sha256",
        sha256_file(FACTORY_MANIFEST_PATH),
        freeze["factory"]["manifest_sha256"],
    )
    _check(
        checks,
        errors,
        "identity.factory_source_sha256",
        sha256_file(ROOT / "env" / "ramp_v6" / "factory.py"),
        freeze["factory"]["source_sha256"],
    )
    for field in (
        "forecast_model",
        "frozen_stats_sha256",
        "raw_acquisition_manifest_sha256",
        "source_panel_manifest_sha256",
    ):
        _check(
            checks,
            errors,
            f"identity.factory.{field}",
            factory[field],
            freeze["factory"][field],
        )
    for row in freeze["source_files"]:
        path = ROOT / row["path"]
        canonical = path.read_bytes().replace(b"\r\n", b"\n")
        _check(
            checks,
            errors,
            f"identity.source_file.{row['path']}",
            hashlib.sha256(canonical).hexdigest(),
            row["current_canonical_sha256"],
        )
    for window_id, row in factory["windows"]["train"].items():
        artifact_root = ROOT / row["artifact_root"]
        for key, name in (
            ("canonical_panel", "canonical_panel.csv"),
            ("fixture", "fixture.json"),
        ):
            _check(
                checks,
                errors,
                f"identity.train_window.{window_id}.{key}",
                sha256_file(artifact_root / name),
                row["source_hashes"][key],
            )


def _verify_seed(
    seed: int,
    checks: dict[str, dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    original_manifest_path = ORIGINAL_ROOT / str(seed) / "training_manifest.json"
    recovered_manifest_path = RECOVERY_ROOT / str(seed) / "training_manifest.json"
    original = _load(original_manifest_path)
    recovered = _load(recovered_manifest_path)
    seed_prefix = f"seed.{seed}"

    for field in COMPARISON_FIELDS:
        _check(
            checks,
            errors,
            f"{seed_prefix}.{field}",
            recovered[field],
            original[field],
        )
    _check(
        checks,
        errors,
        f"{seed_prefix}.resume_fresh",
        {
            key: recovered["resume"][key]
            for key in (
                "resumed",
                "starting_interaction_count",
                "prior_final_policy_sha256",
                "loaded_policy_sha256",
                "loaded_critic_sha256",
            )
        },
        {
            "resumed": False,
            "starting_interaction_count": 0,
            "prior_final_policy_sha256": None,
            "loaded_policy_sha256": None,
            "loaded_critic_sha256": None,
        },
    )

    artifact_rows: dict[str, Any] = {}
    for artifact_name, manifest_key in (
        ("model.zip", "model"),
        ("vecnormalize.pkl", "normalization"),
    ):
        recovered_path = RECOVERY_ROOT / str(seed) / artifact_name
        actual_hash = sha256_file(recovered_path)
        original_hash = original["artifacts"][manifest_key]["sha256"]
        recovered_declared_hash = recovered["artifacts"][manifest_key]["sha256"]
        _check(
            checks,
            errors,
            f"{seed_prefix}.artifact.{artifact_name}.original",
            actual_hash,
            original_hash,
        )
        _check(
            checks,
            errors,
            f"{seed_prefix}.artifact.{artifact_name}.recovered_manifest",
            actual_hash,
            recovered_declared_hash,
        )
        artifact_rows[artifact_name] = {
            "original_sha256": original_hash,
            "reproduced_sha256": actual_hash,
            "matched": actual_hash == original_hash,
            "recovered_path": _relative(recovered_path),
            "recovered_timestamp_utc": _utc_timestamp(recovered_path),
        }

    loaded_hashes = model_hashes(
        PPO.load(RECOVERY_ROOT / str(seed) / "model.zip", device="cpu")
    )
    _check(
        checks,
        errors,
        f"{seed_prefix}.loaded_policy_sha256",
        loaded_hashes["policy"],
        original["final_policy_sha256"],
    )
    _check(
        checks,
        errors,
        f"{seed_prefix}.loaded_critic_sha256",
        loaded_hashes["critic"],
        original["final_critic_sha256"],
    )
    return {
        "seed": seed,
        "original_manifest_path": _relative(original_manifest_path),
        "original_manifest_sha256": sha256_file(original_manifest_path),
        "recovered_manifest_path": _relative(recovered_manifest_path),
        "recovered_manifest_sha256": sha256_file(recovered_manifest_path),
        "command": (
            "$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; "
            "$env:OPENBLAS_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'; "
            f"python scripts\\run_ramp_rl_v3.py train --seed {seed} "
            f"--output {RECOVERY_ROOT / str(seed)}"
        ),
        "artifacts": artifact_rows,
        "model_container_diagnostics": _model_container_diagnostics(
            RECOVERY_ROOT / str(seed) / "model.zip"
        ),
        "published": {},
    }


def _publish(seed_rows: list[dict[str, Any]]) -> None:
    destinations: list[tuple[Path, Path, dict[str, Any], str]] = []
    for row in seed_rows:
        seed = int(row["seed"])
        for artifact_name in ARTIFACT_NAMES:
            source = RECOVERY_ROOT / str(seed) / artifact_name
            destination = ORIGINAL_ROOT / str(seed) / artifact_name
            expected = row["artifacts"][artifact_name]["original_sha256"]
            if destination.exists() and sha256_file(destination) != expected:
                raise RuntimeError(f"refusing to replace mismatched artifact: {destination}")
            destinations.append((source, destination, row, artifact_name))

    created: list[Path] = []
    try:
        for source, destination, row, artifact_name in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                method = "existing_exact"
            else:
                try:
                    os.link(source, destination)
                    method = "hardlink"
                except OSError:
                    shutil.copy2(source, destination)
                    method = "copy"
                created.append(destination)
            row["published"][artifact_name] = {
                "path": _relative(destination),
                "method": method,
                "sha256": sha256_file(destination),
            }
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def recover(*, publish: bool, manifest_path: Path) -> dict[str, Any]:
    freeze = _load(FREEZE_PATH)
    factory = _load(FACTORY_MANIFEST_PATH)
    checks: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    original_manifest_hashes = {
        seed: sha256_file(ORIGINAL_ROOT / str(seed) / "training_manifest.json")
        for seed in EXPECTED_SEEDS
    }
    _verify_identity(freeze, factory, checks, errors)
    seed_rows = [_verify_seed(seed, checks, errors) for seed in EXPECTED_SEEDS]
    exact_match = not errors
    published = False
    if publish and exact_match:
        try:
            _publish(seed_rows)
            published = True
        except Exception as error:
            errors.append(f"publication: {error}")

    for seed, before_hash in original_manifest_hashes.items():
        _check(
            checks,
            errors,
            f"seed.{seed}.original_manifest_unchanged",
            sha256_file(ORIGINAL_ROOT / str(seed) / "training_manifest.json"),
            before_hash,
        )
    exact_match = not errors
    recovery_manifest = {
        "schema_version": "ramp-pure-rl-v3-recovery-v1",
        "status": (
            "exact_match_published"
            if exact_match and published
            else "exact_match_not_published"
            if exact_match
            else "deterministic_divergence"
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_path": str(ROOT),
        "source_identity": {
            "protocol_id": freeze["protocol_id"],
            "source_commits": ["f6cd9c9", "dede685"],
            "training_evidence_commits": ["fc8954b", "1f2a7dc"],
            "source_bundle_sha256": freeze["current_source_bundle_sha256"],
            "protocol_sha256": freeze["protocol_sha256"],
            "factory_manifest_sha256": freeze["factory"]["manifest_sha256"],
            "source_freeze_path": _relative(FREEZE_PATH),
            "source_freeze_sha256": sha256_file(FREEZE_PATH),
            "test_opened": False,
        },
        "recovery_root": _relative(RECOVERY_ROOT),
        "expected_root": _relative(ORIGINAL_ROOT),
        "parallel_training": True,
        "all_exact": exact_match,
        "published": published,
        "errors": errors,
        "seeds": seed_rows,
        "checks": checks,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(recovery_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return recovery_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--publish",
        action="store_true",
        help="publish exact recovered artifacts into the original local paths",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=RECOVERY_MANIFEST_PATH,
        help="recovery evidence output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = recover(publish=args.publish, manifest_path=args.manifest.resolve())
    print(json.dumps({key: result[key] for key in ("status", "all_exact", "published", "errors")}, indent=2))
    if not result["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
