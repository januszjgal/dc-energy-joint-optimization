"""Freeze, run, and seal the immutable original-v1 100k replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.ramp_v6.factory import make_energy_model_v3_env  # noqa: E402
from ramp_rl.evaluation import evaluate_checkpoint  # noqa: E402
from ramp_rl.evidence import sha256_file, verify_pure_rl_manifest  # noqa: E402
from ramp_rl.runner import run_training, source_bundle_hash  # noqa: E402
from ramp_rl.schema import FORBIDDEN_TRAINING_INPUTS  # noqa: E402

V1_COMMIT = "b1bb302"
PROTOCOL_PATH = ROOT / "env" / "protocols" / "v6_pure_ramp_rl_v3.yaml"
FACTORY_MANIFEST = ROOT / "output" / "energy_model_v3" / "ramp_v6" / "factory_manifest.json"
EXPECTED_SEEDS = [2801, 2802, 2803, 2804, 2805]
NOMINAL_TIMESTEPS = 100_000
EFFECTIVE_TIMESTEPS = 110_592
V1_BUNDLE_PATHS = (
    "ramp_rl/campaign.py",
    "ramp_rl/contract.py",
    "ramp_rl/evaluation.py",
    "ramp_rl/evidence.py",
    "ramp_rl/runner.py",
    "ramp_rl/schema.py",
    "env/ramp_v6/environment.py",
    "env/ramp_v6/factory.py",
    "env/ramp_v6/fixture.py",
    "env/ramp_v6/models.py",
    "env/ramp_v6/panel.py",
    "env/ramp_v6/projection.py",
    "env/ramp_v6/protocol.py",
    "env/ramp_v6/reward.py",
    "env/protocols/v6_ramp_pure_rl.yaml",
    "env/protocols/v6_pure_ramp_rl.yaml",
    "env/protocols/v6_pure_ramp_rl.schema.json",
    "env/protocols/v6_ramp_panel.schema.json",
)


def _normalized_sha256(path: Path) -> str:
    payload = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_v3_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    path = path.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "v3 protocol must be a mapping")
    _require(
        payload["protocol"]["id"]
        == "v6-ramp-pure-rl-v1-replication-earlystop-v3",
        "unexpected v3 protocol ID",
    )
    links = payload["protocol"]["scientific_links"]
    _require(
        links
        == {
            "v1_closeout": "b1bb302",
            "v1_100k_screen": "7b21499",
            "v1_500k_confirmation": "3391440",
            "v2_b_failure": "0d3dcab",
            "v2_a_failure": "82f1601",
        },
        "v3 scientific links are not frozen",
    )
    environment = payload["environment_protocol"]
    _require(
        environment
        == {
            "path": "env/protocols/v6_ramp_pure_rl.yaml",
            "id": "ramp-v6-pure-rl-frozen-v1",
            "panel_schema": "env/protocols/v6_ramp_panel.schema.json",
            "cadence": "hourly UTC",
            "semantic_action_id": "ramp-v6-constraint-decoded-preferences-2n-plus-1-v1",
            "dimensions_for_n_sites": "2N+1",
            "bounds": [-6.0, 6.0],
            "execution_source_commit": "b1bb302",
        },
        "v3 must bind the original frozen v1 environment",
    )
    attribution = payload["attribution"]
    _require(attribution["random_initialization_only"] is True, "random initialization is required")
    for key in FORBIDDEN_TRAINING_INPUTS:
        _require(attribution[key] is False, f"forbidden pure-RL input enabled: {key}")
    training = payload["training"]
    _require(training["allowed_seeds"] == EXPECTED_SEEDS, "v3 seed set changed")
    _require(
        int(training["nominal_target_timesteps"]) == NOMINAL_TIMESTEPS
        and int(training["effective_boundary_timesteps"]) == EFFECTIVE_TIMESTEPS,
        "v3 stopping boundary changed",
    )
    _require(training["normalization"]["reward_training_only"] is True, "v1 reward normalization changed")
    ppo = payload["algorithms"]["ppo"]
    _require(
        {
            "learning_rate": float(ppo["learning_rate"]),
            "n_epochs": int(ppo["n_epochs"]),
            "n_steps": int(ppo["n_steps"]),
            "batch_size": int(ppo["batch_size"]),
            "gamma": float(ppo["gamma"]),
            "gae_lambda": float(ppo["gae_lambda"]),
            "net_arch": list(ppo["net_arch"]),
        }
        == {
            "learning_rate": 3e-4,
            "n_epochs": 10,
            "n_steps": 512,
            "batch_size": 256,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "net_arch": [256, 256],
        },
        "original v1 PPO parameters changed",
    )
    _require(payload["campaign"]["seed_values"] == EXPECTED_SEEDS, "campaign seed set changed")
    _require(payload["campaign"]["automatic_follow_on_protocol"] is False, "automatic protocol iteration is prohibited")
    _require(payload["success_gate"]["post_hoc_tolerance"] is False, "post-hoc tolerance is prohibited")
    _require(payload["data"]["split"]["test"]["sealed"] is True, "test must remain sealed")

    environment_path = ROOT / environment["path"]
    environment_payload = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    _require(
        environment_payload["protocol"]["id"] == "ramp-v6-pure-rl-frozen-v1",
        "live environment protocol is not frozen v1",
    )
    objective = environment_payload["ramp_objective"]
    _require(
        objective["scalar_reward"]["id"] == "ramp-v6-pure-rl-scalar-v1"
        and "robust_market_non_harm" not in objective
        and "potential_shaping" not in objective,
        "v2 reward or shaping terms leaked into v3",
    )
    panel_schema_path = ROOT / environment["panel_schema"]
    component_hashes = {
        "campaign": _normalized_sha256(path),
        "environment": _normalized_sha256(environment_path),
        "panel_schema": _normalized_sha256(panel_schema_path),
    }
    payload["_path"] = str(path)
    payload["_document_sha256"] = component_hashes["campaign"]
    payload["_environment_protocol_path"] = str(environment_path)
    payload["_panel_schema_path"] = str(panel_schema_path)
    payload["_component_sha256"] = component_hashes
    payload["_sha256"] = hashlib.sha256(
        json.dumps(component_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _git_blob(commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _bundle_sha256(blobs: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative in V1_BUNDLE_PATHS:
        digest.update(relative.encode("utf-8"))
        digest.update(blobs[relative])
    return digest.hexdigest()


def _git_object_id(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


BEHAVIOR_PROGRAM = r"""
import hashlib
import json
import numpy as np
from env.ramp_v6.factory import make_fixture_env
from ramp_rl.contract import EnvRequest

request = EnvRequest(split="validation", seed=9187, window_id="v1-regression", training=False)
env = make_fixture_env(request)
observation, reset_info = env.reset(seed=9187)
rows = [{
    "observation": hashlib.sha256(np.ascontiguousarray(observation).tobytes()).hexdigest(),
    "source_hashes": reset_info["episode_context"]["source_hashes"],
}]
terminated = False
step = 0
while not terminated:
    action = np.linspace(-1.25, 1.25, env.action_space.shape[0], dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    rows.append({
        "step": step,
        "observation": hashlib.sha256(np.ascontiguousarray(observation).tobytes()).hexdigest(),
        "reward": float(reward).hex(),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "service_unserved": float(info["service_unserved"]).hex(),
        "batch_unfinished": float(info["batch_unfinished"]).hex(),
        "batch_expired": float(info["batch_expired"]).hex(),
        "terminal_work": float(info["terminal_work"]).hex(),
        "certificate_violations": int(info["certificate_violations"]),
        "action_provenance": info["action_provenance"],
    })
    step += 1
env.close()
print(json.dumps(rows, sort_keys=True, separators=(",", ":")))
"""


def _behavior_payload(cwd: Path) -> str:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": str(cwd),
        }
    )
    return subprocess.run(
        [sys.executable, "-c", BEHAVIOR_PROGRAM],
        cwd=cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def freeze(output: Path) -> dict[str, Any]:
    protocol = load_v3_protocol()
    expected_blobs = {
        relative: _git_blob(V1_COMMIT, relative) for relative in V1_BUNDLE_PATHS
    }
    rows = []
    canonical_current_blobs = {}
    for relative in V1_BUNDLE_PATHS:
        current = (ROOT / relative).read_bytes()
        expected = expected_blobs[relative]
        canonical_current = current.replace(b"\r\n", b"\n")
        canonical_current_blobs[relative] = canonical_current
        current_object = _git_object_id(["hash-object", relative])
        expected_object = _git_object_id(["rev-parse", f"{V1_COMMIT}:{relative}"])
        rows.append(
            {
                "path": relative,
                "current_worktree_sha256": hashlib.sha256(current).hexdigest(),
                "current_canonical_sha256": hashlib.sha256(canonical_current).hexdigest(),
                "b1bb302_sha256": hashlib.sha256(expected).hexdigest(),
                "current_git_object": current_object,
                "b1bb302_git_object": expected_object,
                "repository_blob_equal": current_object == expected_object,
                "canonical_byte_equal": canonical_current == expected,
            }
        )
    if not all(
        row["repository_blob_equal"] and row["canonical_byte_equal"]
        for row in rows
    ):
        raise RuntimeError("v1 execution bundle differs from b1bb302")
    expected_bundle = _bundle_sha256(expected_blobs)
    canonical_current_bundle = _bundle_sha256(canonical_current_blobs)
    current_bundle = source_bundle_hash()
    if canonical_current_bundle != expected_bundle:
        raise RuntimeError("canonical v1 source bundle digest differs from b1bb302")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="b1bb302-", dir=output.parent) as directory:
        snapshot = Path(directory)
        archive = snapshot / "source.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={archive}", V1_COMMIT],
            cwd=ROOT,
            check=True,
        )
        extracted = snapshot / "source"
        extracted.mkdir()
        with tarfile.open(archive) as handle:
            handle.extractall(extracted, filter="data")
        reference_behavior = _behavior_payload(extracted)
    current_behavior = _behavior_payload(ROOT)
    if current_behavior != reference_behavior:
        raise RuntimeError("deterministic v1 behavior differs from b1bb302")
    behavior_sha256 = hashlib.sha256(current_behavior.encode("utf-8")).hexdigest()

    factory_manifest = json.loads(FACTORY_MANIFEST.read_text(encoding="utf-8"))
    evidence = {
        "schema_version": "ramp-pure-rl-v3-source-freeze-v1",
        "protocol_id": protocol["protocol"]["id"],
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": protocol["_sha256"],
        "protocol_component_sha256": protocol["_component_sha256"],
        "source_reference_commit": V1_COMMIT,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_files": rows,
        "current_source_bundle_sha256": current_bundle,
        "current_canonical_source_bundle_sha256": canonical_current_bundle,
        "b1bb302_canonical_source_bundle_sha256": expected_bundle,
        "byte_equivalent_to_b1bb302": True,
        "deterministic_regression": {
            "program_sha256": hashlib.sha256(BEHAVIOR_PROGRAM.encode("utf-8")).hexdigest(),
            "current_sha256": behavior_sha256,
            "b1bb302_sha256": behavior_sha256,
            "equal": True,
        },
        "factory": {
            "callable": "env.ramp_v6.factory:make_energy_model_v3_env",
            "source_sha256": sha256_file(ROOT / "env" / "ramp_v6" / "factory.py"),
            "manifest_path": str(FACTORY_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": sha256_file(FACTORY_MANIFEST),
            "frozen_stats_sha256": factory_manifest["frozen_stats_sha256"],
            "forecast_model": factory_manifest["forecast_model"],
            "source_panel_manifest_sha256": factory_manifest["source_panel_manifest_sha256"],
            "raw_acquisition_manifest_sha256": factory_manifest["raw_acquisition_manifest_sha256"],
        },
        "test_opened": False,
        "commands": [
            "python scripts\\run_ramp_rl_v3.py freeze --output output\\ramp_rl_v6\\live_v3\\source_freeze.json",
            "python scripts\\run_ramp_rl_v3.py train --seed <2801-2805> --output models\\ramp_rl_v6\\live_v3\\confirmation\\ppo\\<seed>",
            "python scripts\\run_ramp_rl_v3.py evaluate --seed <2801-2805> --checkpoint models\\ramp_rl_v6\\live_v3\\confirmation\\ppo\\<seed> --output output\\ramp_rl_v6\\live_v3\\validation\\ppo_<seed>_validation.json",
        ],
    }
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def train(seed: int, output: Path) -> dict[str, Any]:
    protocol = load_v3_protocol()
    _require(seed in EXPECTED_SEEDS, "seed is outside the frozen v3 set")
    _require(not output.exists(), "fresh v3 training output already exists")
    manifest = run_training(
        factory=make_energy_model_v3_env,
        protocol=protocol,
        algorithm="ppo",
        seed=seed,
        target_timesteps=NOMINAL_TIMESTEPS,
        output_dir=output,
        n_envs=4,
        resume=False,
        epsilon_pct=2.0,
    )
    _require(
        manifest["requested_target_timesteps"] == NOMINAL_TIMESTEPS
        and manifest["effective_boundary_target_timesteps"] == EFFECTIVE_TIMESTEPS
        and manifest["interaction_count"] == EFFECTIVE_TIMESTEPS,
        "training did not stop at the frozen 100k complete boundary",
    )
    return manifest


def evaluate(seed: int, checkpoint: Path, output: Path) -> dict[str, Any]:
    load_v3_protocol()
    _require(seed in EXPECTED_SEEDS, "seed is outside the frozen v3 set")
    factory = json.loads(FACTORY_MANIFEST.read_text(encoding="utf-8"))
    windows = sorted(factory["windows"]["validation"])
    _require(
        windows and all(factory["windows"]["validation"][name]["period"] == "2026-02" for name in windows),
        "evaluation windows are not exclusively February validation",
    )
    summary = evaluate_checkpoint(
        factory=make_energy_model_v3_env,
        algorithm="ppo",
        checkpoint_dir=checkpoint,
        split="validation",
        seeds=[seed],
        windows=windows,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _git_json(commit: str, relative: str) -> dict[str, Any]:
    return json.loads(_git_blob(commit, relative).decode("utf-8"))


def decide(validation_root: Path, model_root: Path, freeze_path: Path, output: Path, report: Path) -> dict[str, Any]:
    protocol = load_v3_protocol()
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    rows = []
    for seed in EXPECTED_SEEDS:
        validation_path = validation_root / f"ppo_{seed}_validation.json"
        manifest_path = model_root / str(seed) / "training_manifest.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_errors = verify_pure_rl_manifest(manifest)
        _require(not manifest_errors, f"seed {seed} pure-RL evidence failed: {manifest_errors}")
        _require(manifest["protocol_id"] == protocol["protocol"]["id"], "protocol ID mismatch")
        _require(manifest["protocol_sha256"] == protocol["_sha256"], "protocol hash mismatch")
        _require(
            manifest["source_bundle_sha256"] == frozen["current_source_bundle_sha256"],
            "source bundle changed after freeze",
        )
        _require(validation["split"] == "validation", "non-validation evidence supplied")
        rows.append(
            {
                "seed": seed,
                "mean_incremental_ramp_impact": validation["mean_incremental_ramp_impact"],
                "per_market_macro": validation["per_market_macro"],
                "energy_cost_ratio": validation["energy_cost_ratio"],
                "emergency_feasibility_rate": validation["emergency_feasibility_rate"],
                "behavior_audit": validation["behavior_audit"],
                "success_gate_pass": validation["success_gate"]["passed"],
                "failed_gates": validation["success_gate"]["failed_gates"],
                "validation_path": str(validation_path),
                "validation_sha256": sha256_file(validation_path),
                "training_manifest_path": str(manifest_path),
                "training_manifest_sha256": sha256_file(manifest_path),
                "initial_policy_sha256": manifest["initial_policy_sha256"],
                "initial_critic_sha256": manifest["initial_critic_sha256"],
                "final_policy_sha256": manifest["final_policy_sha256"],
                "final_critic_sha256": manifest["final_critic_sha256"],
                "interactions": manifest["interaction_count"],
                "updates": manifest["update_count"],
                "pure_rl_pass": manifest["pure_rl_verification"]["passed"],
                "source_hashes": sorted(
                    {
                        value
                        for episode in manifest["training_data_provenance"]["episodes"]
                        for value in episode["source_hashes"].values()
                    }
                ),
                "forecast_identities": manifest["forecast_identity"]["observed_training_episodes"],
                "factory": manifest["training_data_provenance"]["environment_factory"],
            }
        )
    selected = all(row["success_gate_pass"] for row in rows)
    v1 = json.loads(
        (ROOT / "output" / "ramp_rl_v6" / "live" / "screen_summary.json").read_text(encoding="utf-8")
    )
    v2_a = json.loads(
        (ROOT / "output" / "ramp_rl_v6" / "live_v2" / "candidate_result.json").read_text(encoding="utf-8")
    )
    v2_b = _git_json(
        "0d3dcab",
        "output/ramp_rl_v6/live_v2/validation_candidate_evidence.json",
    )
    aggregate = {
        "seed_count": len(rows),
        "mean_incremental_ramp_impact": mean(row["mean_incremental_ramp_impact"] for row in rows),
        "mean_energy_cost_ratio": mean(row["energy_cost_ratio"] for row in rows),
        "strict_pass_count": sum(row["success_gate_pass"] for row in rows),
    }
    result = {
        "schema_version": "ramp-pure-rl-v3-validation-decision-v1",
        "protocol_id": protocol["protocol"]["id"],
        "protocol_sha256": protocol["_sha256"],
        "selection_split": "validation",
        "all_five_pass": selected,
        "selected": selected,
        "test_opened": False,
        "test_open_count": 0,
        "automatic_follow_on_protocol_launched": False,
        "aggregate_for_reporting_only": aggregate,
        "rows": rows,
        "paired_context": {
            "v1_100k_screen": v1["aggregates"]["ppo"],
            "v2_a": v2_a["aggregate"],
            "v2_b": v2_b,
        },
        "source_freeze": {
            "path": str(freeze_path),
            "sha256": sha256_file(freeze_path),
            "source_bundle_sha256": frozen["current_source_bundle_sha256"],
            "byte_equivalent_to_b1bb302": frozen["byte_equivalent_to_b1bb302"],
            "deterministic_regression_equal": frozen["deterministic_regression"]["equal"],
        },
        "decision": (
            "open-sealed-test-once-for-all-five"
            if selected
            else "publish-v3-failure-and-keep-test-sealed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Immutable pure-RL v3 validation result",
        "",
        f"Protocol `{protocol['protocol']['id']}` replicated original v1 PPO at the "
        "100k nominal / 110,592 complete-boundary stop on fresh seeds 2801-2805.",
        "",
        "| Seed | Raw ramp impact | Cost ratio | Strict pass | Failed gates |",
        "|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['mean_incremental_ramp_impact']:.12g} | "
            f"{row['energy_cost_ratio']:.10f} | {row['success_gate_pass']} | "
            f"{', '.join(row['failed_gates']) or '-'} |"
        )
    lines.extend(
        [
            "",
            f"All-five selection decision: **{selected}**.",
            f"Sealed March-April test opened: **false**.",
            "No follow-on protocol was launched automatically.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--seed", type=int, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    decide_parser = subparsers.add_parser("decide")
    decide_parser.add_argument("--validation-root", type=Path, required=True)
    decide_parser.add_argument("--model-root", type=Path, required=True)
    decide_parser.add_argument("--freeze", type=Path, required=True)
    decide_parser.add_argument("--output", type=Path, required=True)
    decide_parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        result = freeze(args.output)
    elif args.command == "train":
        result = train(args.seed, args.output)
    elif args.command == "evaluate":
        result = evaluate(args.seed, args.checkpoint, args.output)
    else:
        result = decide(
            args.validation_root,
            args.model_root,
            args.freeze,
            args.output,
            args.report,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
