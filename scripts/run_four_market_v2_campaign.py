"""Reproducibly run the paired four-market direct-K=1 PPO validation campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from energy_model_v3.four_market_v2 import (  # noqa: E402
    VARIANTS,
    make_envelope_off_env,
    make_envelope_on_env,
)
from ramp_rl.contract import EnvRequest  # noqa: E402
from ramp_rl.evaluation import evaluate_checkpoint  # noqa: E402
from ramp_rl.runner import run_training  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "four_market_v2" / "campaign"
MODEL_ROOT = ROOT / "models" / "four_market_v2" / "campaign"
SEEDS = (4101, 4102, 4103)
REQUESTED_TIMESTEPS = 100_000
FACTORIES: dict[str, Callable] = {
    "envelope_on": make_envelope_on_env,
    "envelope_off": make_envelope_off_env,
}


def _canonical_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _protocol(variant: str) -> dict[str, Any]:
    path = ROOT / "env" / "protocols" / f"four_market_v2_{variant}.yaml"
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    payload = yaml.safe_load(text)
    if payload["protocol"]["sealed_test_access"] or payload["data"]["split"]["test"]["months"]:
        raise ValueError("v2 campaign must not access a test split")
    if payload["variant"]["id"] != variant:
        raise ValueError("protocol variant mismatch")
    payload["_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
    payload["_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return payload


def _manifest(variant: str) -> dict[str, Any]:
    path = ROOT / "output" / "four_market_v2" / "factory" / variant / "factory_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def preflight() -> dict[str, Any]:
    """Run status quo over every permitted window to certify immediate feasibility."""
    result: dict[str, Any] = {}
    for variant, factory in FACTORIES.items():
        counts = {
            "train": 0,
            "validation": 0,
            "steps": 0,
            "max_immediate_arrival_work": 0.0,
        }
        for split, records in _manifest(variant)["windows"].items():
            for window_id in sorted(records):
                wrapper = factory(
                    EnvRequest(
                        split=split, seed=4101, window_id=window_id, training=False
                    )
                )
                try:
                    current = wrapper._current
                    immediate = (
                        current.workload.service_arrivals
                        + current.workload.batch_arrivals
                    ).sum(axis=1)
                    maximum = float(immediate.max())
                    if maximum > float(current._capacity.sum()) + 1e-12:
                        raise RuntimeError(
                            f"{variant}/{window_id} immediate execution is infeasible"
                        )
                    counts["max_immediate_arrival_work"] = max(
                        counts["max_immediate_arrival_work"], maximum
                    )
                    wrapper.reset(seed=4101)
                    while True:
                        _, _, terminated, _, info = wrapper.step(
                            wrapper.evaluation_action("status_quo")
                        )
                        counts["steps"] += 1
                        if terminated:
                            if (
                                info["service_unserved"] != 0.0
                                or info["batch_unfinished"] != 0.0
                                or info["certificate_violations"] != 0
                            ):
                                raise RuntimeError(
                                    f"{variant}/{window_id} status quo is infeasible"
                                )
                            break
                    counts[split] += 1
                finally:
                    wrapper.close()
        result[variant] = counts
    return result


def train_and_evaluate(variant: str, seed: int) -> dict[str, Any]:
    protocol = _protocol(variant)
    factory = FACTORIES[variant]
    checkpoint = MODEL_ROOT / variant / "ppo" / f"seed-{seed}"
    output_dir = OUTPUT_ROOT / variant / f"seed-{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    training = run_training(
        factory=factory,
        protocol=protocol,
        algorithm="ppo",
        seed=seed,
        target_timesteps=REQUESTED_TIMESTEPS,
        output_dir=checkpoint,
        resume=False,
    )
    elapsed = time.perf_counter() - started
    windows = sorted(_manifest(variant)["windows"]["validation"])
    validation = evaluate_checkpoint(
        factory=factory,
        algorithm="ppo",
        checkpoint_dir=checkpoint,
        split="validation",
        seeds=[seed],
        windows=windows,
    )
    raw_path = output_dir / "validation_raw.json"
    raw_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    compact = {key: value for key, value in validation.items() if not key.endswith("_episodes")}
    summary = {
        "schema_version": "four-market-v2-seed-summary-v1",
        "classification": "large_non_final_validation_only",
        "variant": variant,
        "seed": seed,
        "requested_interactions": REQUESTED_TIMESTEPS,
        "training_elapsed_seconds": elapsed,
        "training_throughput_interactions_per_second": (
            training["interaction_count_this_invocation"] / elapsed
        ),
        "training_manifest": {
            "path": str((checkpoint / "training_manifest.json").relative_to(ROOT)).replace("\\", "/"),
            "sha256": _canonical_hash(checkpoint / "training_manifest.json"),
            "effective_boundary_timesteps": training["effective_boundary_target_timesteps"],
            "interaction_count": training["interaction_count"],
        },
        "validation": compact,
        "validation_raw": {
            "path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _canonical_hash(raw_path),
        },
        "sealed_test_accessed": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return
    variants = (args.variant,) if args.variant else VARIANTS
    seeds = (args.seed,) if args.seed else SEEDS
    started = time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": "four-market-v2-paired-campaign-v1",
        "variants": list(variants),
        "paired_seeds": list(seeds),
        "requested_interactions_per_seed_variant": REQUESTED_TIMESTEPS,
        "requested_interactions_total": len(variants) * len(seeds) * REQUESTED_TIMESTEPS,
        "sealed_test_accessed": False,
        "preflight": preflight(),
    }
    (OUTPUT_ROOT / "campaign_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    summaries = [
        train_and_evaluate(variant, seed) for variant in variants for seed in seeds
    ]
    run_manifest["elapsed_seconds"] = time.perf_counter() - started
    run_manifest["completed_seed_variant_count"] = len(summaries)
    (OUTPUT_ROOT / "campaign_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"elapsed_seconds": run_manifest["elapsed_seconds"], "runs": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
