"""Run the locked ten-seed raw-workload four-market PPO campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from energy_model_v3.four_market_v2 import make_four_market_env  # noqa: E402
from ramp_rl.contract import EnvRequest  # noqa: E402
from ramp_rl.evaluation import evaluate_checkpoint  # noqa: E402
from ramp_rl.runner import (  # noqa: E402
    configure_single_thread_runtime,
    run_training,
    safe_boundary_quantum,
    training_identity,
    validate_completed_training_summary,
)


OUTPUT_ROOT = ROOT / "output" / "four_market_v2" / "campaign"
MODEL_ROOT = ROOT / "models" / "four_market_v2" / "campaign"
SEEDS = tuple(range(4101, 4111))
REQUESTED_TIMESTEPS = 2_000_000
DEFAULT_WORKERS = 5


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _protocol() -> dict[str, Any]:
    payload = yaml.safe_load((ROOT / "env" / "protocols" / "four_market_v2.yaml").read_text())
    if payload["data"]["split"]["test"]["months"]:
        raise ValueError("campaign must not access a test split")
    return payload


def _factory() -> dict[str, Any]:
    return json.loads((ROOT / "output" / "four_market_v2" / "factory" / "factory.json").read_text())


def _validate_campaign_geometry() -> dict[str, list[str]]:
    windows = _factory()["windows"]
    if len(windows["train"]) != 114 or len(windows["validation"]) != 28:
        raise RuntimeError("campaign requires exactly 114 train windows and 28 February validation days")
    return {
        "train": sorted(windows["train"]),
        "validation": sorted(windows["validation"]),
    }


def _campaign_training_geometry() -> dict[str, int]:
    """Derive the locked safe boundary from the current protocol and environment."""
    protocol = _protocol()
    n_envs = int(protocol["training"]["vectorized_environments"])
    n_steps = int(protocol["training"]["ppo"]["n_steps"])
    env = make_four_market_env(
        EnvRequest(split="train", seed=SEEDS[0], rank=0, training=True)
    )
    try:
        action_steps = int(env.ramp_rl_contract()["decision_steps"])
    finally:
        env.close()
    safe_quantum = safe_boundary_quantum(
        action_steps=action_steps, n_envs=n_envs, n_steps=n_steps
    )
    return {
        "action_steps": action_steps,
        "n_envs": n_envs,
        "safe_quantum": safe_quantum,
        "effective_interactions": (
            math.ceil(REQUESTED_TIMESTEPS / safe_quantum) * safe_quantum
        ),
    }


def _campaign_training_identity(
    seed: int, training_geometry: dict[str, int]
) -> dict[str, Any]:
    """Build the current plain training identity for one campaign seed."""
    return {
        "seed": seed,
        "requested_interactions": REQUESTED_TIMESTEPS,
        "effective_interactions": training_geometry["effective_interactions"],
        "n_envs": training_geometry["n_envs"],
        "action_steps": training_geometry["action_steps"],
        "ppo_config": dict(_protocol()["training"]["ppo"]),
        "safe_quantum": training_geometry["safe_quantum"],
    }


def _load_completed_seed_summary(
    seed: int,
    geometry: dict[str, list[str]],
    training_geometry: dict[str, int],
    *,
    output_root: Path | None = None,
    model_root: Path | None = None,
) -> dict[str, Any]:
    """Load a completed result only when it matches the locked campaign identity."""
    output_root = OUTPUT_ROOT if output_root is None else output_root
    model_root = MODEL_ROOT if model_root is None else model_root
    summary_path = output_root / f"seed-{seed}" / "summary.json"
    if not summary_path.is_file():
        if summary_path.exists():
            raise RuntimeError(
                f"invalid completed seed summary for seed {seed}: {summary_path}"
            )
        raise FileNotFoundError(summary_path)
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"invalid completed seed summary for seed {seed}: {summary_path}"
        ) from error
    if not isinstance(summary, dict):
        raise RuntimeError(f"invalid completed seed summary for seed {seed}: {summary_path}")

    expected_ints = {
        "seed": seed,
        "requested_interactions": REQUESTED_TIMESTEPS,
        "effective_interactions": training_geometry["effective_interactions"],
        "safe_boundary_interactions": training_geometry["safe_quantum"],
    }
    for field, expected in expected_ints.items():
        actual = summary.get(field)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise RuntimeError(
                f"completed seed summary has incompatible {field} for seed {seed}: "
                f"{summary_path}"
            )
    if summary.get("validation_window_ids") != geometry["validation"]:
        raise RuntimeError(
            f"completed seed summary has incompatible validation windows for seed {seed}: "
            f"{summary_path}"
        )

    checkpoint = model_root / "ppo" / f"seed-{seed}"
    training_summary_path = checkpoint / "training_summary.json"
    if not training_summary_path.is_file():
        raise RuntimeError(
            f"completed seed summary is missing training summary for seed {seed}: "
            f"{training_summary_path}"
        )
    missing_artifacts = [
        path.name for path in (checkpoint / "model.zip", checkpoint / "vecnormalize.pkl")
        if not path.is_file()
    ]
    if missing_artifacts:
        raise RuntimeError(
            f"completed seed summary is missing training artifacts for seed {seed}: "
            f"{', '.join(missing_artifacts)} in {checkpoint}"
        )
    validated_training = validate_completed_training_summary(
        training_summary_path, _campaign_training_identity(seed, training_geometry)
    )
    if summary.get("training_identity") != training_identity(validated_training):
        raise RuntimeError(
            f"completed seed summary has incompatible training identity for seed {seed}: "
            f"{summary_path}"
        )

    validation = summary.get("validation")
    if not isinstance(validation, dict):
        raise RuntimeError(f"completed seed summary has invalid validation for seed {seed}: {summary_path}")
    episode_count = validation.get("episode_count")
    if (
        isinstance(episode_count, bool)
        or not isinstance(episode_count, int)
        or episode_count != 28
    ):
        raise RuntimeError(
            f"completed seed summary has invalid validation episode count for seed {seed}: "
            f"{summary_path}"
        )
    metric_paths = (
        ("mean_policy_native_relative_incremental_ramp_impact",),
        ("mean_incremental_ramp_impact",),
        ("energy_cost_ratio",),
        ("status_quo_comparison", "policy_native_relative_mean_incremental_ramp_impact"),
        ("status_quo_comparison", "status_quo_native_relative_mean_incremental_ramp_impact"),
        ("status_quo_comparison", "policy_minus_status_quo_mean_incremental_ramp_impact"),
    )
    for path in metric_paths:
        value: Any = validation
        for field in path:
            if not isinstance(value, dict):
                break
            value = value.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise RuntimeError(
                f"completed seed summary is missing valid metric {'.'.join(path)} "
                f"for seed {seed}: {summary_path}"
            )
    return summary


def preflight() -> dict[str, Any]:
    """Exercise the status quo across every active window once."""
    geometry = _validate_campaign_geometry()
    counts = {"train": 0, "validation": 0, "steps": 0, "max_immediate_arrival_work": 0.0}
    for split in ("train", "validation"):
        for window_id in geometry[split]:
            env = make_four_market_env(EnvRequest(split=split, seed=SEEDS[0], window_id=window_id))
            try:
                current = env._current
                arrivals = (current.workload.service_arrivals + current.workload.batch_arrivals).sum(axis=1)
                counts["max_immediate_arrival_work"] = max(counts["max_immediate_arrival_work"], float(arrivals.max()))
                if arrivals.max() > current._capacity.sum() + 1e-12:
                    raise RuntimeError(f"{window_id} raw arrivals exceed hard capacity")
                env.reset(seed=SEEDS[0])
                while True:
                    _, _, terminated, _, info = env.step(env.evaluation_action("status_quo"))
                    counts["steps"] += 1
                    if terminated:
                        if info["service_unserved"] or info["batch_unfinished"] or info["certificate_violations"]:
                            raise RuntimeError(f"{window_id} status quo is infeasible")
                        break
                counts[split] += 1
            finally:
                env.close()
    return counts


def train_and_evaluate(seed: int) -> dict[str, Any]:
    """Run one independent optimizer seed or return its validated completed result."""
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    configure_single_thread_runtime()
    geometry = _validate_campaign_geometry()
    checkpoint = MODEL_ROOT / "ppo" / f"seed-{seed}"
    output = OUTPUT_ROOT / f"seed-{seed}"
    summary_path = output / "summary.json"
    if summary_path.exists():
        return _load_completed_seed_summary(
            seed, geometry, _campaign_training_geometry()
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "status.json", {
        "status": "running",
        "seed": seed,
        "requested_interactions": REQUESTED_TIMESTEPS,
    })
    started = time.perf_counter()
    try:
        training = run_training(
            factory=make_four_market_env, protocol=_protocol(), seed=seed,
            target_timesteps=REQUESTED_TIMESTEPS, output_dir=checkpoint,
            curve_path=output / "learning_curve.csv",
            progress_path=output / "progress.json",
        )
        elapsed = time.perf_counter() - started
        validation = evaluate_checkpoint(
            factory=make_four_market_env, checkpoint_dir=checkpoint, split="validation",
            seeds=[seed], windows=geometry["validation"],
        )
        if validation["episode_count"] != 28:
            raise RuntimeError("each seed must evaluate exactly the 28 February validation days")
        compact_validation = {
            key: value
            for key, value in validation.items()
            if not key.endswith("_episodes")
        }
        summary = {
            "seed": seed,
            "requested_interactions": REQUESTED_TIMESTEPS,
            "effective_interactions": training["effective_interactions"],
            "resumed_from_interactions": training["resumed_from_interactions"],
            "safe_boundary_interactions": training["safe_boundary_interactions"],
            "update_count": training["update_count"],
            "training_elapsed_seconds": elapsed,
            "training_interactions_per_second": training["effective_interactions"] / elapsed,
            "training_identity": training_identity(training),
            "validation_window_ids": geometry["validation"],
            "validation": compact_validation,
        }
        _write_json(summary_path, summary)
        _write_json(output / "status.json", {
            "status": "completed",
            "seed": seed,
            "requested_interactions": REQUESTED_TIMESTEPS,
            "effective_interactions": training["effective_interactions"],
            "resumed_from_interactions": training["resumed_from_interactions"],
        })
        return summary
    except Exception as error:
        _write_json(output / "status.json", {
            "status": "failed",
            "seed": seed,
            "requested_interactions": REQUESTED_TIMESTEPS,
            "error": f"{type(error).__name__}: {error}",
        })
        raise


def _run_seed_worker(seed: int) -> dict[str, Any]:
    # Spawned Windows workers inherit neither parent torch configuration nor state.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    return train_and_evaluate(seed)


def run_campaign(*, workers: int = DEFAULT_WORKERS) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers > len(SEEDS):
        raise ValueError(f"workers must not exceed seed count ({len(SEEDS)})")
    geometry = _validate_campaign_geometry()
    training_geometry = _campaign_training_geometry()
    completed: dict[int, dict[str, Any]] = {}
    unfinished: list[int] = []
    for seed in SEEDS:
        summary_path = OUTPUT_ROOT / f"seed-{seed}" / "summary.json"
        if summary_path.exists():
            completed[seed] = _load_completed_seed_summary(
                seed, geometry, training_geometry
            )
        else:
            unfinished.append(seed)
    if unfinished:
        with ProcessPoolExecutor(max_workers=min(workers, len(unfinished))) as executor:
            completed.update(zip(unfinished, executor.map(_run_seed_worker, unfinished)))
    return [completed[seed] for seed in SEEDS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--seed", type=int, choices=SEEDS, help="run exactly one optimizer seed")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="full-campaign worker count (default: 5)")
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return
    started = time.perf_counter()
    results = [train_and_evaluate(args.seed)] if args.seed else run_campaign(workers=args.workers)
    print(json.dumps({
        "runs": len(results),
        "workers": 1 if args.seed else args.workers,
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2))


if __name__ == "__main__":
    main()
