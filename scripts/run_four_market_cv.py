"""Four-fold cross-validation of the PPO scheduler over the 2025 months.

Each fold trains fresh policies (one per seed) on its nine training months and
evaluates every saved milestone on each of its three held-out months, one
continuous episode per month, against the no-flexibility baseline on the same
hours. Every 2025 month is therefore validated exactly once per seed.

Results go to ``output/four_market_joint_v3/cv/<tag>/<fold>/seed-<seed>/`` and
checkpoints to ``models/four_market_joint_v3/cv/<tag>/<fold>/seed-<seed>/``.
``cv.json`` in the tag directory collects the final improvement of every
(fold, seed, month) and its averages. The test month is never touched here.

    python scripts/run_four_market_cv.py --seeds 4101 4102 4103 --timesteps 1000000 --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from energy_model_v3.four_market_v2 import ARTIFACT_NAMESPACE, make_four_market_env  # noqa: E402
from energy_model_v3.splits import CV_FOLDS, spec_months  # noqa: E402
from ramp_rl.evaluation import evaluate_checkpoint  # noqa: E402
from ramp_rl.runner import configure_single_thread_runtime, run_training  # noqa: E402

PROTOCOL_PATH = ROOT / "env" / "protocols" / "four_market_v2.yaml"
DEFAULT_TAG = "cv-raw-obs-1m"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _validate_fold(spec: str) -> list[str]:
    """Check the fold factory and return its validation months."""
    from scripts.build_four_market_v2_factory import validate

    validate(spec)
    windows = list(spec_months(spec)["validation"])
    if len(windows) != 3:
        raise RuntimeError(f"{spec} must hold out exactly three months")
    return windows


def _compact(validation: dict[str, Any]) -> dict[str, Any]:
    comparison = validation["status_quo_comparison"]
    return {
        "step_count": validation["step_count"],
        "objective": validation["objective"],
        "policy_J": comparison["policy_J"],
        "status_quo_J": comparison["status_quo_J"],
        "improvement_J": comparison["improvement_J"],
        "markets_better_count": comparison["markets_better_count"],
        "components": comparison["components"],
        "per_market_joint_improvement": {
            market: row["improvement"] for market, row in comparison["per_market"].items()
        },
        "per_market_peak_reduction_mw": {
            market: row["peak_reduction_mw"] for market, row in comparison["per_market"].items()
        },
        "regional_metrics": comparison["per_market"],
        "abs_adjusted_ramp_h1_p95": validation["abs_adjusted_ramp_h1_fraction_s_per_hour_p95"],
        "service_unserved": validation["service_unserved"],
        "batch_unfinished": validation["batch_unfinished"],
        "batch_expired": validation["batch_expired"],
        "terminal_work": validation["terminal_work"],
        "certificate_violations": validation["certificate_violations"],
        "decoder_adjustment_rate": validation["semantic_adjustment"]["adjustment_rate"],
        "behavior_audit": validation["behavior_audit"],
    }


def run_job(
    fold: str, seed: int, *, timesteps: int, tag: str, ramp_weight: float | None = None,
    raw_observation: bool = True,
) -> dict[str, Any]:
    """Train one seed on one fold and evaluate it on the fold's held-out months."""
    configure_single_thread_runtime()
    spec = f"cv-{fold}"
    windows = _validate_fold(spec)
    factory = partial(
        make_four_market_env, spec=spec, ramp_weight=ramp_weight, raw_observation=raw_observation,
    )
    output = ROOT / "output" / ARTIFACT_NAMESPACE / "cv" / tag / fold / f"seed-{seed}"
    checkpoint = ROOT / "models" / ARTIFACT_NAMESPACE / "cv" / tag / fold / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    previous = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file() else None
    )
    reused = all(
        (checkpoint / name).is_file()
        for name in ("training_summary.json", "model.zip", "vecnormalize.pkl")
    )
    started = time.perf_counter()
    training = run_training(
        factory=factory, protocol=_protocol(), seed=seed,
        target_timesteps=timesteps, output_dir=checkpoint,
        curve_path=output / "learning_curve.csv", progress_path=output / "progress.json",
    )
    invocation_elapsed = time.perf_counter() - started
    elapsed = invocation_elapsed
    if reused:
        elapsed = (
            previous["training_elapsed_seconds"]
            if previous is not None and previous.get("training_identity") == training
            else None
        )

    def evaluate(directory: Path) -> dict[str, Any]:
        return {
            window: _compact(evaluate_checkpoint(
                factory=factory, checkpoint_dir=directory,
                split="validation", seeds=[seed], windows=[window],
            ))
            for window in windows
        }

    evaluations: dict[str, Any] = {}
    index_path = checkpoint / "milestones" / "index.json"
    if index_path.is_file():
        milestones = json.loads(index_path.read_text(encoding="utf-8"))["milestones"]
        for record in sorted(milestones.values(), key=lambda item: item["actual_interactions"]):
            if record["actual_interactions"] < training["effective_interactions"]:
                evaluations[str(record["actual_interactions"])] = evaluate(ROOT / record["path"])
    evaluations[str(training["effective_interactions"])] = evaluate(checkpoint)
    summary = {
        "fold": fold,
        "spec": spec,
        "seed": seed,
        "tag": tag,
        "raw_observation": raw_observation,
        "training_months": list(spec_months(spec)["train"]),
        "validation_window_ids": windows,
        "training_identity": training,
        "training_reused": reused,
        "training_invocation_elapsed_seconds": invocation_elapsed,
        "effective_interactions": training["effective_interactions"],
        "training_elapsed_seconds": elapsed,
        "training_interactions_per_second": (
            training["effective_interactions"] / elapsed if elapsed is not None else None
        ),
        "validation_by_interactions": evaluations,
    }
    _write_json(summary_path, summary)
    return summary


def _run_job_worker(job: tuple[str, int], **kwargs: Any) -> dict[str, Any]:
    # Spawned Windows workers inherit neither parent torch configuration nor state.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    fold, seed = job
    return run_job(fold, seed, **kwargs)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Final improvement per (fold, seed, month), and its averages.

    The cross-validation estimate for one seed is the mean over the twelve
    held-out months; the spread across seeds measures training randomness.
    """
    by_month: dict[str, dict[str, float]] = {}
    for result in results:
        final = result["validation_by_interactions"][str(result["effective_interactions"])]
        for month, row in final.items():
            by_month.setdefault(month, {})[str(result["seed"])] = row["improvement_J"]
    seeds = sorted({str(result["seed"]) for result in results})
    complete = all(len(row) == len(seeds) for row in by_month.values())
    per_seed = {
        seed: statistics.fmean(row[seed] for row in by_month.values())
        for seed in seeds
    } if complete else {}
    return {
        "improvement_J_by_month": dict(sorted(by_month.items())),
        "mean_improvement_J_by_seed": per_seed,
        "mean_improvement_J": statistics.fmean(per_seed.values()) if per_seed else None,
        "seed_std_improvement_J": statistics.stdev(per_seed.values()) if len(per_seed) > 1 else None,
        "months": len(by_month),
        "seeds": seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", choices=tuple(CV_FOLDS), default=list(CV_FOLDS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[4101, 4102, 4103])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--ramp-weight", type=float, choices=(0.0, 0.5, 1.0), default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--scaled-observation", action="store_true",
                        help="use the environment's fixed observation scaling of the earlier "
                             "cv-trace-clock-1m runs instead of raw observations")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in {"", ".", ".."}:
        parser.error("--tag must be a single directory name")
    if args.timesteps <= 0 or (args.workers is not None and args.workers <= 0):
        parser.error("timesteps and workers must be positive")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be unique")
    jobs = [(fold, seed) for fold in args.folds for seed in args.seeds]
    workers = args.workers or min(len(jobs), 6)
    started = time.perf_counter()
    job = partial(
        _run_job_worker, timesteps=args.timesteps, tag=args.tag, ramp_weight=args.ramp_weight,
        raw_observation=not args.scaled_observation,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(job, jobs))
    summary = {
        "folds": args.folds, "seeds": args.seeds, "timesteps": args.timesteps,
        "elapsed_seconds": time.perf_counter() - started, **aggregate(results),
    }
    _write_json(ROOT / "output" / ARTIFACT_NAMESPACE / "cv" / args.tag / "cv.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "improvement_J_by_month"}, indent=2))


if __name__ == "__main__":
    main()
