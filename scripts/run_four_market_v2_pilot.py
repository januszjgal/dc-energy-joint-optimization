"""Short PPO pilot: a few seeds with a chosen budget, evaluated on May.

Unlike the locked ten-seed campaign, the seeds and interaction budget are
arguments. Results go to ``output/four_market_v2/pilot/<tag>/`` and checkpoints
to ``models/four_market_v2/pilot/<tag>/``, so campaign directories are never
touched. Every saved milestone is evaluated too, which shows whether the May
improvement grows with training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from energy_model_v3.four_market_v2 import make_four_market_env  # noqa: E402
from ramp_rl.evaluation import evaluate_checkpoint  # noqa: E402
from ramp_rl.runner import configure_single_thread_runtime, run_training  # noqa: E402
from scripts.run_four_market_v2_campaign import (  # noqa: E402
    _protocol,
    _validate_campaign_geometry,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compact(validation: dict[str, Any]) -> dict[str, Any]:
    comparison = validation["status_quo_comparison"]
    return {
        "step_count": validation["step_count"],
        "policy_J": comparison["policy_J"],
        "status_quo_J": comparison["status_quo_J"],
        "improvement_J": comparison["improvement_J"],
        "markets_better_count": comparison["markets_better_count"],
        "per_market_improvement_mean_per_hour": {
            market: row["improvement"] for market, row in comparison["per_market"].items()
        },
        "abs_adjusted_ramp_h1_p95": validation["abs_adjusted_ramp_h1_fraction_s_per_hour_p95"],
        "service_unserved": validation["service_unserved"],
        "batch_unfinished": validation["batch_unfinished"],
        "decoder_adjustment_rate": validation["semantic_adjustment"]["adjustment_rate"],
        "behavior_audit": validation["behavior_audit"],
    }


def run_seed(seed: int, *, timesteps: int, tag: str) -> dict[str, Any]:
    configure_single_thread_runtime()
    windows = _validate_campaign_geometry()["validation"]
    output = ROOT / "output" / "four_market_v2" / "pilot" / tag / f"seed-{seed}"
    checkpoint = ROOT / "models" / "four_market_v2" / "pilot" / tag / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    training = run_training(
        factory=make_four_market_env, protocol=_protocol(), seed=seed,
        target_timesteps=timesteps, output_dir=checkpoint,
        curve_path=output / "learning_curve.csv", progress_path=output / "progress.json",
    )
    elapsed = time.perf_counter() - started

    def evaluate(directory: Path) -> dict[str, Any]:
        return _compact(evaluate_checkpoint(
            factory=make_four_market_env, checkpoint_dir=directory,
            split="validation", seeds=[seed], windows=windows,
        ))

    evaluations: dict[str, Any] = {}
    index_path = checkpoint / "milestones" / "index.json"
    if index_path.is_file():
        milestones = json.loads(index_path.read_text(encoding="utf-8"))["milestones"]
        for record in sorted(milestones.values(), key=lambda item: item["actual_interactions"]):
            if record["actual_interactions"] < training["effective_interactions"]:
                evaluations[str(record["actual_interactions"])] = evaluate(ROOT / record["path"])
    evaluations[str(training["effective_interactions"])] = evaluate(checkpoint)
    summary = {
        "seed": seed,
        "tag": tag,
        "effective_interactions": training["effective_interactions"],
        "training_elapsed_seconds": elapsed,
        "training_interactions_per_second": training["effective_interactions"] / elapsed,
        "validation_window_ids": windows,
        "validation_by_interactions": evaluations,
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[4101, 4102, 4103])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--tag", default="one-hour-1m")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    workers = args.workers or len(args.seeds)
    started = time.perf_counter()
    job = partial(run_seed, timesteps=args.timesteps, tag=args.tag)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(job, args.seeds))
    final = {
        str(result["seed"]): result["validation_by_interactions"][str(result["effective_interactions"])]
        for result in results
    }
    _write_json(
        ROOT / "output" / "four_market_v2" / "pilot" / args.tag / "pilot.json",
        {"seeds": args.seeds, "timesteps": args.timesteps,
         "elapsed_seconds": time.perf_counter() - started, "final": final},
    )
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
