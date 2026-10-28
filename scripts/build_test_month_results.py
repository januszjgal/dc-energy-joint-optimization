"""Collect the test-month numbers quoted in Chapter 6 of latex/thesis_draft.tex.

Reads the ten-seed final campaign under ``output/four_market_joint_v3/campaign/``
(its per-seed ``summary.json`` and ``aggregation.json``), replays each final
policy and the no-flexibility baseline on the May 2026 test month, recomputes
J_e from the raw panel and site-power trajectory, and computes the same
per-market and behavior measures as the cross-validation figures. The policies
are fixed and act deterministically, so the replay reproduces the campaign's
scores; nothing here is chosen from the test month. Results go to
``docs/figures/ppo_test/test_month.json``.

    python scripts/build_test_month_results.py
"""

from __future__ import annotations

import json
import pickle
import sys
from functools import partial
from pathlib import Path
from statistics import fmean

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from energy_model_v3.four_market_v2 import ARTIFACT_NAMESPACE, MARKETS, factory_root, make_four_market_env  # noqa: E402
from energy_model_v3.splits import FINAL_SPEC, TEST_MONTHS  # noqa: E402
from ramp_rl.contract import EnvRequest  # noqa: E402
from ramp_rl.evaluation import _run_episode  # noqa: E402
from scripts.build_ppo_cv_figures import (  # noqa: E402
    MonthInputs, behavior_metrics, month_arrays, no_flex_month, percentage_reductions,
)

CAMPAIGN = ROOT / "output" / ARTIFACT_NAMESPACE / "campaign"
MODELS = ROOT / "models" / ARTIFACT_NAMESPACE / "campaign" / "ppo"
OUT = ROOT / "docs" / "figures" / "ppo_test" / "test_month.json"
SEEDS = tuple(range(4101, 4111))
MONTH = TEST_MONTHS[0]


def peaks_mw() -> dict[str, dict[str, float]]:
    """Mean monthly peak of each market: original grid, no flexibility, and the policies."""
    rows = [
        json.loads((CAMPAIGN / f"seed-{seed}" / "summary.json").read_text(encoding="utf-8"))
        ["test"]["status_quo_comparison"]["per_market"]
        for seed in SEEDS
    ]
    return {
        market: {
            key: fmean(row[market][key] for row in rows)
            for key in ("native_peak_mw", "status_quo_peak_mw", "policy_peak_mw")
        }
        for market in MARKETS
    }


def arrival_range(split: str, month: str) -> dict[str, float]:
    """How far the sites' service and batch arrivals lie from the training averages.

    The arrivals are rescaled with the statistics that final policy 1 learned in
    training. Arrivals do not depend on the schedule, so this compares the test
    fleet's inputs with a training month's on equal terms.
    """
    with (MODELS / f"seed-{SEEDS[0]}" / "vecnormalize.pkl").open("rb") as handle:
        stats = pickle.load(handle).obs_rms
    env = make_four_market_env(EnvRequest(split=split, seed=0, window_id=month), spec=FINAL_SPEC)
    try:
        observation, _ = env.reset(seed=0)
        rows = [observation]
        while True:
            observation, _, done, _, _ = env.step(env.evaluation_action("status_quo"))
            if done:
                break
            rows.append(observation)
        schema = env._current.observation_schema
    finally:
        env.close()
    index = [i for i, name in enumerate(schema) if name.startswith("site-") and "arrival" in name]
    z = (np.asarray(rows)[:, index] - stats.mean[index]) / np.sqrt(stats.var[index] + 1e-8)
    return {"mean_abs_z": float(np.abs(z).mean()), "share_beyond_3": float((np.abs(z) > 3).mean())}


def main() -> None:
    aggregation = json.loads((CAMPAIGN / "aggregation.json").read_text(encoding="utf-8"))
    inputs = MonthInputs("", MONTH, spec=FINAL_SPEC, split="test")
    factory = json.loads((factory_root(FINAL_SPEC) / "factory.json").read_text(encoding="utf-8"))
    scales = json.loads((ROOT / factory["frozen_stats_path"]).read_text(encoding="utf-8"))["gross_q95_mw"]
    nf_J, nf_infos = no_flex_month(FINAL_SPEC, "test", MONTH)
    baseline = month_arrays(nf_infos, inputs.site_ids)
    seeds = {}
    for seed in SEEDS:
        summary = json.loads((CAMPAIGN / f"seed-{seed}" / "summary.json").read_text(encoding="utf-8"))
        comparison = summary["test"]["status_quo_comparison"]
        checkpoint = MODELS / f"seed-{seed}"
        infos, _ = _run_episode(
            factory=partial(make_four_market_env, spec=FINAL_SPEC), split="test", seed=seed,
            window_id=MONTH, model=PPO.load(checkpoint / "model.zip", device="cpu"),
            normalization_path=checkpoint / "vecnormalize.pkl",
        )
        run = month_arrays(infos, inputs.site_ids)
        row = {
            "regional_metrics": comparison["per_market"],
            "step_count": summary["test"]["step_count"],
            "components": comparison["components"],
        }
        seeds[str(seed)] = {
            "policy_J": comparison["policy_J"],
            "status_quo_J": comparison["status_quo_J"],
            "improvement_J": comparison["improvement_J"],
            "ramp_weighted_improvement": comparison["components"]["ramp"]["weighted_improvement"],
            "peak_weighted_improvement": comparison["components"]["net_load_peak"]["weighted_improvement"],
            "recomputed_J_error": abs(inputs.recomputed_J(run["power"]) - comparison["policy_J"]),
            "training_minutes": summary["training_elapsed_seconds"] / 60.0,
            "final_training_return": float(
                pd.read_csv(CAMPAIGN / f"seed-{seed}" / "learning_curve.csv")
                .query("episode_count > 0")["mean_completed_episode_return"].tail(10).mean()
            ),
            "behavior": {
                **behavior_metrics(run, baseline, inputs),
                "no_flexibility_percentage_reductions": percentage_reductions(row, scales),
            },
        }
    values = list(seeds.values())

    def mean_of(key):
        return fmean(v[key] for v in values)

    result = {
        "month": MONTH,
        "no_flex_J": nf_J,
        "mean_improvement_J": mean_of("improvement_J"),
        "improvement_J_range": [min(v["improvement_J"] for v in values), max(v["improvement_J"] for v in values)],
        "seeds_improving": sum(v["improvement_J"] > 0 for v in values),
        "mean_ramp_weighted_improvement": mean_of("ramp_weighted_improvement"),
        "mean_peak_weighted_improvement": mean_of("peak_weighted_improvement"),
        "max_abs_recomputed_J_error": max(v["recomputed_J_error"] for v in values),
        "wilcoxon_p_value": aggregation["paired_policy_minus_status_quo"]["wilcoxon_signed_rank"]["p_value"],
        "safety_totals": aggregation["safety_totals"],
        "per_market_mean": {
            market: {
                field: fmean(v["behavior"]["per_market"][market][field] for v in values)
                for field in ("squared_ramp_cut_pct", "peak_change_mw")
            }
            for market in MARKETS
        },
        "per_market_range": {
            market: {
                field: [min(v["behavior"]["per_market"][market][field] for v in values),
                        max(v["behavior"]["per_market"][market][field] for v in values)]
                for field in ("squared_ramp_cut_pct", "peak_change_mw")
            }
            for market in MARKETS
        },
        "behavior_range": {
            field: [min(v["behavior"][field] for v in values), max(v["behavior"][field] for v in values)]
            for field in ("service_moved_pct", "mean_batch_wait_hours", "decoder_adjustment_pct")
        },
        "percentage_reductions_mean": {
            field: fmean(v["behavior"]["no_flexibility_percentage_reductions"][field] for v in values)
            for field in values[0]["behavior"]["no_flexibility_percentage_reductions"]
        },
        "peak_rises_count": {
            market: sum(v["behavior"]["per_market"][market]["peak_change_mw"] > 0 for v in values)
            for market in MARKETS
        },
        "peaks_mw": peaks_mw(),
        "site_arrival_range": {
            "test": arrival_range("test", MONTH),
            "training_month_2025_05": arrival_range("train", "2025-05"),
        },
        "seeds": seeds,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if k != "seeds"}, indent=2))


if __name__ == "__main__":
    main()
