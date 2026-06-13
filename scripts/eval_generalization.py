"""Held-out generalization evaluation (peer-review M2).

Evaluates FROZEN policies — trained on cells a–d — on environments they have
never seen, without any retraining. Two axes:

  --axis cells   (default): held-out workloads — cells e–h replace a–d
                 (scenarios us_model_eh.yaml / global_model_eh.yaml; same
                 market series), testing whether the learned routing/deferral
                 strategy transfers across workloads, tier mixes, and per-cell
                 power characteristics.
  --axis market  : held-out market conditions — cells a–d with the NET-DEMAND
                 series shifted to real EIA-930 May-2024 (a genuine 2019->2024
                 duck-curve shift, corr ~0.74). Prices are the documented
                 synthetic series held fixed (real hourly LMP is unavailable
                 for the GA/SC non-ISO regions and EIA exposes no hourly price
                 route; see M7 provenance), so this isolates the net-demand
                 axis — the OBSERVED quantity at the core of the objective.
                 Tests the calendar-memorization concern on the demand signal.

For each config, evaluates the no-optimization Status Quo, Round Robin, and
the foresighted Trough-Slot heuristic (all parameter-free, so they transfer
trivially) plus all five review-campaign PPO seeds. Reports each seed's
savings vs the held-out Status Quo next to its on-training-cells savings.

Writes output/review_campaign/generalization_{axis}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from baselines import RoundRobinPolicy, StatusQuoPolicy, TroughSlotLookaheadPolicy  # noqa: E402
from evaluate import _make_env, compute_summary, run_episode  # noqa: E402

ALPHA = 0.015
SEEDS = [101, 102, 103, 104, 105]

CONFIGS = {
    "us_spatial": ("us_model", False),
    "us_batch": ("us_model", True),
    "global_spatial": ("global_model", False),
    "global_batch": ("global_model", True),
}


def scenario_for(stem: str, axis: str) -> Path:
    suffix = {"cells": "_eh", "market": "_yshift"}[axis]
    return ROOT / "env" / "scenarios" / f"{stem}{suffix}.yaml"


MODEL_ROOT = ROOT / "models" / "review"


def model_for(cfg_key: str, stem: str, batch: bool, seed: int) -> Path:
    name = f"ppo_{stem}" + ("_batch" if batch else "")
    return MODEL_ROOT / f"s{seed}" / f"{name}.zip"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["cells", "market"], default="cells")
    ap.add_argument("--model-dir", default="models/review",
                    help="campaign model root (e.g. models/review_ctx)")
    ap.add_argument("--campaign-dir", default="output/review_campaign",
                    help="campaign output root with stats.json")
    args = ap.parse_args()
    global MODEL_ROOT
    MODEL_ROOT = ROOT / args.model_dir

    # On-training-cells reference (from the multi-seed campaign)
    train_stats = {}
    stats_path = ROOT / args.campaign_dir / "stats.json"
    if stats_path.exists():
        st = json.load(open(stats_path, encoding="utf-8"))
        train_stats = {k: v["per_algo"]["ppo"]["mean_savings_vs_status_quo_pct"]
                       for k, v in st.items() if "ppo" in v.get("per_algo", {})}

    out: dict = {}
    for cfg_key, (stem, batch) in CONFIGS.items():
        scen = scenario_for(stem, args.axis)
        if not scen.exists():
            print(f"[skip] {cfg_key}: {scen.name} not found")
            continue
        print(f"===== {cfg_key} on {scen.name} =====", flush=True)

        def fresh():
            return _make_env(scen, batch_enabled=batch, peak_penalty_weight=ALPHA)

        entry: dict = {"baselines": {}, "ppo_seeds": {}}
        for cls in (StatusQuoPolicy, RoundRobinPolicy, TroughSlotLookaheadPolicy):
            pol = cls()
            _, h = run_episode(fresh(), pol.predict, is_sb3=False)
            s = compute_summary(h, batch_enabled=batch)
            entry["baselines"][pol.name] = s
            print(f"  {pol.name}: {s['total_cost']:,.0f}", flush=True)
        sq = entry["baselines"]["Status Quo (local, no deferral)"]["total_cost"]

        for seed in SEEDS:
            mp = model_for(cfg_key, stem, batch, seed)
            if not mp.exists():
                print(f"  [missing] {mp}")
                continue
            model = PPO.load(str(mp))
            _, h = run_episode(fresh(), model.predict, is_sb3=True)
            s = compute_summary(h, batch_enabled=batch)
            s["savings_vs_status_quo_pct"] = 100 * (sq - s["total_cost"]) / sq
            entry["ppo_seeds"][str(seed)] = s
            print(f"  PPO s{seed}: {s['total_cost']:,.0f} "
                  f"({s['savings_vs_status_quo_pct']:+.2f}% vs held-out SQ)", flush=True)

        sav = np.array([v["savings_vs_status_quo_pct"]
                        for v in entry["ppo_seeds"].values()])
        if len(sav):
            entry["summary"] = {
                "status_quo_cost": sq,
                "ppo_mean_savings_pct": float(sav.mean()),
                "ppo_std_savings_pct": float(sav.std(ddof=1)) if len(sav) > 1 else 0.0,
                "ppo_worst_seed_savings_pct": float(sav.min()),
                "on_training_cells_mean_savings_pct": train_stats.get(cfg_key),
            }
            print(f"  ==> held-out savings {sav.mean():+.2f}% ± {sav.std(ddof=1):.2f} "
                  f"(worst {sav.min():+.2f}%) | on-training: "
                  f"{train_stats.get(cfg_key, float('nan')):+.2f}%", flush=True)
        out[cfg_key] = entry

    path = ROOT / args.campaign_dir / f"generalization_{args.axis}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
