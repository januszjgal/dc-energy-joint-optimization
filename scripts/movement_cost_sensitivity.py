"""Movement-cost sensitivity (peer-review M6, post-hoc scoring sweep).

The spatial lever assumes free, instant, latency-blind fungibility: service
demand is re-routed across DCs every 5 min with no migration/egress cost.
Reviewer M6 asks how much of the Global savings survives once moving work
costs something (data egress, residency). SustainCluster [27] models per-GB
transfer pricing; we adopt the same idea as a per-unit-CPU movement cost.

This is a post-hoc *scoring* sweep on the canonical context-aware (ctx) PPO
seeds — no retraining. For each policy we measure total inter-site movement
  moved = sum_t sum_i max(0, assigned_{t,i} - local_demand_{t,i})
(net inflow to over-served sites = total work that left its home cell), then
add w * moved to its episode cost for a sweep of movement prices w. The
no-optimization Status Quo serves locally (moved = 0), so its cost is
invariant — the clean reference. Because the policies never saw the movement
cost during training, this is a CONSERVATIVE lower bound on surviving
savings: a movement-aware policy would route less and recover some advantage.
We report, per config, the break-even price w* at which PPO's mean advantage
over the Status Quo reaches zero, in absolute $/unit and as a multiple of the
mean energy cost of serving one unit.

Writes output/review_campaign_ctx/movement_cost_sensitivity.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from baselines import StatusQuoPolicy  # noqa: E402
from evaluate import _make_env, run_episode  # noqa: E402

ALPHA = 0.015
SEEDS = [101, 102, 103, 104, 105]
MODEL_DIR = ROOT / "models" / "review_ctx"
# Movement prices to sweep ($/unit-CPU-NCU moved between cells)
WEIGHTS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
CONFIGS = [
    ("global_spatial", "global_model", False),
    ("global_batch", "global_model", True),
    ("us_spatial", "us_model", False),
]


def episode_metrics(history) -> tuple[float, float, float]:
    """Return (base_cost, moved_volume, served_volume)."""
    base = sum(h["total_cost"] for h in history)
    moved = 0.0
    served = 0.0
    for t, h in enumerate(history):
        for dc in h["per_dc"]:
            served += dc["served"]
        moved += h.get("_moved", 0.0)
    return base, moved, served


def run_with_movement(env, predict, is_sb3):
    """run_episode but also tag each step with inter-site movement volume."""
    _, hist = run_episode(env, predict, is_sb3=is_sb3)
    batch = env.batch_enabled
    for t, h in enumerate(hist):
        mv = 0.0
        for dc in h["per_dc"]:
            site = next(s for s in env.sites if s.name == dc["name"])
            # Non-deferrable service routed away from its home cell. (Spatial
            # mode logs 'assigned' over total local demand; batch mode logs
            # 'service_assigned' over the home service demand.) Batch-drain
            # routing adds further movement we do NOT charge here, making the
            # surviving-savings bound even more conservative for batch configs.
            if batch:
                assigned = dc["service_assigned"]
                local = site.get_service_demand(t)
            else:
                assigned = dc["assigned"]
                local = site.get_local_demand(t)
            mv += max(0.0, assigned - local)
        h["_moved"] = mv
    return hist


def main() -> None:
    out: dict = {}
    for cfg_key, stem, batch in CONFIGS:
        def fresh():
            return _make_env(ROOT / "env" / "scenarios" / f"{stem}.yaml",
                             batch_enabled=batch, peak_penalty_weight=ALPHA)

        sq_hist = run_with_movement(fresh(), StatusQuoPolicy().predict, False)
        sq_cost, sq_moved, sq_served = episode_metrics(sq_hist)

        ppo_costs, ppo_moved = [], []
        for seed in SEEDS:
            mp = MODEL_DIR / f"s{seed}" / f"ppo_{stem}{'_batch' if batch else ''}.zip"
            if not mp.exists():
                continue
            h = run_with_movement(fresh(), PPO.load(str(mp)).predict, True)
            c, m, _ = episode_metrics(h)
            ppo_costs.append(c)
            ppo_moved.append(m)
        ppo_costs = np.array(ppo_costs)
        ppo_moved = np.array(ppo_moved)

        unit_energy = sum(sum(dc["energy_cost"] for dc in h["per_dc"])
                          for h in sq_hist) / max(sq_served, 1e-9)

        curve = []
        for w in WEIGHTS:
            adj_ppo = ppo_costs + w * ppo_moved      # SQ moved = 0
            sav = 100 * (sq_cost - adj_ppo) / sq_cost
            curve.append({"w": w, "mean_savings_pct": float(sav.mean()),
                          "min_savings_pct": float(sav.min())})
        # break-even w*: mean PPO advantage = 0  ->  w* = (sq - mean_ppo_base)/mean_moved
        adv0 = sq_cost - ppo_costs.mean()
        w_star = float(adv0 / ppo_moved.mean()) if ppo_moved.mean() > 0 else float("inf")

        out[cfg_key] = {
            "status_quo_cost": sq_cost,
            "ppo_base_mean": float(ppo_costs.mean()),
            "ppo_mean_moved_units": float(ppo_moved.mean()),
            "mean_unit_energy_cost": float(unit_energy),
            "breakeven_w_usd_per_unit": w_star,
            "breakeven_as_mult_of_unit_energy": w_star / unit_energy if unit_energy else None,
            "sweep": curve,
        }
        print(f"=== {cfg_key} ===")
        print(f"  PPO moves {ppo_moved.mean():,.0f} units; break-even movement price "
              f"w* = ${w_star:,.1f}/unit "
              f"({w_star/unit_energy:.1f}x the ${unit_energy:.1f}/unit energy cost)")
        for c in curve:
            print(f"    w=${c['w']:>4}: PPO savings {c['mean_savings_pct']:+.2f}% "
                  f"(worst seed {c['min_savings_pct']:+.2f}%)")

    path = MODEL_DIR.parent.parent / "output" / "review_campaign_ctx" / \
        "movement_cost_sensitivity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
