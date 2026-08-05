"""Demand-charge audit: what the operator's tariff bill would be under each policy.

The shaped peak penalty Φ = α·g²·d in the reward is a grid-stress shadow price,
not a tariff — α is calibrated to a target share of total cost, and Φ is a
per-step quadratic, whereas a real demand charge is a *max* over the billing
period, linear in kW, and indifferent to grid net demand. This script reports
the tariff quantity for every policy, computed from the grid-draw traces, and
bounds how much of it is reducible at all.

Two headline quantities:

  billed peak = Σᵢ maxₜ gᵢ,ₜ  — billing is per meter, so the operator pays on the
  sum of per-SITE maxima, not the fleet coincident peak.

  headroom — the floor on that quantity. Since the objective depends only on the
  peak vector, and site i can serve at most uᵢ(Dᵢ) = min(κᵢ, (Dᵢ/Rᵢ − idleᵢ)/slopeᵢ),
  feasibility binds only at the busiest step, so the clairvoyant LP collapses to a
  greedy lowest-slope·R fill. Underneath sits Σᵢ idleᵢ·Rᵢ, which no policy can
  touch because sites never power off in this model.

CAVEAT on the floors: they are conditional on SERVING the binding step's load.
A policy that leaves work unserved prints a lower billed peak while paying for it
in backlog and deadline penalties — Cheapest Price First reaches 261.7 MW, below
every floor, at 3-4x the total cost. Compare billed peak only among policies that
serve all demand (check `service_served_total` / `service_demand_total` ≈ 1).

Usage:
    python scripts/analyze_demand_charge.py [--rate 15.0] [--models-dir models/review_ctx/s101]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from baselines import ALL_BASELINES
from env.multi_dc_env import REFERENCE_DEMAND_CHARGE_RATE
from evaluate import _make_env, compute_summary, run_episode

ALPHA = 0.015
SCENARIOS = [
    ("us_model", ROOT / "env" / "scenarios" / "us_model.yaml"),
    ("global_model", ROOT / "env" / "scenarios" / "global_model.yaml"),
]


def site_params(env):
    idle = np.array([(s.power_model or env.power_model).idle_power for s in env.sites])
    slope = np.array([(s.power_model or env.power_model).slope for s in env.sites])
    R = np.array([s.rated_power_mw for s in env.sites])
    kappa = np.array([s.capacity for s in env.sites])
    return idle, slope, R, kappa


def min_billed_peak(idle, slope, R, kappa, load) -> float:
    """Greedy lowest-slope·R fill: min Σᵢ (idleᵢ + slopeᵢ·uᵢ)·Rᵢ s.t. Σuᵢ = load."""
    alloc = np.zeros(len(R))
    rem = float(load)
    for i in np.argsort(slope * R):
        take = min(kappa[i], rem)
        alloc[i] = take
        rem -= take
        if rem <= 1e-12:
            break
    return float(((idle + slope * alloc) * R).sum())


def analyze(scen_name: str, scen_path: Path, batch: bool, rate: float,
            models_dir: Path | None) -> dict:
    mode = "batch" if batch else "spatial"
    print(f"\n{'='*74}\n{scen_name} [{mode}]  demand charge @ ${rate}/kW-month\n{'='*74}")

    def fresh():
        return _make_env(scen_path, batch_enabled=batch, peak_penalty_weight=ALPHA)

    policies: list[tuple[str, object, bool]] = [
        (cls().name, cls().predict, False) for cls in ALL_BASELINES
    ]
    if models_dir is not None:
        stem = f"ppo_{scen_name}" + ("_batch" if batch else "")
        mp = models_dir / f"{stem}.zip"
        if mp.exists():
            from stable_baselines3 import PPO
            model = PPO.load(mp)
            policies.insert(0, ("PPO", model.predict, True))
        else:
            print(f"  (no model at {mp}; baselines only)")

    rows = {}
    for name, predict, is_sb3 in policies:
        env = fresh()
        try:
            _, hist = run_episode(env, predict, is_sb3=is_sb3)
        except ValueError as e:  # obs-shape mismatch against a stale model
            print(f"  SKIP {name}: {e}")
            continue
        s = compute_summary(hist, batch_enabled=batch)
        served_frac = (s["service_served_total"] / s["service_demand_total"]
                       if s["service_demand_total"] > 0 else 0.0)
        rows[name] = {
            # Below ~1.0 the billed peak is bought with unserved work, not
            # shaping, and is not comparable to the floors.
            "served_fraction": served_frac,
            "total_cost": s["total_cost"],
            "energy_cost": s["total_energy_cost"],
            "peak_penalty": s["total_peak_penalty"],
            "billed_peak_sum_mw": s["billed_peak_sum_mw"],
            "demand_charge": s["billed_peak_sum_mw"] * 1000.0 * rate,
            "per_dc_billed_peak_mw": s["per_dc_billed_peak_mw"],
            "fleet_coincident_peak_mw": s["peak_grid_mw"],
            "load_factor": s["load_factor"],
        }

    # ---- headroom bounds ---------------------------------------------------
    env = fresh()
    idle, slope, R, kappa = site_params(env)
    _, hist = run_episode(env, next(
        c().predict for c in ALL_BASELINES if c.name.startswith("Status Quo")
    ), is_sb3=False)
    tot = np.array([h.get("total_demand", 0.0) for h in hist])
    if batch:
        bat = np.array([
            sum(dc.get("batch_served", 0.0) for dc in h["per_dc"]) for h in hist
        ])
        svc = np.array([
            sum(dc.get("service_served", 0.0) for dc in h["per_dc"]) for h in hist
        ])
        load = svc + bat
    else:
        load = np.array([
            sum(dc.get("served", 0.0) for dc in h["per_dc"]) for h in hist
        ])
        bat = np.zeros_like(load)
    worst = int(np.argmax(load))
    idle_floor = float((idle * R).sum())
    bounds = {
        "spatial_only": min_billed_peak(idle, slope, R, kappa, load.max()),
        "defer_batch_out_of_worst_step": min_billed_peak(
            idle, slope, R, kappa, max(load.max() - bat[worst], 0.0)),
        "perfect_flatten_to_mean": min_billed_peak(
            idle, slope, R, kappa, float(load.mean())),
        "irreducible_idle_floor": idle_floor,
    }

    sq = rows.get("Status Quo (local, no deferral)", {})
    sq_peak = sq.get("billed_peak_sum_mw", float("nan"))
    print(f"\n  {'policy':34s} {'billed pk':>10s} {'charge':>10s} {'vs SQ':>8s} "
          f"{'energy':>10s} {'served':>7s}")
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]["billed_peak_sum_mw"]):
        delta = (r["billed_peak_sum_mw"] - sq_peak) / sq_peak * 100 if sq_peak else 0.0
        flag = "" if r["served_fraction"] > 0.999 else "  <- unserved work"
        print(f"  {name:34s} {r['billed_peak_sum_mw']:9.1f} MW "
              f"${r['demand_charge']/1e6:8.3f}M {delta:7.1f}% "
              f"${r['energy_cost']/1e6:8.3f}M {r['served_fraction']:7.4f}{flag}")

    print(f"\n  floors on the billed peak:")
    for k, v in bounds.items():
        delta = (v - sq_peak) / sq_peak * 100 if sq_peak else 0.0
        print(f"    {k:34s} {v:9.1f} MW ${v*1000*rate/1e6:8.3f}M {delta:7.1f}%")
    best = min(bounds[k] for k in
               ("spatial_only", "defer_batch_out_of_worst_step", "perfect_flatten_to_mean"))
    print(f"    {'-> best achievable':34s} {best:9.1f} MW "
          f"${best*1000*rate/1e6:8.3f}M  (prize ${(sq_peak-best)*1000*rate/1e6:.3f} M)")
    if "PPO" in rows:
        ppo_peak = rows["PPO"]["billed_peak_sum_mw"]
        captured = (sq_peak - ppo_peak) / (sq_peak - best) * 100 if sq_peak > best else 0.0
        print(f"    PPO captures {captured:.0f}% of that prize with the term NOT in its reward")

    return {"policies": rows, "bounds": bounds, "rate": rate,
            "worst_step": worst, "deferrable_at_worst_step": float(bat[worst]),
            "load_at_worst_step": float(load[worst])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=REFERENCE_DEMAND_CHARGE_RATE,
                    help="demand charge in $/kW-month")
    ap.add_argument("--models-dir", type=Path,
                    default=ROOT / "models" / "review_ctx" / "s101",
                    help="directory holding ppo_<scenario>[_batch].zip")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "output" / "demand_charge_analysis.json")
    args = ap.parse_args()

    out = {}
    for scen_name, scen_path in SCENARIOS:
        for batch in (False, True):
            key = f"{scen_name}_{'batch' if batch else 'spatial'}"
            out[key] = analyze(scen_name, scen_path, batch, args.rate, args.models_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
