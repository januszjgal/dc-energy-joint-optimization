"""Smoke test the demand-charge term (§3.4).

Checks the four properties the formulation depends on:

  1. Default-off regression — `demand_charge_rate=0.0` leaves the reward
     bit-identical to the pre-demand-charge environment, so every committed
     result stays reproducible, while the charge is still REPORTED.
  2. Exact telescoping — the per-step incremental charge
     `c·max(0, g_t − D_{t−1})` sums to `c·max_t g_t` per window, per site.
  3. Prorated windows — a partial trailing window is charged by its true
     length, not a full period, and prorated daily billing lower-bounds
     monthly billing (equality iff every day's peak is equal).
  4. Observability — enabling the term adds exactly one observation dim per
     DC (the running billed peak), without which the cost is not Markov.

Usage:
    python scripts/smoke_test_demand_charge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from baselines import StatusQuoPolicy
from env.multi_dc_env import STEPS_PER_MONTH
from evaluate import _make_env, compute_summary, run_episode

SCEN = ROOT / "env" / "scenarios" / "global_model.yaml"
ALPHA = 0.015
RATE = 15.0


def _run(**kw):
    env = _make_env(SCEN, batch_enabled=True, peak_penalty_weight=ALPHA, **kw)
    _, hist = run_episode(env, StatusQuoPolicy().predict, is_sb3=False)
    return env, hist


def main() -> int:
    failures = []

    # 1. default off ---------------------------------------------------------
    env, hist = _run()
    summary = compute_summary(hist, batch_enabled=True)
    print("[1] default OFF")
    print(f"    total_cost                ${summary['total_cost']:,.2f}")
    print(f"    demand charge IN reward   ${summary['total_demand_charge']:,.2f}")
    print(f"    demand charge REPORTED    ${summary['demand_charge_ref']/1e6:.3f} M "
          f"@ ${summary['demand_charge_ref_rate']}/kW-month")
    print(f"    billed peak (sum of per-site maxima) {summary['billed_peak_sum_mw']:.1f} MW")
    if env.demand_charge_enabled:
        failures.append("demand charge should default to disabled")
    if summary["total_demand_charge"] != 0.0:
        failures.append("disabled term still charged the reward")
    if summary["demand_charge_ref"] <= 0.0:
        failures.append("reference charge not reported when term is disabled")

    # 2/3. telescoping + proration ------------------------------------------
    for period, label in [(8917, "single window"), (288, "daily")]:
        _, hist = _run(demand_charge_rate=RATE, demand_charge_period_steps=period)
        paid = sum(h["total_demand_charge"] for h in hist)
        g = np.array([[dc["grid_mw"] for dc in h["per_dc"]] for h in hist])
        expected = sum(
            RATE * (len(w) / STEPS_PER_MONTH) * 1000.0 * w.max(axis=0).sum()
            for w in (g[s:s + period] for s in range(0, len(g), period))
        )
        err = abs(paid - expected)
        print(f"[2] telescoping, {label:14s} paid ${paid:,.2f} vs closed form "
              f"${expected:,.2f}  (err ${err:.6f})")
        if err >= 1e-6:
            failures.append(f"telescoping not exact for {label} (err {err})")

    _, hm = _run(demand_charge_rate=RATE, demand_charge_period_steps=8640)
    _, hd = _run(demand_charge_rate=RATE, demand_charge_period_steps=288)
    monthly = sum(h["total_demand_charge"] for h in hm)
    daily = sum(h["total_demand_charge"] for h in hd)
    print(f"[3] monthly window ${monthly/1e6:.3f} M, daily window ${daily/1e6:.3f} M "
          f"({(daily-monthly)/monthly*100:+.1f}%)")
    if daily > monthly + 1e-6:
        failures.append("prorated daily billing exceeded monthly billing")

    # 4. observability -------------------------------------------------------
    off = _make_env(SCEN, batch_enabled=True, peak_penalty_weight=ALPHA)
    on = _make_env(SCEN, batch_enabled=True, peak_penalty_weight=ALPHA,
                   demand_charge_rate=RATE)
    d_off, d_on = off.observation_space.shape[0], on.observation_space.shape[0]
    print(f"[4] obs dim {d_off} -> {d_on} (+{d_on - d_off}, expect +{off.n_dc})")
    if d_on - d_off != off.n_dc:
        failures.append(f"expected +{off.n_dc} obs dims, got +{d_on - d_off}")
    obs, _ = on.reset(seed=42)
    if obs.shape != on.observation_space.shape or not np.isfinite(obs).all():
        failures.append("observation malformed with demand charge enabled")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
