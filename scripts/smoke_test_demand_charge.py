"""Smoke test the demand-charge term (§3.4).

Checks the properties the corrected formulation depends on:

  1. Default-off regression — `demand_charge_rate=0.0` leaves the reward
     unchanged while the full-cycle reference charge is still reported.
  2. Full-cycle consistency — in-reward and post-hoc charges use the same
     full-episode billing convention.
  3. Exact telescoping — incremental charges sum to each complete period
     maximum; incomplete trailing periods are rejected.
  4. Boundary observability — the next period's peak resets before the next
     action, and billing progress/rate context are observable.
  5. Discount safety — demand-charge training resolves to gamma=1.0 and rejects
     discounted configurations that would change the billed objective.
  6. Economic safety — backlog/expiry penalties exceed the largest modeled
     one-step saving from dropping work, and raw dollar costs are reward-scaled.
  7. Finite-horizon safety — dense arrival-minus-completion shaping telescopes
     exactly to expired plus terminal batch work.

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
from env.dc_site import DataCenterSite
from env.multi_dc_env import (
    REFERENCE_DEMAND_CHARGE_RATE,
    MultiDCEnv,
    resolve_training_gamma,
)
from env.power_model import PowerModel
from evaluate import _make_env, compute_summary, run_episode

SCEN = ROOT / "env" / "scenarios" / "global_model_v2_2025.yaml"
ALPHA = 0.015
RATE = REFERENCE_DEMAND_CHARGE_RATE


def _run(**kw):
    env = _make_env(SCEN, batch_enabled=True, peak_penalty_weight=ALPHA, **kw)
    _, hist = run_episode(env, StatusQuoPolicy().predict, is_sb3=False)
    return env, hist


def _tiny_env(rate: float, period: int, n: int = 4) -> MultiDCEnv:
    site = DataCenterSite(
        name="meter",
        workload=np.full(n, 0.5),
        solar=np.zeros(n),
        price=np.zeros(n),
        net_demand=np.zeros(n),
        rated_power_mw=1.0,
        capacity=1.0,
    )
    return MultiDCEnv(
        [site],
        PowerModel(0.0, 1.0, 1.0),
        max_steps=n,
        demand_charge_rate=rate,
        demand_charge_period_steps=period,
        steps_per_day=period,
        site_context=False,
    )


def _tiny_batch_env() -> MultiDCEnv:
    n = 2
    site = DataCenterSite(
        name="batch-meter",
        workload=np.ones(n),
        solar=np.zeros(n),
        price=np.zeros(n),
        net_demand=np.zeros(n),
        rated_power_mw=1.0,
        capacity=1.0,
        batch_fraction=1.0,
        batch_mean_duration_sec=3600.0,
    )
    return MultiDCEnv(
        [site],
        PowerModel(0.0, 1.0, 1.0),
        max_steps=n,
        demand_charge_rate=1.0,
        batch_enabled=True,
        batch_spatial_routing=False,
        site_context=False,
    )


def main() -> int:
    failures = []

    # 1. default off ---------------------------------------------------------
    env, hist = _run()
    summary = compute_summary(hist, batch_enabled=True)
    print("[1] default OFF")
    print(f"    total_cost                ${summary['total_cost']:,.2f}")
    print(f"    demand charge IN reward   ${summary['total_demand_charge']:,.2f}")
    print(f"    demand charge REPORTED    ${summary['demand_charge_ref']/1e6:.3f} M "
          f"@ ${summary['demand_charge_ref_rate']}/kW-cycle")
    print(f"    billed peak (sum of per-site maxima) "
          f"{summary['billed_peak_sum_mw']:.1f} MW")
    if env.demand_charge_enabled:
        failures.append("demand charge should default to disabled")
    if summary["total_demand_charge"] != 0.0:
        failures.append("disabled term still charged the reward")
    if summary["demand_charge_ref"] <= 0.0:
        failures.append("reference charge not reported when term is disabled")

    # 2. full-cycle consistency ---------------------------------------------
    env, hist = _run(demand_charge_rate=RATE)
    summary = compute_summary(hist, batch_enabled=True)
    paid = summary["total_demand_charge"]
    g = np.array([[dc["grid_mw"] for dc in h["per_dc"]] for h in hist])
    expected = RATE * 1000.0 * g.max(axis=0).sum()
    err = abs(paid - expected)
    ref_err = abs(paid - summary["demand_charge_ref"])
    print(f"[2] full-cycle paid ${paid:,.2f} vs closed form ${expected:,.2f} "
          f"(err ${err:.6f}); reference err ${ref_err:.6f}")
    if env.demand_charge_period_steps != env.max_steps:
        failures.append("default billing period is not the full episode")
    if err >= 1e-6 or ref_err >= 1e-6:
        failures.append(
            f"full-cycle reward/reference mismatch (closed {err}, ref {ref_err})"
        )
    if summary["demand_charge_rate"] != RATE:
        failures.append("configured demand charge rate missing from summary")
    if summary["demand_charge_period_steps"] != env.max_steps:
        failures.append("configured billing period missing from summary")

    # 3. multiple complete periods; reject incomplete trailing period --------
    period = 2
    period_rate = 0.5
    periodic = _tiny_env(period_rate, period)
    _, hist = run_episode(
        periodic, StatusQuoPolicy().predict, is_sb3=False
    )
    paid = sum(h["total_demand_charge"] for h in hist)
    g = np.array([[dc["grid_mw"] for dc in h["per_dc"]] for h in hist])
    windows = [g[s:s + period] for s in range(0, len(g), period)]
    expected = sum(
        period_rate * 1000.0 * w.max(axis=0).sum()
        for w in windows
    )
    err = abs(paid - expected)
    print(f"[3] {len(windows)} configured periods paid ${paid:,.2f} vs "
          f"closed form ${expected:,.2f} (err ${err:.6f})")
    if err >= 1e-6:
        failures.append(f"multi-period telescoping not exact (err {err})")
    try:
        _tiny_env(rate=1.0, period=2, n=5)
    except ValueError:
        pass
    else:
        failures.append("incomplete trailing billing period was not rejected")
    site = DataCenterSite(
        name="batch-period",
        workload=np.ones(4),
        solar=np.zeros(4),
        price=np.zeros(4),
        net_demand=np.zeros(4),
        batch_fraction=1.0,
    )
    try:
        MultiDCEnv(
            [site],
            PowerModel(0.0, 1.0, 1.0),
            max_steps=4,
            batch_enabled=True,
            demand_charge_rate=1.0,
            demand_charge_period_steps=2,
            site_context=False,
        )
    except ValueError:
        pass
    else:
        failures.append(
            "batch demand charge allowed a hidden non-episode billing phase"
        )

    # 4. observation shape + boundary semantics -----------------------------
    off = _make_env(SCEN, batch_enabled=True, peak_penalty_weight=ALPHA)
    on = _make_env(
        SCEN,
        batch_enabled=True,
        peak_penalty_weight=ALPHA,
        demand_charge_rate=RATE,
    )
    d_off, d_on = off.observation_space.shape[0], on.observation_space.shape[0]
    expected_extra = off.n_dc + 2
    print(f"[4] obs dim {d_off} -> {d_on} (+{d_on - d_off}, "
          f"expect +{expected_extra})")
    if d_on - d_off != expected_extra:
        failures.append(
            f"expected +{expected_extra} observation dims, got +{d_on - d_off}"
        )
    obs, _ = on.reset(seed=42)
    if obs.shape != on.observation_space.shape or not np.isfinite(obs).all():
        failures.append("observation malformed with demand charge enabled")
    if not np.allclose(obs[-2:], [0.0, 1.0]):
        failures.append(f"initial billing context malformed: {obs[-2:]}")

    tiny = _tiny_env(rate=1.0, period=2)
    obs, _ = tiny.reset()
    peak_idx = 6  # six spatial-only site features, then billed peak
    obs, _, _, _, _ = tiny.step(np.array([0.0]))
    if obs[peak_idx] <= 0.0:
        failures.append("running peak was not exposed after first charge")
    obs, _, _, _, _ = tiny.step(np.array([0.0]))
    if obs[peak_idx] != 0.0 or obs[-2] != 0.0:
        failures.append(
            "new billing period was not reset before its first observation"
        )

    # 5. discount safety -----------------------------------------------------
    print("[5] gamma defaults: disabled -> 0.99, enabled -> 1.0")
    if resolve_training_gamma(None, 0.0) != 0.99:
        failures.append("disabled demand charge no longer preserves gamma=0.99")
    if resolve_training_gamma(None, RATE) != 1.0:
        failures.append("enabled demand charge did not default to gamma=1.0")
    try:
        resolve_training_gamma(0.99, RATE)
    except ValueError:
        pass
    else:
        failures.append("discounted demand-charge training was not rejected")

    # 6. tariff-aware penalty floor + reward scale ---------------------------
    guarded = _make_env(
        SCEN,
        batch_enabled=True,
        peak_penalty_weight=ALPHA,
        demand_charge_rate=RATE,
    )
    obs, _ = guarded.reset(seed=42)
    action = StatusQuoPolicy().predict(obs, guarded)
    _, reward, _, _, info = guarded.step(action)
    print(
        "[6] economic floor "
        f"${guarded.economic_penalty_floor:,.2f}/unit, "
        f"reward scale {guarded.reward_scale:g}"
    )
    if guarded.economic_penalty_floor <= 0.0:
        failures.append("demand charge did not activate an economic penalty floor")
    if guarded.backlog_weight < guarded.economic_penalty_floor:
        failures.append("backlog penalty is below the economic safety floor")
    if guarded.deadline_penalty_weight < guarded.economic_penalty_floor:
        failures.append("expiry penalty is below the economic safety floor")
    if guarded.reward_scale != 1e-4:
        failures.append("demand-charge reward scale is not 1e-4")
    if not np.isclose(reward, -info["total_cost"] * guarded.reward_scale):
        failures.append("scaled reward does not preserve raw total cost")
    if info["effective_backlog_weight"] != guarded.backlog_weight:
        failures.append("effective backlog penalty missing from step metadata")

    # 7. terminal batch carryover -------------------------------------------
    terminal_env = _tiny_batch_env()
    obs, _ = terminal_env.reset()
    hold = np.array([0.0, -3.0])
    terminal_info = None
    terminal_history = []
    for _ in range(terminal_env.max_steps):
        obs, _, _, _, terminal_info = terminal_env.step(hold)
        terminal_history.append(terminal_info)
    print(
        "[7] terminal pool "
        f"{terminal_info['total_batch_pool']:.4f}, accounting cost "
        f"${sum(h['total_batch_accounting_cost'] for h in terminal_history):,.2f}"
    )
    if terminal_info["total_batch_pool"] <= 0.0:
        failures.append("terminal carryover probe left no batch work")
    expected_unfinished = (
        terminal_env.deadline_penalty_weight
        * terminal_info["total_batch_pool"]
    )
    actual_unfinished = sum(
        h["total_batch_accounting_cost"] for h in terminal_history
    )
    if not np.isclose(actual_unfinished, expected_unfinished):
        failures.append(
            "arrival-minus-completion shaping did not telescope to terminal work"
        )
    terminal_summary = compute_summary(
        terminal_history, batch_enabled=True
    )
    if terminal_summary["work_completed_fraction"] >= 1.0:
        failures.append("summary treated terminal batch carryover as completed")
    if not np.isclose(
        terminal_summary["total_unfinished_batch_cost"],
        expected_unfinished,
    ):
        failures.append("summary lost the dense unfinished-work accounting")

    print()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
