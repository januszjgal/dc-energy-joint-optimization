"""Plot the aggregate grid power-draw profile: no-optimization Status Quo vs optimized policies.

Shows how grid-aware routing + temporal deferral flattens the fleet's power draw
relative to the "run it as-is" status quo — the demand-smoothing story made visual.
Status Quo is the no-optimization reference (each cell served locally, immediately);
the optimized policies should shave the peak (higher load factor).

Usage:
    python analysis/plot_power_profile.py                       # baselines only
    python analysis/plot_power_profile.py --ppo-model models/ppo_us_model_batch.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

from baselines import StatusQuoPolicy, TroughSlotLookaheadPolicy  # noqa: E402
from evaluate import _make_env, run_episode  # noqa: E402

STEPS_PER_DAY = 288  # 5-min intervals


def _profile(scenario: Path, predict_fn, is_sb3: bool, alpha: float) -> np.ndarray:
    """Return the per-timestep aggregate grid-MW draw for one policy."""
    env = _make_env(scenario, batch_enabled=True, peak_penalty_weight=alpha)
    _, history = run_episode(env, predict_fn, is_sb3=is_sb3)
    return np.array([h.get("total_grid_mw", 0.0) for h in history], dtype=np.float64)


def _label(name: str, draw: np.ndarray) -> str:
    peak, mean = draw.max(), draw.mean()
    lf = mean / peak if peak > 0 else 0.0
    return f"{name}  (peak {peak:.0f} MW, load factor {lf:.3f})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Status-Quo vs optimized power draw")
    ap.add_argument("--scenario", type=Path, default=Path("env/scenarios/us_model.yaml"))
    ap.add_argument("--ppo-model", type=Path, default=None, help="optional trained PPO .zip")
    ap.add_argument("--alpha", type=float, default=0.015, help="peak_penalty_weight")
    ap.add_argument("--zoom-days", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("output/power_profile_comparison.png"))
    args = ap.parse_args()

    profiles: dict[str, np.ndarray] = {}

    # Always-available baselines
    profiles["Status Quo (no optimization)"] = _profile(
        args.scenario, StatusQuoPolicy().predict, is_sb3=False, alpha=args.alpha
    )
    profiles["Trough-Slot Lookahead"] = _profile(
        args.scenario, TroughSlotLookaheadPolicy().predict, is_sb3=False, alpha=args.alpha
    )

    # Optional trained PPO (skips cleanly if the action space doesn't match, e.g. stale model)
    if args.ppo_model is not None:
        try:
            from stable_baselines3 import PPO

            model = PPO.load(str(args.ppo_model))
            profiles["PPO"] = _profile(
                args.scenario, model.predict, is_sb3=True, alpha=args.alpha
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [skip PPO] {type(e).__name__}: {e}")

    n = min(len(p) for p in profiles.values())
    days = np.arange(n) / STEPS_PER_DAY
    colors = {"Status Quo (no optimization)": "#888888"}

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(14, 9))

    # Top: whole episode, 1-hour rolling mean for readability
    for name, draw in profiles.items():
        d = draw[:n]
        smooth = np.convolve(d, np.ones(12) / 12, mode="same")
        ax_full.plot(days, smooth, label=_label(name, d), linewidth=1.0,
                     color=colors.get(name), alpha=0.9)
    ax_full.set_title("Aggregate fleet grid draw over the full trace (1-hour rolling mean)")
    ax_full.set_xlabel("Days")
    ax_full.set_ylabel("Aggregate grid draw (MW)")
    ax_full.legend(fontsize=9, loc="upper right")
    ax_full.grid(True, alpha=0.3)

    # Bottom: first `zoom_days` days, raw — shows the diurnal peak shaving
    zoom_n = min(args.zoom_days * STEPS_PER_DAY, n)
    for name, draw in profiles.items():
        ax_zoom.plot(days[:zoom_n], draw[:zoom_n], label=name, linewidth=1.0,
                     color=colors.get(name), alpha=0.9)
    for d in range(args.zoom_days + 1):
        ax_zoom.axvline(d, color="gray", linestyle="--", alpha=0.25)
    ax_zoom.set_title(f"Zoom: first {args.zoom_days} days (raw) — peak shaving detail")
    ax_zoom.set_xlabel("Days")
    ax_zoom.set_ylabel("Aggregate grid draw (MW)")
    ax_zoom.legend(fontsize=9, loc="upper right")
    ax_zoom.grid(True, alpha=0.3)

    fig.suptitle("Power-draw profile: no-optimization Status Quo vs optimized scheduling",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")
    for name, draw in profiles.items():
        print(f"  {_label(name, draw[:n])}")


if __name__ == "__main__":
    main()
