"""Clean PPO vs Status Quo grid-stress comparison.

Reproduces evaluate.py's "DC Contribution to Grid Stress" metric
(Σ_i grid_mw_i × net_demand_norm_i, 1-hour rolling mean) but for just two
policies — the canonical context-aware PPO (seed 101) and the no-optimization
Status Quo — so the demand-smoothing gap is legible.

Usage:  python scripts/plot_ppo_vs_statusquo_stress.py [--scenario global_model] [--batch]
Writes output/ppo_vs_statusquo_gridstress_<scenario>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from baselines import StatusQuoPolicy  # noqa: E402
from evaluate import _make_env, run_episode  # noqa: E402

ALPHA = 0.015
WINDOW = 12  # 1-hour smoothing (12 × 5-min steps)


def grid_stress(history) -> np.ndarray:
    series = [sum(dc.get("grid_mw", 0.0) * dc.get("net_demand", 0.0)
                  for dc in h["per_dc"]) for h in history]
    return pd.Series(series).rolling(WINDOW, min_periods=1).mean().to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="global_model")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    scen = ROOT / "env" / "scenarios" / f"{args.scenario}.yaml"
    suffix = "_batch" if args.batch else ""
    model_path = ROOT / "models" / "review_ctx" / f"s{args.seed}" / \
        f"ppo_{args.scenario}{suffix}.zip"

    env = _make_env(scen, batch_enabled=args.batch, peak_penalty_weight=ALPHA)
    _, h_sq = run_episode(env, StatusQuoPolicy().predict, is_sb3=False)
    env = _make_env(scen, batch_enabled=args.batch, peak_penalty_weight=ALPHA)
    _, h_ppo = run_episode(env, PPO.load(str(model_path)).predict, is_sb3=True)

    sq, ppo = grid_stress(h_sq), grid_stress(h_ppo)
    # area under each curve = total grid-stress contribution (the Φ driver)
    redux = 100 * (sq.sum() - ppo.sum()) / sq.sum()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sq, color="#9e9e9e", lw=1.1, label=f"Status Quo (mean {sq.mean():.1f})")
    ax.plot(ppo, color="#27ae60", lw=1.1, label=f"PPO (mean {ppo.mean():.1f})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Σ grid_mw × net_demand_norm (1-h smoothed)")
    ax.set_title(f"DC Contribution to Grid Stress — PPO vs Status Quo "
                 f"({args.scenario}{suffix}): −{redux:.1f}% area")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = ROOT / "output" / f"ppo_vs_statusquo_gridstress_{args.scenario}{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Status Quo mean {sq.mean():.2f}, PPO mean {ppo.mean():.2f}, "
          f"area reduction {redux:.1f}%")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
