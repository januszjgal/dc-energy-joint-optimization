"""Where in time does PPO's advantage over the status quo concentrate?

No-retrain diagnostic replacing the superseded burst analysis (thesis_overview
§7.6): decompose the per-timestep cost advantage of PPO over the no-optimization
Status Quo by fleet-mean grid net-demand quartile. If grid-aware optimization is
doing what it claims, the advantage should concentrate in the HIGH-net-demand
windows (the duck-curve neck), not be uniform over time.

Outputs: prints per-quartile advantage shares and writes
output/peak_window_analysis.json (+ optional figure via build_paper_figures.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO  # noqa: E402

from baselines import StatusQuoPolicy  # noqa: E402
from evaluate import _make_env, run_episode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = [
    ("US batch", "env/scenarios/us_model.yaml", "models/ppo_us_model_batch.zip"),
    ("Global batch", "env/scenarios/global_model.yaml", "models/ppo_global_model_batch.zip"),
]


def per_step_cost_and_nd(history) -> tuple[np.ndarray, np.ndarray]:
    cost = np.array([h["total_cost"] for h in history])
    nd = np.array([np.mean([dc["net_demand"] for dc in h["per_dc"]]) for h in history])
    return cost, nd


def main() -> None:
    results = {}
    for label, scenario, model_path in CONFIGS:
        env = _make_env(Path(scenario), batch_enabled=True, peak_penalty_weight=0.015)
        _, hist_sq = run_episode(env, StatusQuoPolicy().predict, is_sb3=False)

        model = PPO.load(str(ROOT / model_path))
        env = _make_env(Path(scenario), batch_enabled=True, peak_penalty_weight=0.015)
        _, hist_ppo = run_episode(env, model.predict, is_sb3=True)

        c_sq, nd = per_step_cost_and_nd(hist_sq)
        c_ppo, _ = per_step_cost_and_nd(hist_ppo)
        n = min(len(c_sq), len(c_ppo))
        adv = c_sq[:n] - c_ppo[:n]  # $ saved by PPO at each step
        nd = nd[:n]

        qs = np.quantile(nd, [0.25, 0.5, 0.75])
        bucket = np.digitize(nd, qs)  # 0..3 = Q1(low ND)..Q4(high ND)
        shares = []
        for b in range(4):
            m = bucket == b
            shares.append({
                "quartile": f"Q{b+1}",
                "nd_range": [float(nd[m].min()), float(nd[m].max())],
                "step_share": float(m.mean()),
                "advantage_share": float(adv[m].sum() / adv.sum()),
                "advantage_usd": float(adv[m].sum()),
                "avg_advantage_per_step": float(adv[m].mean()),
            })
        results[label] = {
            "total_advantage_usd": float(adv.sum()),
            "by_nd_quartile": shares,
            "top_vs_bottom_per_step_ratio": float(shares[3]["avg_advantage_per_step"] /
                                                  max(shares[0]["avg_advantage_per_step"], 1e-9)),
        }

        print(f"\n=== {label}: PPO advantage over Status Quo by net-demand quartile ===")
        print(f"  total advantage: ${adv.sum():,.0f}")
        for s in shares:
            print(f"  {s['quartile']} (nd {s['nd_range'][0]:.2f}-{s['nd_range'][1]:.2f}): "
                  f"{s['advantage_share']:.1%} of savings in {s['step_share']:.0%} of steps "
                  f"(${s['avg_advantage_per_step']:,.0f}/step)")

    out = ROOT / "output" / "peak_window_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
