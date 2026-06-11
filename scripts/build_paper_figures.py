"""Generate the figures for the thesis paper (thesis_paper.docx).

Outputs to output/paper_figs/*.png at column width (~3.3in @ 200dpi).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output" / "paper_figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 7, "figure.dpi": 200})
FIGW = (3.35, 2.3)  # single-column


def fig1_results() -> None:
    configs = ["US legacy", "US batch", "Global legacy", "Global batch"]
    status_quo = [9.848, 9.848, 14.062, 14.062]
    trough = [9.934, 9.908, 13.751, 13.755]
    best_dqn = [9.759, 9.798, 13.760, 13.006]
    ppo = [9.666, 9.524, 12.617, 12.421]

    x = np.arange(len(configs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    ax.bar(x - 1.5 * w, status_quo, w, label="Status Quo", color="#9e9e9e")
    ax.bar(x - 0.5 * w, trough, w, label="Trough-Slot (oracle)", color="#7fb3d5")
    ax.bar(x + 0.5 * w, best_dqn, w, label="Best DQN", color="#f5b041")
    ax.bar(x + 1.5 * w, ppo, w, label="PPO", color="#27ae60")
    for i, v in enumerate(ppo):
        sq = status_quo[i]
        ax.annotate(f"−{100*(sq-v)/sq:.1f}%", (x[i] + 1.5 * w, v), ha="center",
                    va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=7)
    ax.set_ylabel("Total episode cost (M$)")
    ax.set_ylim(9, 15)
    ax.legend(ncol=2, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_results.png", bbox_inches="tight")
    plt.close(fig)


def fig2_tiers() -> None:
    t = pd.read_csv(ROOT / "data/cells/cell_b_tiers.csv")
    n = 3 * 288
    days = np.arange(n) / 288
    fig, ax = plt.subplots(figsize=FIGW)
    ax.plot(days, t["cpu_demand_norm"][:n], color="#333", lw=0.9, label="aggregate (measured)")
    ax.plot(days, t["service_demand_norm"][:n], color="#7fb3d5", lw=0.8, label="service (SLO tiers)")
    ax.plot(days, t["batch_demand_norm"][:n], color="#27ae60", lw=0.8, label="batch (no-SLO tiers)")
    ax.set_xlabel("Days")
    ax.set_ylabel("CPU (fraction of capacity)")
    ax.set_title("Cell b: measured per-tier demand")
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_tiers.png", bbox_inches="tight")
    plt.close(fig)


def fig3_power() -> None:
    sc = pd.read_csv(ROOT / "data/power_model_scatter.csv")
    pm = json.load(open(ROOT / "data/power_model_params.json"))
    colors = {"a": "#e74c3c", "b": "#f5b041", "c": "#7fb3d5", "d": "#27ae60"}
    fig, ax = plt.subplots(figsize=FIGW)
    for cell, g in sc.groupby("cell"):
        ax.scatter(g["cpu_util"], g["power_util"], s=2, alpha=0.25, color=colors[cell])
        m = pm["per_cell_cpu_model"][cell]
        xs = np.linspace(g["cpu_util"].min(), g["cpu_util"].max(), 20)
        ax.plot(xs, m["idle_power"] + m["slope"] * xs, color=colors[cell], lw=1.2,
                label=f"cell {cell}: {m['idle_power']:.2f}+{m['slope']:.2f}u (R²={m['r_squared']:.2f})")
    xs = np.linspace(sc["cpu_util"].min(), sc["cpu_util"].max(), 20)
    ax.plot(xs, pm["idle_power"] + pm["slope"] * xs, "k--", lw=1.2,
            label=f"pooled (R²={pm['r_squared']:.2f})")
    ax.set_xlabel("CPU utilization")
    ax.set_ylabel("Power utilization")
    ax.set_title("Per-cell power calibration (PowerData2019)")
    ax.legend(loc="lower right", fontsize=6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_power.png", bbox_inches="tight")
    plt.close(fig)


def fig4_profile() -> None:
    from stable_baselines3 import PPO

    from baselines import StatusQuoPolicy
    from evaluate import _make_env, run_episode

    sc = ROOT / "env/scenarios/us_model.yaml"
    env = _make_env(sc, batch_enabled=True, peak_penalty_weight=0.015)
    _, h_sq = run_episode(env, StatusQuoPolicy().predict, is_sb3=False)
    model = PPO.load(str(ROOT / "models/ppo_us_model_batch.zip"))
    env = _make_env(sc, batch_enabled=True, peak_penalty_weight=0.015)
    _, h_ppo = run_episode(env, model.predict, is_sb3=True)

    g_sq = np.array([h["total_grid_mw"] for h in h_sq])
    g_ppo = np.array([h["total_grid_mw"] for h in h_ppo])
    n = 4 * 288
    days = np.arange(n) / 288
    fig, ax = plt.subplots(figsize=FIGW)
    ax.plot(days, g_sq[:n], color="#9e9e9e", lw=0.7,
            label=f"Status Quo (peak {g_sq.max():.0f} MW)")
    ax.plot(days, g_ppo[:n], color="#27ae60", lw=0.7,
            label=f"PPO (peak {g_ppo.max():.0f} MW)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Fleet grid draw (MW)")
    ax.set_title("US batch: fleet power profile (first 4 days)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_profile.png", bbox_inches="tight")
    plt.close(fig)


def fig5_peakwindow() -> None:
    d = json.load(open(ROOT / "output/peak_window_analysis.json"))
    labels = ["Q1\n(slack)", "Q2", "Q3", "Q4\n(peak)"]
    us = [s["advantage_share"] * 100 for s in d["US batch"]["by_nd_quartile"]]
    gl = [s["advantage_share"] * 100 for s in d["Global batch"]["by_nd_quartile"]]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=FIGW)
    ax.bar(x - 0.18, us, 0.36, label="US batch", color="#7fb3d5")
    ax.bar(x + 0.18, gl, 0.36, label="Global batch", color="#27ae60")
    ax.axhline(25, color="k", ls="--", lw=0.8, label="uniform (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of PPO savings (%)")
    ax.set_title("PPO advantage by grid net-demand quartile")
    ax.set_ylim(0, 35)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_peakwindow.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_results()
    fig2_tiers()
    fig3_power()
    fig4_profile()
    fig5_peakwindow()
    print(f"Figures written to {OUT}")
