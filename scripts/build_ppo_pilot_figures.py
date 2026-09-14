"""Build the PPO pilot figures and the numbers quoted in latex/ppo_solution.tex.

Reads a tagged joint-objective pilot from ``output/four_market_joint_v3/pilot/<tag>/``
and its checkpoints under ``models/``, reruns the no-flexibility baseline on every
training month (for a reference line) and on May, and replays each seed's final
policy on May. Vector PDFs, PNG previews, and ``figure_data.json`` are written to
``docs/figures/ppo_pilot/``.

Colors are the first three categorical slots of the reference data-viz palette,
validated all-pairs for colorblind separation. The aqua slot is below 3:1 contrast
on the surface, so every figure carries a legend and the chapter tabulates values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from energy_model_v3.four_market_v2 import (  # noqa: E402
    ARTIFACT_NAMESPACE, FACTORY_ROOT, MARKETS, make_four_market_env,
)
from ramp_rl.contract import EnvRequest  # noqa: E402
from ramp_rl.evaluation import _run_episode  # noqa: E402

OUT_DIR = ROOT / "docs" / "figures" / "ppo_pilot"
SEEDS = (4101, 4102, 4103)
VALIDATION_MONTH = "2025-05"
SEED_COLORS = dict(zip(SEEDS, ("#2a78d6", "#eb6834", "#1baf7a")))
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)
MARKET_LABELS = {
    "CAISO_NP15": "CAISO NP15",
    "MISO_MINN_HUB": "MISO Minnesota",
    "SPP_NORTH_HUB": "SPP North",
    "ISONE_NEMA": "ISO-NE NEMA",
}
COLUMN_WIDTH_IN = 3.45
PAGE_WIDTH_IN = 7.0


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 8,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def no_flex_month(split: str, window_id: str) -> tuple[float, list[dict[str, Any]]]:
    """Run the no-flexibility baseline for one month; return J and step infos."""
    env = make_four_market_env(EnvRequest(split=split, seed=0, window_id=window_id))
    try:
        env.reset(seed=0)
        total, infos = 0.0, []
        while True:
            _, reward, done, _, info = env.step(env.evaluation_action("status_quo"))
            total += reward
            infos.append(info)
            if done:
                return -total, infos
    finally:
        env.close()


def policy_month(tag: str, seed: int) -> list[dict[str, Any]]:
    checkpoint = ROOT / "models" / ARTIFACT_NAMESPACE / "pilot" / tag / f"seed-{seed}"
    model = PPO.load(checkpoint / "model.zip", device="cpu")
    infos, _ = _run_episode(
        factory=make_four_market_env, split="validation", seed=seed,
        window_id=VALIDATION_MONTH, model=model,
        normalization_path=checkpoint / "vecnormalize.pkl",
    )
    return infos


def month_arrays(infos: list[dict[str, Any]], site_ids: list[str]) -> dict[str, np.ndarray]:
    return {
        "power": np.array([[i["per_site"][s]["power_mw"] for s in site_ids] for i in infos]),
        "service": np.array([i["service_allocation"] for i in infos]),
        "queued": np.array([i["batch_queue"]["queued"] for i in infos]),
        "adjusted": np.array([i["semantic_adjustment_applied"] for i in infos]),
    }


def behavior_metrics(
    arrays: dict[str, np.ndarray], baseline: dict[str, np.ndarray],
    net: np.ndarray, warm: np.ndarray, scales: np.ndarray,
    service_arrivals: np.ndarray, batch_arrivals: np.ndarray,
) -> dict[str, Any]:
    native = np.diff(net / scales, axis=0)
    result: dict[str, Any] = {"per_market": {}}
    for index, market in enumerate(MARKETS):
        run = np.concatenate([warm[:, index], arrays["power"][:, index]])
        base = np.concatenate([warm[:, index], baseline["power"][:, index]])
        adjusted = np.diff((net[:, index] + run) / scales[index])
        impact = float(np.sum(adjusted**2 - native[:, index] ** 2))
        result["per_market"][market] = {
            "squared_ramp_cut_pct": -100.0 * impact / float(np.sum(native[:, index] ** 2)),
            "peak_change_mw": float((net[1:, index] + run[1:]).max() - (net[1:, index] + base[1:]).max()),
        }
    result["service_moved_pct"] = float(
        100.0 * np.abs(arrays["service"] - service_arrivals).sum() / 2.0 / service_arrivals.sum()
    )
    result["mean_batch_wait_hours"] = float(arrays["queued"].sum() / batch_arrivals.sum())
    result["decoder_adjustment_pct"] = float(100.0 * arrays["adjusted"].mean())
    return result


def figure_learning_curves(curves: dict[int, pd.DataFrame], baseline_return: float) -> None:
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.2))
    for seed, frame in curves.items():
        done = frame[frame["episode_count"] > 0]
        smooth = done["mean_completed_episode_return"].rolling(10, min_periods=1).mean()
        ax.plot(done["interaction_count"] / 1e6, smooth, color=SEED_COLORS[seed], label=f"Seed {seed}")
    ax.axhline(baseline_return, color=MUTED, linewidth=1.0)
    ax.annotate(
        "No-flexibility baseline, mean training month", xy=(1.02, baseline_return),
        xytext=(0, -4), textcoords="offset points", ha="right", va="top", color=INK_2, fontsize=6.5,
    )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.0105, 0.0165)
    ax.set_xlabel("Training interactions (millions)")
    ax.set_ylabel("Month return, $-J_e$")
    ax.set_title("Training return per completed month")
    ax.legend(loc="center right", ncols=1, handlelength=1.4, bbox_to_anchor=(1.0, 0.42))
    save(fig, "learning_curves")


def figure_checkpoints(summaries: dict[int, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.1))
    for seed, summary in summaries.items():
        rows = sorted(
            (int(k), v["improvement_J"]) for k, v in summary["validation_by_interactions"].items()
        )
        x = [k / 1e6 for k, _ in rows]
        y = [value for _, value in rows]
        ax.plot(x, y, color=SEED_COLORS[seed], label=f"Seed {seed}", marker="o", markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 0.02)
    ax.set_xticks([0.1536, 0.512, 1.024], ["0.15", "0.51", "1.02"])
    ax.set_xlabel("Checkpoint (millions of interactions)")
    ax.set_ylabel(r"$J_e(\mu_{\mathrm{NF}})-J_e(\mu)$")
    ax.set_title("May improvement over the no-flexibility baseline")
    ax.legend(loc="lower right", ncols=3, handlelength=1.4, columnspacing=1.0)
    save(fig, "checkpoint_improvement")


def figure_markets(metrics: dict[int, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH_IN, 2.3))
    positions = np.arange(len(MARKETS))
    width = 0.22
    panels = (
        ("squared_ramp_cut_pct", "Cut in squared one-hour ramp (%)", "Ramp smoothing by region"),
        ("peak_change_mw", "Peak change vs. no flexibility (MW)", "Monthly peak by region"),
    )
    for ax, (field, ylabel, title) in zip(axes, panels):
        for offset, seed in enumerate(SEEDS):
            values = [metrics[seed]["per_market"][market][field] for market in MARKETS]
            ax.bar(positions + (offset - 1) * width, values, width=width * 0.9,
                   color=SEED_COLORS[seed], label=f"Seed {seed}", linewidth=0)
        ax.axhline(0.0, color=AXIS, linewidth=0.8)
        ax.set_xticks(positions, [MARKET_LABELS[m] for m in MARKETS])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[1].legend(loc="lower left", ncols=3, handlelength=1.0, columnspacing=1.0)
    fig.tight_layout(w_pad=2.5)
    save(fig, "regional_results")


def figure_isone_window(
    net: np.ndarray, warm: np.ndarray, baseline: dict[str, np.ndarray],
    run: dict[str, np.ndarray], timestamps: pd.DatetimeIndex, seed: int,
) -> dict[str, Any]:
    index = MARKETS.index("ISONE_NEMA")
    adjusted_nf = net[1:, index] + baseline["power"][:, index]
    peak_hour = int(np.argmax(adjusted_nf))
    start, stop = max(peak_hour - 36, 0), min(peak_hour + 36, len(adjusted_nf))
    hours = timestamps[start:stop]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(PAGE_WIDTH_IN, 3.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]},
    )
    top.plot(hours, net[1:, index][start:stop], color=INK_2)
    top.set_ylabel("Original net load (MW)")
    top.set_title("ISO-NE NEMA around its May monthly peak hour")
    bottom.plot(hours, baseline["power"][start:stop, index], color=MUTED, label="No flexibility")
    bottom.plot(hours, run["power"][start:stop, index], color=SEED_COLORS[seed], label=f"PPO, seed {seed}")
    bottom.set_ylabel("Site power (MW)")
    for ax in (top, bottom):
        ax.axvline(timestamps[peak_hour], color=AXIS, linewidth=0.8)
    top.annotate("Peak hour (no flexibility)", xy=(timestamps[peak_hour], 0.0),
                 xycoords=("data", "axes fraction"), xytext=(4, 3), textcoords="offset points",
                 ha="left", va="bottom", color=INK_2, fontsize=6.5)
    bottom.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2, borderaxespad=0.1)
    bottom.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d\n%H:%M UTC"))
    fig.tight_layout(h_pad=1.8)
    save(fig, "isone_peak_window")
    return {
        "peak_hour_utc": timestamps[peak_hour].isoformat(),
        "window_start_utc": hours[0].isoformat(),
        "window_end_utc": hours[-1].isoformat(),
        "no_flex_power_at_peak_mw": float(baseline["power"][peak_hour, index]),
        "policy_power_at_peak_mw": float(run["power"][peak_hour, index]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="impact-50-50-1m")
    parser.add_argument("--example-seed", type=int, default=4102, choices=SEEDS)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    pilot = ROOT / "output" / ARTIFACT_NAMESPACE / "pilot" / args.tag
    summaries = {s: json.loads((pilot / f"seed-{s}" / "summary.json").read_text(encoding="utf-8")) for s in SEEDS}
    curves = {s: pd.read_csv(pilot / f"seed-{s}" / "learning_curve.csv") for s in SEEDS}

    factory = json.loads((FACTORY_ROOT / "factory.json").read_text(encoding="utf-8"))
    training_J = {month: no_flex_month("train", month)[0] for month in sorted(factory["windows"]["train"])}
    baseline_return = -float(np.mean(list(training_J.values())))

    fixture = json.loads((FACTORY_ROOT / "windows" / VALIDATION_MONTH / "fixture.json").read_text(encoding="utf-8"))
    site_ids = [site["site_id"] for site in fixture["sites"]]
    service_arrivals = np.array(fixture["workload"]["service_arrivals"])
    batch_arrivals = np.array(fixture["workload"]["batch_arrivals"])
    warm = np.array(fixture["workload"]["warm_power_mw"])
    stats = json.loads((ROOT / "data" / "four_market_2025" / "frozen_stats.json").read_text(encoding="utf-8"))
    scales = np.array([stats["gross_q95_mw"][m] for m in MARKETS])
    panel = pd.read_csv(ROOT / "data" / "four_market_2025" / "months" / VALIDATION_MONTH / "canonical_panel.csv")
    pivot = panel.pivot(index="timestamp_utc", columns="market_id", values="net_load_mw").sort_index()
    net = pivot[list(MARKETS)].to_numpy(dtype=float)
    timestamps = pd.DatetimeIndex(pd.to_datetime(pivot.index[1:], utc=True))

    may_J, may_infos = no_flex_month("validation", VALIDATION_MONTH)
    baseline = month_arrays(may_infos, site_ids)
    runs = {seed: month_arrays(policy_month(args.tag, seed), site_ids) for seed in SEEDS}
    metrics = {
        seed: behavior_metrics(runs[seed], baseline, net, warm, scales, service_arrivals, batch_arrivals)
        for seed in SEEDS
    }

    figure_learning_curves(curves, baseline_return)
    figure_checkpoints(summaries)
    figure_markets(metrics)
    window = figure_isone_window(net, warm, baseline, runs[args.example_seed], timestamps, args.example_seed)

    data = {
        "tag": args.tag,
        "no_flex_training_month_J": training_J,
        "no_flex_training_mean_J": -baseline_return,
        "no_flex_may_J": may_J,
        "seeds": {
            str(seed): {
                "training_minutes": summaries[seed]["training_elapsed_seconds"] / 60.0,
                "interactions_per_second": summaries[seed]["training_interactions_per_second"],
                "by_checkpoint": {
                    k: {
                        "policy_J": v["policy_J"],
                        "improvement_J": v["improvement_J"],
                        "ramp_weighted_improvement": v["components"]["ramp"]["weighted_improvement"],
                        "peak_weighted_improvement": v["components"]["net_load_peak"]["weighted_improvement"],
                        "peak_impact": v["components"]["net_load_peak"]["policy"],
                        "status_quo_peak_impact": v["components"]["net_load_peak"]["status_quo"],
                        "service_unserved": v["service_unserved"],
                        "batch_unfinished": v["batch_unfinished"],
                    }
                    for k, v in summaries[seed]["validation_by_interactions"].items()
                },
                "may_behavior": metrics[seed],
            }
            for seed in SEEDS
        },
        "isone_window": {"seed": args.example_seed, **window},
    }
    (OUT_DIR / "figure_data.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in data.items() if k != "seeds"}, indent=2))
    for seed in SEEDS:
        print(seed, json.dumps(metrics[seed]))


if __name__ == "__main__":
    main()
