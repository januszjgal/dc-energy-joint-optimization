"""Build the cross-validation figures and the numbers quoted in latex/thesis_draft.tex (Chapter 6).

Reads a tagged cross-validation run from ``output/four_market_joint_v3/cv/<tag>/``
and its checkpoints under ``models/``. For every fold it reruns the
no-flexibility baseline on the fold's training months (for the learning-curve
reference line) and on its held-out months, and replays each seed's final
policy on each held-out month. Each replay recomputes J_e from the raw panel
and site-power trajectory as an independent check of the reported score.
Vector PDFs, PNG previews, and ``figure_data.json`` go to ``docs/figures/ppo_cv/``.
The test month is never touched.

Colors are the first three categorical slots of the reference data-viz palette,
validated all-pairs for colorblind separation. The aqua slot is below 3:1 contrast
on the surface, so every figure carries a legend and the chapter tabulates values.

    python scripts/build_ppo_cv_figures.py --tag cv-trace-clock-1m
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
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
    ARTIFACT_NAMESPACE, MARKETS, factory_root, make_four_market_env, objective_from_factory,
)
from energy_model_v3.splits import CV_FOLDS, spec_months  # noqa: E402
from ramp_rl.contract import EnvRequest  # noqa: E402
from ramp_rl.evaluation import _run_episode  # noqa: E402

OUT_DIR = ROOT / "docs" / "figures" / "ppo_cv"
SEEDS = (4101, 4102, 4103)
FOLDS = tuple(CV_FOLDS)
EXAMPLE = ("k1", "2025-05", 4102)  # fold, held-out month, seed for the ISO-NE window
SEED_COLORS = dict(zip(SEEDS, ("#2a78d6", "#eb6834", "#1baf7a")))
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)
MARKET_LABELS = {
    "CAISO": "CAISO",
    "MISO": "MISO",
    "SPP": "SPP",
    "ISONE": "ISO-NE",
}
MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
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


def no_flex_month(spec: str, split: str, month: str) -> tuple[float, list[dict[str, Any]]]:
    """Run the no-flexibility baseline for one month; return J and step infos."""
    env = make_four_market_env(EnvRequest(split=split, seed=0, window_id=month), spec=spec)
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


def policy_month(tag: str, fold: str, seed: int, month: str) -> list[dict[str, Any]]:
    checkpoint = ROOT / "models" / ARTIFACT_NAMESPACE / "cv" / tag / fold / f"seed-{seed}"
    model = PPO.load(checkpoint / "model.zip", device="cpu")
    infos, _ = _run_episode(
        factory=partial(make_four_market_env, spec=f"cv-{fold}"), split="validation", seed=seed,
        window_id=month, model=model, normalization_path=checkpoint / "vecnormalize.pkl",
    )
    return infos


def month_arrays(infos: list[dict[str, Any]], site_ids: list[str]) -> dict[str, np.ndarray]:
    return {
        "power": np.array([[i["per_site"][s]["power_mw"] for s in site_ids] for i in infos]),
        "service": np.array([i["service_allocation"] for i in infos]),
        "queued": np.array([i["batch_queue"]["queued"] for i in infos]),
        "adjusted": np.array([i["semantic_adjustment_applied"] for i in infos]),
    }


class MonthInputs:
    """The panel, fixture, scales, and objective of one fold's held-out month.

    ``spec`` and ``split`` select another factory's month instead, such as the
    final factory's test month.
    """

    def __init__(self, fold: str, month: str, *, spec: str | None = None, split: str = "validation") -> None:
        spec = spec or f"cv-{fold}"
        factory = json.loads((factory_root(spec) / "factory.json").read_text(encoding="utf-8"))
        record = factory["windows"][split][month]
        fixture = json.loads((ROOT / record["fixture_path"]).read_text(encoding="utf-8"))
        stats = json.loads((ROOT / factory["frozen_stats_path"]).read_text(encoding="utf-8"))
        panel = pd.read_csv(ROOT / record["panel_path"])
        pivot = panel.pivot(index="timestamp_utc", columns="market_id", values="net_load_mw").sort_index()
        self.objective = objective_from_factory(factory)
        self.site_ids = [site["site_id"] for site in fixture["sites"]]
        self.service_arrivals = np.array(fixture["workload"]["service_arrivals"])
        self.batch_arrivals = np.array(fixture["workload"]["batch_arrivals"])
        self.warm = np.array(fixture["workload"]["warm_power_mw"])
        self.scales = np.array([stats["gross_q95_mw"][m] for m in MARKETS])
        self.net = pivot[list(MARKETS)].to_numpy(dtype=float)
        self.timestamps = pd.DatetimeIndex(pd.to_datetime(pivot.index[1:], utc=True))

    def recomputed_J(self, power: np.ndarray) -> float:
        """J_e from the raw panel and power trajectory, independent of the rewards."""
        full = np.vstack([self.warm, power])
        adjusted, native = (self.net + full) / self.scales, self.net / self.scales
        ramp = float((np.diff(adjusted, axis=0) ** 2 - np.diff(native, axis=0) ** 2).sum())
        peak = float((adjusted[1:].max(axis=0) - native[1:].max(axis=0)).sum())
        return self.objective.score(ramp, peak)


def behavior_metrics(run: dict[str, np.ndarray], baseline: dict[str, np.ndarray], inputs: MonthInputs) -> dict[str, Any]:
    native = np.diff(inputs.net / inputs.scales, axis=0)
    result: dict[str, Any] = {"per_market": {}}
    for index, market in enumerate(MARKETS):
        policy = np.concatenate([inputs.warm[:, index], run["power"][:, index]])
        base = np.concatenate([inputs.warm[:, index], baseline["power"][:, index]])
        adjusted = np.diff((inputs.net[:, index] + policy) / inputs.scales[index])
        impact = float(np.sum(adjusted**2 - native[:, index] ** 2))
        result["per_market"][market] = {
            "squared_ramp_cut_pct": -100.0 * impact / float(np.sum(native[:, index] ** 2)),
            "peak_change_mw": float(
                (inputs.net[1:, index] + policy[1:]).max() - (inputs.net[1:, index] + base[1:]).max()
            ),
        }
    result["service_moved_pct"] = float(
        100.0 * np.abs(run["service"] - inputs.service_arrivals).sum() / 2.0 / inputs.service_arrivals.sum()
    )
    result["mean_batch_wait_hours"] = float(run["queued"].sum() / inputs.batch_arrivals.sum())
    result["decoder_adjustment_pct"] = float(100.0 * run["adjusted"].mean())
    return result


def percentage_reductions(row: dict[str, Any], scales: dict[str, float]) -> dict[str, float | None]:
    """Reductions from no flexibility, as in the pilot, for one month's results."""
    regions = row["regional_metrics"]
    hours = row["step_count"]

    def reduction(baseline: float, policy: float) -> float | None:
        return 100.0 * (baseline - policy) / baseline if baseline > 0.0 else None

    ramp = {p: hours * sum(r[f"{p}_ramp_mean_squared"] for r in regions.values()) for p in ("status_quo", "policy")}
    peaks = {p: sum(r[f"{p}_peak_mw"] / scales[m] for m, r in regions.items()) for p in ("status_quo", "policy")}
    impact = row["components"]["net_load_peak"]
    return {
        "total_squared_adjusted_ramps_pct": reduction(ramp["status_quo"], ramp["policy"]),
        "summed_normalized_adjusted_regional_peaks_pct": reduction(peaks["status_quo"], peaks["policy"]),
        "summed_normalized_fleet_added_peak_impact_pct": reduction(impact["status_quo"], impact["policy"]),
    }


def figure_learning_curves(curves: dict[tuple[str, int], pd.DataFrame], baselines: dict[str, float]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(PAGE_WIDTH_IN, 2.1), sharey=True)
    for ax, fold in zip(axes, FOLDS):
        for seed in SEEDS:
            frame = curves[(fold, seed)]
            done = frame[frame["episode_count"] > 0]
            smooth = done["mean_completed_episode_return"].rolling(10, min_periods=1).mean()
            ax.plot(done["interaction_count"] / 1e6, smooth, color=SEED_COLORS[seed], label=f"Seed {seed}")
        ax.axhline(baselines[fold], color=MUTED, linewidth=1.0)
        held = ", ".join(MONTH_LABELS[int(m[-2:]) - 1] for m in CV_FOLDS[fold])
        ax.set_title(f"Fold {fold[-1]}\nheld out: {held}")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Interactions (millions)")
    axes[0].set_ylabel("Month return, $-J_e$")
    axes[0].annotate("No flexibility", xy=(1.02, baselines[FOLDS[0]]), xytext=(0, -4),
                     textcoords="offset points", ha="right", va="top", color=INK_2, fontsize=6.5)
    axes[-1].legend(loc="center right", handlelength=1.2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "learning_curves")


def figure_months(final: dict[tuple[str, int], dict[str, Any]], month_fold: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 2.3))
    months = sorted(month_fold)
    positions = np.arange(len(months))
    width = 0.26
    for offset, seed in enumerate(SEEDS):
        values = [final[(month_fold[m], seed)][m]["improvement_J"] for m in months]
        ax.bar(positions + (offset - 1) * width, values, width=width * 0.9,
               color=SEED_COLORS[seed], label=f"Seed {seed}", linewidth=0)
    ax.axhline(0.0, color=AXIS, linewidth=0.8)
    ax.set_xticks(positions, [f"{MONTH_LABELS[int(m[-2:]) - 1]}\nfold {month_fold[m][-1]}" for m in months])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"$J_e(\mu_{\mathrm{NF}})-J_e(\mu)$")
    ax.set_ylim(0, 0.026)
    ax.set_title("Improvement over no flexibility in each held-out 2025 month")
    ax.legend(loc="upper right", ncols=3, handlelength=1.2)
    save(fig, "cv_months")


def figure_checkpoints(summaries: dict[tuple[str, int], dict[str, Any]]) -> dict[str, dict[str, float]]:
    means: dict[str, dict[str, float]] = {}
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.1))
    for seed in SEEDS:
        by_checkpoint: dict[int, list[float]] = {}
        for fold in FOLDS:
            for count, months in summaries[(fold, seed)]["validation_by_interactions"].items():
                by_checkpoint.setdefault(int(count), []).extend(row["improvement_J"] for row in months.values())
        rows = sorted((count, float(np.mean(values))) for count, values in by_checkpoint.items() if len(values) == 12)
        means[str(seed)] = {str(count): value for count, value in rows}
        ax.plot([c / 1e6 for c, _ in rows], [v for _, v in rows], color=SEED_COLORS[seed], label=f"Seed {seed}",
                marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0.1536, 0.512, 1.024], ["0.15", "0.51", "1.02"])
    ax.set_xlabel("Checkpoint (millions of interactions)")
    ax.set_ylabel("Mean improvement, 12 months")
    ax.set_title("Cross-validated improvement by checkpoint")
    ax.legend(loc="lower right", ncols=3, handlelength=1.2, columnspacing=1.0)
    save(fig, "checkpoint_improvement")
    return means


def figure_markets(metrics: dict[tuple[str, int, str], dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    averages: dict[str, dict[str, dict[str, float]]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH_IN, 2.3))
    positions = np.arange(len(MARKETS))
    width = 0.22
    panels = (
        ("squared_ramp_cut_pct", "Cut in squared one-hour ramp (%)", "Ramp smoothing by market"),
        ("peak_change_mw", "Peak change vs. no flexibility (MW)", "Monthly peak by market"),
    )
    for seed in SEEDS:
        averages[str(seed)] = {
            market: {
                field: float(np.mean([v["per_market"][market][field] for (f, s, m), v in metrics.items() if s == seed]))
                for field, _, _ in panels
            }
            for market in MARKETS
        }
    for ax, (field, ylabel, title) in zip(axes, panels):
        for offset, seed in enumerate(SEEDS):
            values = [averages[str(seed)][market][field] for market in MARKETS]
            ax.bar(positions + (offset - 1) * width, values, width=width * 0.9,
                   color=SEED_COLORS[seed], label=f"Seed {seed}", linewidth=0)
        ax.axhline(0.0, color=AXIS, linewidth=0.8)
        ax.set_xticks(positions, [MARKET_LABELS[m] for m in MARKETS])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(ylabel)
        ax.set_title(title + ", mean of 12 months")
    axes[1].legend(loc="lower left", ncols=3, handlelength=1.0, columnspacing=1.0)
    fig.tight_layout(w_pad=2.5)
    save(fig, "regional_results")
    return averages


def figure_isone_window(inputs: MonthInputs, baseline: dict[str, np.ndarray], run: dict[str, np.ndarray],
                        seed: int, month: str) -> dict[str, Any]:
    index = MARKETS.index("ISONE")
    adjusted_nf = inputs.net[1:, index] + baseline["power"][:, index]
    peak_hour = int(np.argmax(adjusted_nf))
    start, stop = max(peak_hour - 36, 0), min(peak_hour + 36, len(adjusted_nf))
    hours = inputs.timestamps[start:stop]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(PAGE_WIDTH_IN, 3.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]},
    )
    top.plot(hours, inputs.net[1:, index][start:stop], color=INK_2)
    top.set_ylabel("Original net load (MW)")
    top.set_title(f"ISO-NE around its {pd.Timestamp(month).strftime('%B %Y')} monthly peak hour")
    bottom.plot(hours, baseline["power"][start:stop, index], color=MUTED, label="No flexibility")
    bottom.plot(hours, run["power"][start:stop, index], color=SEED_COLORS[seed], label=f"PPO, seed {seed}")
    bottom.set_ylabel("Site power (MW)")
    for ax in (top, bottom):
        ax.axvline(inputs.timestamps[peak_hour], color=AXIS, linewidth=0.8)
    top.annotate("Peak hour (no flexibility)", xy=(inputs.timestamps[peak_hour], 0.0),
                 xycoords=("data", "axes fraction"), xytext=(4, 3), textcoords="offset points",
                 ha="left", va="bottom", color=INK_2, fontsize=6.5)
    bottom.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2, borderaxespad=0.1)
    bottom.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d\n%H:%M UTC"))
    fig.tight_layout(h_pad=1.8)
    save(fig, "isone_peak_window")
    window = baseline["power"][start:stop, index]
    policy = run["power"][start:stop, index]
    return {
        "month": month, "peak_hour_utc": inputs.timestamps[peak_hour].isoformat(),
        "window_start_utc": hours[0].isoformat(), "window_end_utc": hours[-1].isoformat(),
        "no_flex_power_at_peak_mw": float(baseline["power"][peak_hour, index]),
        "policy_power_at_peak_mw": float(run["power"][peak_hour, index]),
        "no_flex_power_window_range_mw": [float(window.min()), float(window.max())],
        "policy_power_window_range_mw": [float(policy.min()), float(policy.max())],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="cv-trace-clock-1m")
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw the summary figures from an existing figure_data.json, without replays")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    root = ROOT / "output" / ARTIFACT_NAMESPACE / "cv" / args.tag
    summaries = {(f, s): json.loads((root / f / f"seed-{s}" / "summary.json").read_text(encoding="utf-8"))
                 for f in FOLDS for s in SEEDS}
    curves = {(f, s): pd.read_csv(root / f / f"seed-{s}" / "learning_curve.csv") for f in FOLDS for s in SEEDS}
    final = {key: value["validation_by_interactions"][str(value["effective_interactions"])]
             for key, value in summaries.items()}
    month_fold = {month: fold for fold, months in CV_FOLDS.items() for month in months}

    if args.figures_only:
        saved = json.loads((OUT_DIR / "figure_data.json").read_text(encoding="utf-8"))
        figure_learning_curves(curves, {f: -v for f, v in saved["no_flex_training_mean_J_by_fold"].items()})
        figure_months(final, month_fold)
        figure_checkpoints(summaries)
        # The per-month market metrics were saved by the last full run.
        figure_markets({
            (fold, seed, month): record["behavior"]
            for fold in FOLDS for seed in SEEDS
            for month, record in saved["runs"][f"{fold}/{seed}"]["months"].items()
        })
        return

    training_J = {fold: {m: no_flex_month(f"cv-{fold}", "train", m)[0] for m in spec_months(f"cv-{fold}")["train"]}
                  for fold in FOLDS}
    baselines = {fold: -float(np.mean(list(values.values()))) for fold, values in training_J.items()}

    metrics: dict[tuple[str, int, str], dict[str, Any]] = {}
    recompute_errors: list[float] = []
    example_window: dict[str, Any] = {}
    nf_J: dict[str, float] = {}
    for fold in FOLDS:
        stats_scales = json.loads((ROOT / json.loads((factory_root(f"cv-{fold}") / "factory.json")
                                   .read_text(encoding="utf-8"))["frozen_stats_path"]).read_text())["gross_q95_mw"]
        for month in CV_FOLDS[fold]:
            inputs = MonthInputs(fold, month)
            nf_J[month], nf_infos = no_flex_month(f"cv-{fold}", "validation", month)
            baseline = month_arrays(nf_infos, inputs.site_ids)
            for seed in SEEDS:
                run = month_arrays(policy_month(args.tag, fold, seed, month), inputs.site_ids)
                recompute_errors.append(abs(inputs.recomputed_J(run["power"]) - final[(fold, seed)][month]["policy_J"]))
                row = behavior_metrics(run, baseline, inputs)
                row["no_flexibility_percentage_reductions"] = percentage_reductions(final[(fold, seed)][month], stats_scales)
                metrics[(fold, seed, month)] = row
                if (fold, month, seed) == EXAMPLE:
                    example_window = figure_isone_window(inputs, baseline, run, seed, month)

    figure_learning_curves(curves, baselines)
    figure_months(final, month_fold)
    checkpoint_means = figure_checkpoints(summaries)
    regional = figure_markets(metrics)

    per_seed_mean = {str(s): float(np.mean([final[(month_fold[m], s)][m]["improvement_J"] for m in month_fold]))
                     for s in SEEDS}
    data = {
        "tag": args.tag,
        "no_flex_training_mean_J_by_fold": {fold: -value for fold, value in baselines.items()},
        "no_flex_J_by_month": dict(sorted(nf_J.items())),
        "max_abs_recomputed_J_error": max(recompute_errors),
        "cv_mean_improvement_J_by_seed": per_seed_mean,
        "cv_mean_improvement_J": float(np.mean(list(per_seed_mean.values()))),
        "cv_seed_std_improvement_J": float(np.std(list(per_seed_mean.values()), ddof=1)),
        "mean_improvement_J_by_checkpoint": checkpoint_means,
        "regional_mean_of_months": regional,
        "isone_window": {"fold": EXAMPLE[0], "seed": EXAMPLE[2], **example_window},
        "runs": {
            f"{fold}/{seed}": {
                "training_minutes": (summaries[(fold, seed)]["training_elapsed_seconds"] or float("nan")) / 60.0,
                "interactions_per_second": summaries[(fold, seed)]["training_interactions_per_second"],
                "months": {
                    month: {
                        "policy_J": row["policy_J"],
                        "status_quo_J": row["status_quo_J"],
                        "improvement_J": row["improvement_J"],
                        "ramp_weighted_improvement": row["components"]["ramp"]["weighted_improvement"],
                        "peak_weighted_improvement": row["components"]["net_load_peak"]["weighted_improvement"],
                        "service_unserved": row["service_unserved"],
                        "batch_unfinished": row["batch_unfinished"],
                        "certificate_violations": row["certificate_violations"],
                        "behavior": metrics[(fold, seed, month)],
                    }
                    for month, row in final[(fold, seed)].items()
                },
            }
            for fold in FOLDS for seed in SEEDS
        },
    }
    (OUT_DIR / "figure_data.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in data.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
