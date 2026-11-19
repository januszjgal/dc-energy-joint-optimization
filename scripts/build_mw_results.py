"""Replay the trained policies and report their effect on ramps and peaks in MW.

Run from the repository root:

    python scripts/build_mw_results.py

Writes docs/figures/ppo_mw/mw_results.json and the test-month figure
docs/figures/ppo_mw/test_month_mw.pdf (and .png). Read-only otherwise.

Every evaluated month is simulated three ways from the same panel: the market's own net load
(no data center), the no-flexibility baseline, and each trained policy. Reported per market:

  - one-hour ramps of the net load in MW: the mean absolute ramp and the steepest ramp of the
    month (the largest absolute hour-to-hour change);
  - the monthly net-load peak in MW, what the fleet adds to it under the no-flexibility baseline,
    what the policies remove, and the most any schedule could remove, given that a site always
    draws at least its idle power;
  - decoder corrections: the share of hours adjusted, how many sites the policy over-requests,
    the share of the hour's work the decoder moves, and the site power that this work carries.

Cross-validation covers the twelve validation months with the three runs of their iteration;
the test month covers the ten final policies. Each replay recomputes the joint score from the
power trajectory and stops if it differs from the evaluated score.
"""
from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from energy_model.four_market import (  # noqa: E402
    ARTIFACT_NAMESPACE, MARKETS, factory_root, make_four_market_env,
)
from energy_model.splits import CV_FOLDS, FINAL_SPEC, TEST_MONTHS  # noqa: E402
from env.models import SiteConfig  # noqa: E402
from env.projection import softmax  # noqa: E402
from ramp_rl.contract import EnvRequest, RampEnvAdapter  # noqa: E402
from scripts.build_ppo_cv_figures import (  # noqa: E402
    AXIS, INK_2, MARKET_LABELS, MUTED, PAGE_WIDTH_IN, MonthInputs, apply_style, month_arrays,
    no_flex_month,
)

OUT_DIR = ROOT / "docs" / "figures" / "ppo_mw"
CV_TAG = "cv-1m"
CV_SEEDS = (4101, 4102, 4103)
FINAL_SEEDS = tuple(range(4101, 4111))
TOL = 1e-6
POLICY_COLOR = "#2a78d6"
J_TOLERANCE = 1e-9


def site_configs(spec: str, split: str, month: str) -> list[SiteConfig]:
    factory = json.loads((factory_root(spec) / "factory.json").read_text(encoding="utf-8"))
    record = factory["windows"][split][month]
    fixture = json.loads((ROOT / record["fixture_path"]).read_text(encoding="utf-8"))
    return [SiteConfig(**site) for site in fixture["sites"]]


def replay(spec: str, split: str, month: str, model_dir: Path, seed: int):
    """Deterministic replay of one saved policy; returns step infos and actions."""
    model = PPO.load(model_dir / "model.zip", device="cpu")
    factory = partial(make_four_market_env, spec=spec)

    def make():
        request = EnvRequest(split=split, seed=seed, window_id=month, training=False)
        return RampEnvAdapter(factory(request), request)

    vec = VecNormalize.load(model_dir / "vecnormalize.pkl", DummyVecEnv([make]))
    vec.training = False
    vec.norm_reward = False
    infos, actions = [], []
    try:
        obs = vec.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(np.asarray(action[0], dtype=float))
            obs, _, dones, step_infos = vec.step(action)
            infos.append(step_infos[0])
            if bool(dones[0]):
                break
    finally:
        vec.close()
    return infos, np.asarray(actions)


def ramp_stats(series: np.ndarray) -> dict[str, float]:
    ramps = np.diff(series)
    magnitude = np.abs(ramps)
    return {
        "mean_abs_ramp_mw": float(magnitude.mean()),
        "steepest_ramp_mw": float(magnitude.max()),
        "sum_squared_ramp_mw2": float((ramps ** 2).sum()),
    }


def market_record(inputs: MonthInputs, sites: list[SiteConfig], nf_power: np.ndarray,
                  power: np.ndarray) -> dict[str, Any]:
    out = {}
    for j, market in enumerate(MARKETS):
        net = inputs.net[:, j]
        no_flex = net + np.concatenate([inputs.warm[:, j], nf_power[:, j]])
        policy = net + np.concatenate([inputs.warm[:, j], power[:, j]])
        idle_mw = sites[j].power_mw(0.0)
        peak_native = float(net[1:].max())
        out[market] = {
            "native": ramp_stats(net), "no_flex": ramp_stats(no_flex), "policy": ramp_stats(policy),
            "peak_native_mw": peak_native,
            "peak_no_flex_mw": float(no_flex[1:].max()),
            "peak_policy_mw": float(policy[1:].max()),
            # No schedule can bring the adjusted peak below max_t(N_t + idle power).
            "peak_floor_mw": float((net[1:] + idle_mw).max()),
            "idle_power_mw": float(idle_mw),
        }
    return out


def decoder_stats(infos: list[dict[str, Any]], actions: np.ndarray, sites: list[SiteConfig],
                  site_ids: list[str]) -> dict[str, float]:
    """What the decoder changes between the requested and the executed schedule."""
    n = len(infos)
    mw_per_unit = np.array([s.rated_power_mw * s.dynamic_power_fraction / s.compute_capacity for s in sites])
    adjusted = np.zeros(n, bool)
    service_capped_sites = np.zeros(n, int)
    top_service_share = np.zeros(n)
    moved_share = np.zeros(n)
    moved_mw = np.zeros(n)
    release_raised = np.zeros(n, bool)
    for t, (info, action) in enumerate(zip(infos, actions)):
        capacity = np.array([info["per_site"][s]["capacity"] for s in site_ids], float)
        service_total = float(info["service_arrived"])
        shares = softmax(action[:4])
        service_request = shares * service_total
        service_exec = np.array(info["service_allocation"], float)
        service_capped_sites[t] = int(np.sum(service_request > capacity + TOL))
        top_service_share[t] = float(shares.max())
        requested_release = float(info["batch_requested"])
        executed_release = float(info["batch_completed"])
        release_raised[t] = executed_release > requested_release + TOL
        batch_request = softmax(action[5:]) * executed_release
        batch_exec = np.array(info["batch_by_destination"], float)
        delta = (service_exec + batch_exec) - (service_request + batch_request)
        moved = float(np.clip(delta, 0.0, None).sum())
        hour_work = service_total + executed_release
        moved_share[t] = moved / hour_work if hour_work > 0 else 0.0
        # Site power that the moved work adds at the sites that receive it.
        moved_mw[t] = float((np.clip(delta, 0.0, None) * mw_per_unit).sum())
        adjusted[t] = bool(info["semantic_adjustment_applied"])
    capped = service_capped_sites > 0
    return {
        "hours": n,
        "adjusted_pct": 100 * float(adjusted.mean()),
        "service_capped_pct": 100 * float(capped.mean()),
        "service_capped_one_site_pct": 100 * float((service_capped_sites == 1).mean()),
        "top_service_share_when_capped_mean": float(top_service_share[capped].mean()) if capped.any() else 0.0,
        "release_raised_pct": 100 * float(release_raised.mean()),
        "work_moved_pct_of_hour_mean": 100 * float(moved_share.mean()),
        "work_moved_over_5pct_of_hour_pct": 100 * float((moved_share > 0.05).mean()),
        "power_moved_mw_mean": float(moved_mw.mean()),
        "power_moved_mw_when_adjusted_mean": float(moved_mw[adjusted].mean()) if adjusted.any() else 0.0,
    }


def evaluate(spec: str, split: str, month: str, inputs: MonthInputs, model_dirs: dict[str, Path],
             expected_J: dict[str, float] | None = None) -> dict[str, Any]:
    sites = site_configs(spec, split, month)
    _, nf_infos = no_flex_month(spec, split, month)
    nf_power = month_arrays(nf_infos, inputs.site_ids)["power"]
    runs = {}
    for seed, model_dir in model_dirs.items():
        infos, actions = replay(spec, split, month, model_dir, int(seed))
        power = month_arrays(infos, inputs.site_ids)["power"]
        J = inputs.recomputed_J(power)
        if expected_J is not None and abs(J - expected_J[seed]) > J_TOLERANCE:
            raise SystemExit(f"{spec} {month} seed {seed}: replayed J {J} differs from {expected_J[seed]}")
        runs[seed] = {
            "J": J,
            "markets": market_record(inputs, sites, nf_power, power),
            "decoder": decoder_stats(infos, actions, sites, inputs.site_ids),
        }
    return runs


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Means over month-runs of each market quantity and of the decoder statistics."""
    out: dict[str, Any] = {"markets": {}, "decoder": {}}
    for market in MARKETS:
        rows = [r["markets"][market] for r in records]
        entry: dict[str, Any] = {}
        for case in ("native", "no_flex", "policy"):
            for stat in ("mean_abs_ramp_mw", "steepest_ramp_mw"):
                entry[f"{case}_{stat}"] = float(np.mean([row[case][stat] for row in rows]))
        for key in ("peak_native_mw", "peak_no_flex_mw", "peak_policy_mw", "peak_floor_mw", "idle_power_mw"):
            entry[key] = float(np.mean([row[key] for row in rows]))
        squared_change = [100 * (row["policy"]["sum_squared_ramp_mw2"] / row["no_flex"]["sum_squared_ramp_mw2"] - 1)
                          for row in rows]
        entry["squared_ramp_change_pct"] = float(np.mean(squared_change))
        for stat in ("mean_abs_ramp_mw", "steepest_ramp_mw"):
            change = [row["policy"][stat] - row["no_flex"][stat] for row in rows]
            entry[f"{stat}_change"] = float(np.mean(change))
            entry[f"{stat}_change_range"] = [float(np.min(change)), float(np.max(change))]
            entry[f"{stat}_change_pct"] = float(np.mean(
                [100 * (row["policy"][stat] / row["no_flex"][stat] - 1) for row in rows]))
        added = [row["peak_no_flex_mw"] - row["peak_native_mw"] for row in rows]
        removed = [row["peak_no_flex_mw"] - row["peak_policy_mw"] for row in rows]
        removable = [row["peak_no_flex_mw"] - row["peak_floor_mw"] for row in rows]
        entry["fleet_added_peak_mw"] = float(np.mean(added))
        entry["peak_removed_mw"] = float(np.mean(removed))
        entry["peak_removed_range_mw"] = [float(np.min(removed)), float(np.max(removed))]
        entry["peak_removable_mw"] = float(np.mean(removable))
        entry["removed_share_of_added_pct"] = 100 * float(np.mean(removed) / np.mean(added))
        entry["removed_share_of_removable_pct"] = 100 * float(np.mean(removed) / np.mean(removable))
        out["markets"][market] = entry
    for key in records[0]["decoder"]:
        values = [r["decoder"][key] for r in records]
        out["decoder"][key] = {"mean": float(np.mean(values)), "min": float(np.min(values)),
                               "max": float(np.max(values))}
    return out


def figure_test_month(summary: dict[str, Any]) -> None:
    """Peaks and ramps of the test month in MW, relative to the market's own net load."""
    apply_style()
    labels = [MARKET_LABELS.get(m, m) for m in MARKETS]
    x = np.arange(len(MARKETS))
    fig, (left, right) = plt.subplots(1, 2, figsize=(PAGE_WIDTH_IN, 2.6))
    markets = summary["markets"]
    added = np.array([markets[m]["fleet_added_peak_mw"] for m in MARKETS])
    remaining = added - np.array([markets[m]["peak_removed_mw"] for m in MARKETS])
    floor = np.array([markets[m]["peak_floor_mw"] - markets[m]["peak_native_mw"] for m in MARKETS])
    width = 0.36
    left.bar(x - width / 2, added, width, color=MUTED, label="No flexibility")
    left.bar(x + width / 2, remaining, width, color=POLICY_COLOR, label="PPO policies")
    left.scatter(x, floor, marker="_", s=260, color=INK_2, linewidths=1.4, zorder=3,
                 label="Idle-power floor")
    left.set_xticks(x, labels)
    left.set_ylabel("Added to monthly peak (MW)")
    left.set_title("Monthly net-load peak above\nthe market's own peak")
    left.set_ylim(0, 1.32 * float(added.max()))
    left.legend(loc="upper left", ncols=3, fontsize=7, columnspacing=0.8, handlelength=1.2)
    avg = np.array([markets[m]["mean_abs_ramp_mw_change"] for m in MARKETS])
    steep = np.array([markets[m]["steepest_ramp_mw_change"] for m in MARKETS])
    right.bar(x - width / 2, avg, width, color=POLICY_COLOR, label="Average hourly ramp")
    right.bar(x + width / 2, steep, width, color="#9cc3ee", label="Steepest hourly ramp")
    right.axhline(0.0, color=AXIS, linewidth=0.8)
    right.set_xticks(x, labels)
    right.set_ylabel("Change from no flexibility (MW)")
    right.set_title("One-hour net-load ramps,\nPPO policies")
    right.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(w_pad=2.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "test_month_mw.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "test_month_mw.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    if "--figure-only" in sys.argv:
        saved = json.loads((OUT_DIR / "mw_results.json").read_text(encoding="utf-8"))
        figure_test_month(saved["summary"]["test"])
        return
    result: dict[str, Any] = {"test": {}, "cv": {}}
    month = TEST_MONTHS[0]
    inputs = MonthInputs("", month, spec=FINAL_SPEC, split="test")
    # Evaluated test scores, as written by scripts/build_test_month_results.py.
    test_results = json.loads((ROOT / "docs" / "figures" / "ppo_test" / "test_month.json").read_text(encoding="utf-8"))
    expected = {str(s): test_results["seeds"][str(s)]["policy_J"] for s in FINAL_SEEDS}
    model_dirs = {str(s): ROOT / "models" / ARTIFACT_NAMESPACE / "campaign" / "ppo" / f"seed-{s}" for s in FINAL_SEEDS}
    test_runs = evaluate(FINAL_SPEC, "test", month, inputs, model_dirs, expected)
    result["test"]["runs"] = test_runs
    print("test month done", flush=True)
    cv_records = []
    for fold, months in CV_FOLDS.items():
        cv_root = ROOT / "output" / ARTIFACT_NAMESPACE / "cv" / CV_TAG / fold
        for cv_month in months:
            inputs = MonthInputs(fold, cv_month)
            expected_cv = {}
            for seed in CV_SEEDS:
                summary = json.loads((cv_root / f"seed-{seed}" / "summary.json").read_text(encoding="utf-8"))
                final = summary["validation_by_interactions"][str(summary["effective_interactions"])]
                expected_cv[str(seed)] = final[cv_month]["policy_J"]
            dirs = {str(s): ROOT / "models" / ARTIFACT_NAMESPACE / "cv" / CV_TAG / fold / f"seed-{s}" for s in CV_SEEDS}
            runs = evaluate(f"cv-{fold}", "validation", cv_month, inputs, dirs, expected_cv)
            result["cv"][cv_month] = runs
            cv_records.extend(runs.values())
            print("cv", fold, cv_month, "done", flush=True)
    result["summary"] = {"test": summarize(list(test_runs.values())), "cv": summarize(cv_records)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mw_results.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n",
                                            encoding="utf-8", newline="\n")
    figure_test_month(result["summary"]["test"])
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
