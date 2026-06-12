"""Peer-review response campaign (M1): multi-seed training + evaluation + stats.

5 seeds x {PPO, DQN routing-grid, DQN flat-idx} x {US, Global} x {spatial-only,
batch} = 60 training runs on the settled environment (corrected <=115 tier
curves, per-cell power, deadline-preserving queueing, w=250, backlog 25,
action bound +/-3), followed by deterministic evaluation of every seed model
plus the heuristic baselines, and seed-level statistics (mean +/- std, paired
PPO-vs-best-DQN differences, sign/rank tests, bootstrap CIs).

Resumable: training is skipped when the model zip already exists; evaluation
re-runs cheaply. Outputs:
    models/review/s{seed}/...                    trained models
    output/review_campaign/{config}.json         per-config eval summaries
    output/review_campaign/stats.json            aggregated statistics
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PY = ROOT / ".venv" / "Scripts" / "python.exe"
SEEDS = [101, 102, 103, 104, 105]
TIMESTEPS = 500_000
ALPHA = 0.015

CONFIGS = [
    # (config key, scenario, batch_mode)
    ("us_spatial", "env/scenarios/us_model.yaml", False),
    ("us_batch", "env/scenarios/us_model.yaml", True),
    ("global_spatial", "env/scenarios/global_model.yaml", False),
    ("global_batch", "env/scenarios/global_model.yaml", True),
]
ALGOS = _ARGS.algos if _ARGS.algos else ["ppo", "dqn", "flatidx"]

_ap = argparse.ArgumentParser()
_ap.add_argument("--tag", default="",
                 help="suffix for model/output dirs (e.g. 'ctx' -> models/review_ctx)")
_ap.add_argument("--algos", nargs="*", default=None,
                 help="subset of {ppo,dqn,flatidx} (default: all)")
_ap.add_argument("--domain-rand", action="store_true",
                 help="pass --domain-rand to PPO trainings (train.py only)")
_ARGS = _ap.parse_args()
_SUF = f"_{_ARGS.tag}" if _ARGS.tag else ""
MODEL_DIR = ROOT / "models" / f"review{_SUF}"
OUT_DIR = ROOT / "output" / f"review_campaign{_SUF}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def model_path(algo: str, cfg_key: str, scenario: str, batch: bool, seed: int) -> Path:
    stem = Path(scenario).stem
    name = {"ppo": f"ppo_{stem}", "dqn": f"dqn_{stem}", "flatidx": f"dqn_{stem}"}[algo]
    if batch:
        name += "_batch"
    if algo == "flatidx":
        name += "_flatidx"
    return MODEL_DIR / f"s{seed}" / f"{name}.zip"


def train_one(algo: str, cfg_key: str, scenario: str, batch: bool, seed: int) -> None:
    mp = model_path(algo, cfg_key, scenario, batch, seed)
    if mp.exists():
        print(f"[skip] {mp.relative_to(ROOT)} exists")
        return
    mp.parent.mkdir(parents=True, exist_ok=True)
    script = "train.py" if algo == "ppo" else "train_dqn.py"
    cmd = [str(PY), script, "--scenario", scenario,
           "--timesteps", str(TIMESTEPS), "--peak-penalty-weight", str(ALPHA),
           "--seed", str(seed), "--output-dir", str(mp.parent)]
    if batch:
        cmd.append("--batch-mode")
    if algo == "flatidx":
        cmd += ["--action-scheme", "cfws-style"]
    if _ARGS.domain_rand and algo == "ppo":
        cmd.append("--domain-rand")
    t0 = time.time()
    print(f"[train] {algo} {cfg_key} seed={seed} ...", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError(f"training failed: {algo} {cfg_key} s{seed}")
    print(f"        done in {(time.time()-t0)/60:.1f} min", flush=True)


def evaluate_config(cfg_key: str, scenario: str, batch: bool) -> dict:
    """Baselines once + every (algo, seed) model, deterministic episodes."""
    from stable_baselines3 import DQN, PPO

    from baselines import ALL_BASELINES
    from env.cfws_style_wrapper import CFWSStyleDiscretizedEnv
    from env.discrete_wrapper import DiscretizedMultiDCEnv
    from evaluate import _make_env, compute_summary, run_episode

    def fresh_env():
        return _make_env(Path(scenario), batch_enabled=batch,
                         peak_penalty_weight=ALPHA)

    results: dict = {"baselines": {}, "agents": {a: {} for a in ALGOS}}
    for cls in ALL_BASELINES:
        pol = cls()
        _, hist = run_episode(fresh_env(), pol.predict, is_sb3=False)
        results["baselines"][pol.name] = compute_summary(hist, batch_enabled=batch)
        print(f"  [eval] baseline {pol.name}: "
              f"{results['baselines'][pol.name]['total_cost']:,.0f}", flush=True)

    for algo, seed in itertools.product(ALGOS, SEEDS):
        mp = model_path(algo, cfg_key, scenario, batch, seed)
        if not mp.exists():
            print(f"  [eval] MISSING {mp}")
            continue
        env = fresh_env()
        if algo == "ppo":
            model = PPO.load(str(mp))
        else:
            model = DQN.load(str(mp))
            env = (CFWSStyleDiscretizedEnv(env) if algo == "flatidx"
                   else DiscretizedMultiDCEnv(env))
        _, hist = run_episode(env, model.predict, is_sb3=True)
        results["agents"][algo][str(seed)] = compute_summary(hist, batch_enabled=batch)
        print(f"  [eval] {algo} s{seed}: "
              f"{results['agents'][algo][str(seed)]['total_cost']:,.0f}", flush=True)

    with open(OUT_DIR / f"{cfg_key}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 20_000, alpha: float = 0.05):
    rng = np.random.default_rng(0)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def aggregate_stats(all_results: dict) -> dict:
    from scipy import stats as sp

    out: dict = {}
    for cfg_key, res in all_results.items():
        costs = {a: np.array([res["agents"][a][str(s)]["total_cost"]
                              for s in SEEDS if str(s) in res["agents"][a]])
                 for a in ALGOS}
        sq = res["baselines"]["Status Quo (local, no deferral)"]["total_cost"]
        entry: dict = {"status_quo": sq, "per_algo": {}}
        for a, c in costs.items():
            if len(c) == 0:
                continue
            entry["per_algo"][a] = {
                "n": int(len(c)),
                "mean": float(c.mean()), "std": float(c.std(ddof=1)) if len(c) > 1 else 0.0,
                "min": float(c.min()), "max": float(c.max()),
                "mean_savings_vs_status_quo_pct": float(100 * (sq - c.mean()) / sq),
                "worst_seed_savings_vs_status_quo_pct": float(100 * (sq - c.max()) / sq),
            }
        # paired PPO vs best-DQN-per-seed (paired by seed)
        if len(costs["ppo"]) and (len(costs["dqn"]) or len(costs["flatidx"])):
            best_dqn = np.minimum(
                costs["dqn"] if len(costs["dqn"]) else np.inf,
                costs["flatidx"] if len(costs["flatidx"]) else np.inf,
            )
            diffs = best_dqn - costs["ppo"]  # >0 means PPO cheaper
            wilcoxon_p = None
            if len(diffs) >= 5 and not np.allclose(diffs, 0):
                try:
                    wilcoxon_p = float(sp.wilcoxon(diffs, alternative="greater").pvalue)
                except ValueError:
                    wilcoxon_p = None
            entry["ppo_vs_best_dqn"] = {
                "paired_diffs": diffs.tolist(),
                "mean_diff": float(diffs.mean()),
                "ppo_wins_seeds": int((diffs > 0).sum()),
                "sign_test_p_one_sided": float(sp.binomtest(
                    int((diffs > 0).sum()), len(diffs), 0.5,
                    alternative="greater").pvalue),
                "wilcoxon_p_one_sided": wilcoxon_p,
                "bootstrap_95ci_mean_diff": bootstrap_ci(diffs),
            }
        out[cfg_key] = entry
    return out


def main() -> None:
    t0 = time.time()
    print(f"Review campaign start: {time.ctime()}  "
          f"({len(SEEDS)} seeds x {len(ALGOS)} algos x {len(CONFIGS)} configs)")
    for cfg_key, scenario, batch in CONFIGS:
        for algo in ALGOS:
            for seed in SEEDS:
                train_one(algo, cfg_key, scenario, batch, seed)
    print(f"\nAll training done at {(time.time()-t0)/3600:.2f} h. Evaluating...")
    all_results = {}
    for cfg_key, scenario, batch in CONFIGS:
        print(f"[eval] config {cfg_key}")
        all_results[cfg_key] = evaluate_config(cfg_key, scenario, batch)
    stats = aggregate_stats(all_results)
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2)[:4000])
    print(f"\nCampaign finished in {(time.time()-t0)/3600:.2f} h: {time.ctime()}")


if __name__ == "__main__":
    main()
