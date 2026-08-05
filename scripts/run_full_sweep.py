"""Sequentially train all 8 RL models under the new demand-smoothing formulation.

Sweep:
  - {PPO, DQN} x {US, Global} x {spatial, spatial+temporal}  -> 8 runs
  - peak_penalty_weight = 0.015 across all (calibrated so peak penalty is
    ~15-20% of total cost on the Round Robin baseline; see
    scripts/smoke_test_env.py for the calibration probe)

Outputs go to models/<algo>_<scenario>[_batch].zip. Per-run stdout/stderr
captured to logs/sweep_<run_label>.log. A consolidated summary is written
at logs/sweep_summary.txt when the sweep finishes.

Usage:
    python scripts/run_full_sweep.py [--timesteps 500000] [--alpha 0.015]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# Each run: (algorithm, scenario_yaml, label, extra_flags)
SWEEP = [
    # PPO - US
    ("ppo", "env/scenarios/us_model.yaml",     "ppo_us_spatial",              []),
    ("ppo", "env/scenarios/us_model.yaml",     "ppo_us_spatial_temporal",     ["--batch-mode"]),
    # PPO - Global
    ("ppo", "env/scenarios/global_model.yaml", "ppo_global_spatial",          []),
    ("ppo", "env/scenarios/global_model.yaml", "ppo_global_spatial_temporal", ["--batch-mode"]),
    # DQN - US
    ("dqn", "env/scenarios/us_model.yaml",     "dqn_us_spatial",              []),
    ("dqn", "env/scenarios/us_model.yaml",     "dqn_us_spatial_temporal",     ["--batch-mode"]),
    # DQN - Global
    ("dqn", "env/scenarios/global_model.yaml", "dqn_global_spatial",          []),
    ("dqn", "env/scenarios/global_model.yaml", "dqn_global_spatial_temporal", ["--batch-mode"]),
]


def run_one(
    algo: str,
    scenario: str,
    label: str,
    extra: list[str],
    timesteps: int,
    alpha: float,
    python: Path,
) -> tuple[bool, float]:
    script = "train.py" if algo == "ppo" else "train_dqn.py"
    cmd = [
        str(python),
        script,
        "--scenario", scenario,
        "--timesteps", str(timesteps),
        "--peak-penalty-weight", str(alpha),
        *extra,
    ]
    log_path = LOG_DIR / f"sweep_{label}.log"
    print(f"\n>>> Starting [{label}]")
    print(f"    cmd: {' '.join(cmd)}")
    print(f"    log: {log_path}")
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# {label}\n# cmd: {' '.join(cmd)}\n# started: {time.ctime(start)}\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        )
    elapsed = time.time() - start
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAIL (exit={proc.returncode})"
    print(f"    {status}  in {elapsed/60:.1f} min")
    return ok, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--alpha", type=float, default=0.015)
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / ".venv" / "Scripts" / "python.exe",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"Full sweep: {len(SWEEP)} runs at {args.timesteps:,} timesteps each")
    print(f"alpha (peak_penalty_weight) = {args.alpha}")
    print(f"python = {args.python}")
    print("=" * 70)

    results = []
    total_start = time.time()
    for algo, scenario, label, extra in SWEEP:
        ok, elapsed = run_one(
            algo, scenario, label, extra, args.timesteps, args.alpha, args.python
        )
        results.append((label, ok, elapsed))

    total_elapsed = time.time() - total_start
    summary_path = LOG_DIR / "sweep_summary.txt"
    lines = [
        f"# Full training sweep summary",
        f"# timesteps = {args.timesteps}",
        f"# alpha = {args.alpha}",
        f"# total elapsed: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} h)",
        "",
        f"{'label':<28} {'status':<8} {'elapsed (min)':>14}",
        "-" * 52,
    ]
    for label, ok, elapsed in results:
        lines.append(f"{label:<28} {'OK' if ok else 'FAIL':<8} {elapsed/60:>14.1f}")
    summary = "\n".join(lines) + "\n"
    summary_path.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Summary written to {summary_path}")

    if any(not ok for _, ok, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
