"""Step 2 burst-aware sweep: train 4 burst-aware + memory-enabled models.

Variants:
  - PPO US batch    (burst + memory)
  - PPO Global batch (burst + memory)
  - DQN-flatidx US batch    (burst + memory)
  - DQN-flatidx Global batch (burst + memory)

Memory is enabled (--memory) for these runs, justified by the
check_memory_binding.py diagnostic showing memory does not bind in our
env (diff vs memory-disabled = 0.0000%). Burst-aware augmentation adds
the per-DC `burst_severity = arrival[t] / rolling_24h_mean` observation
feature (motivated by Step 1 finding that RL agents' optimization
signal concentrates in burst windows; see §7-burst).

Same hyperparams + step budget as the main sweep so results are
directly comparable.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


SWEEP = [
    # (script, scenario_yaml, label, extra_flags)
    ("train.py",     "env/scenarios/us_model.yaml",     "ppo_us_batch_burst",     ["--batch-mode", "--burst-aware", "--memory"]),
    ("train.py",     "env/scenarios/global_model.yaml", "ppo_global_batch_burst", ["--batch-mode", "--burst-aware", "--memory"]),
    ("train_dqn.py", "env/scenarios/us_model.yaml",     "dqn_us_batch_flatidx_burst",     ["--batch-mode", "--burst-aware", "--memory", "--action-scheme", "cfws-style"]),
    ("train_dqn.py", "env/scenarios/global_model.yaml", "dqn_global_batch_flatidx_burst", ["--batch-mode", "--burst-aware", "--memory", "--action-scheme", "cfws-style"]),
]


def run_one(script, scenario, label, extra, timesteps, alpha, python):
    cmd = [
        str(python), script,
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
        logf.write(
            f"# {label}\n# cmd: {' '.join(cmd)}\n"
            f"# started: {time.ctime(start)}\n\n"
        )
        logf.flush()
        proc = subprocess.run(
            cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    elapsed = time.time() - start
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAIL (exit={proc.returncode})"
    print(f"    {status}  in {elapsed/60:.1f} min")
    return ok, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--alpha", type=float, default=0.015)
    parser.add_argument(
        "--python", type=Path,
        default=ROOT / ".venv" / "Scripts" / "python.exe",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"Burst-aware + memory sweep: {len(SWEEP)} runs at {args.timesteps:,} steps each")
    print(f"alpha (peak_penalty_weight) = {args.alpha}")
    print("burst_aware = True, memory_enabled = True")
    print("=" * 70)

    results = []
    total_start = time.time()
    for script, scenario, label, extra in SWEEP:
        ok, elapsed = run_one(
            script, scenario, label, extra,
            args.timesteps, args.alpha, args.python,
        )
        results.append((label, ok, elapsed))

    total_elapsed = time.time() - total_start
    summary_path = LOG_DIR / "sweep_burst_summary.txt"
    lines = [
        "# Burst-aware (+ memory) sweep summary",
        f"# timesteps = {args.timesteps}",
        f"# alpha = {args.alpha}",
        f"# total elapsed: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} h)",
        "",
        f"{'label':<40} {'status':<8} {'elapsed (min)':>14}",
        "-" * 64,
    ]
    for label, ok, elapsed in results:
        lines.append(f"{label:<40} {'OK' if ok else 'FAIL':<8} {elapsed/60:>14.1f}")
    summary = "\n".join(lines) + "\n"
    summary_path.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Summary written to {summary_path}")

    if any(not ok for _, ok, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
