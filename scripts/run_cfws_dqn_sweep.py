"""Train DQN with the CFWS-style flattened-index action scheme across all 4 configs.

Same training budget as the main sweep (500K steps each, alpha=0.015) so
the results are directly comparable to the existing routing-grid DQN.
Logs to logs/sweep_*_flatidx.log; saves models to models/dqn_*_flatidx.zip.
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
    # (scenario_yaml, label, extra_flags)
    ("env/scenarios/us_model.yaml",     "dqn_us_batch_flatidx",     ["--batch-mode"]),
    ("env/scenarios/us_model.yaml",     "dqn_us_legacy_flatidx",    []),
    ("env/scenarios/global_model.yaml", "dqn_global_batch_flatidx", ["--batch-mode"]),
    ("env/scenarios/global_model.yaml", "dqn_global_legacy_flatidx",[]),
]


def run_one(
    scenario: str, label: str, extra: list[str],
    timesteps: int, alpha: float, python: Path,
) -> tuple[bool, float]:
    cmd = [
        str(python), "train_dqn.py",
        "--scenario", scenario,
        "--timesteps", str(timesteps),
        "--peak-penalty-weight", str(alpha),
        "--action-scheme", "cfws-style",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--alpha", type=float, default=0.015)
    parser.add_argument(
        "--python", type=Path,
        default=ROOT / ".venv" / "Scripts" / "python.exe",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"CFWS-style DQN sweep: {len(SWEEP)} runs at {args.timesteps:,} steps each")
    print(f"alpha (peak_penalty_weight) = {args.alpha}")
    print(f"action scheme = cfws-style (48-action flattened index)")
    print("=" * 70)

    results = []
    total_start = time.time()
    for scenario, label, extra in SWEEP:
        ok, elapsed = run_one(
            scenario, label, extra, args.timesteps, args.alpha, args.python
        )
        results.append((label, ok, elapsed))

    total_elapsed = time.time() - total_start
    summary_path = LOG_DIR / "sweep_cfws_flatidx_summary.txt"
    lines = [
        "# CFWS-style DQN sweep summary",
        f"# timesteps = {args.timesteps}",
        f"# alpha = {args.alpha}",
        f"# total elapsed: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} h)",
        "",
        f"{'label':<32} {'status':<8} {'elapsed (min)':>14}",
        "-" * 56,
    ]
    for label, ok, elapsed in results:
        lines.append(f"{label:<32} {'OK' if ok else 'FAIL':<8} {elapsed/60:>14.1f}")
    summary = "\n".join(lines) + "\n"
    summary_path.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Summary written to {summary_path}")

    if any(not ok for _, ok, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
