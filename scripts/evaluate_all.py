"""Run evaluate.py for all 4 scenarios (US/Global x legacy/batch) once
the full training sweep has completed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = [
    # (scenario, batch, ppo_model_stem, dqn_model_stem)
    ("env/scenarios/us_model.yaml",     False, "ppo_us_model",        "dqn_us_model"),
    ("env/scenarios/us_model.yaml",     True,  "ppo_us_model_batch",  "dqn_us_model_batch"),
    ("env/scenarios/global_model.yaml", False, "ppo_global_model",        "dqn_global_model"),
    ("env/scenarios/global_model.yaml", True,  "ppo_global_model_batch",  "dqn_global_model_batch"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.015)
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / ".venv" / "Scripts" / "python.exe",
    )
    args = parser.parse_args()

    for scenario, batch, ppo_stem, dqn_stem in CONFIGS:
        ppo_path = ROOT / "models" / f"{ppo_stem}.zip"
        dqn_path = ROOT / "models" / f"{dqn_stem}.zip"
        if not ppo_path.exists():
            print(f"SKIP {scenario} batch={batch}: missing {ppo_path}")
            continue

        cmd = [
            str(args.python),
            "evaluate.py",
            "--scenario", scenario,
            "--model", str(ppo_path),
            "--peak-penalty-weight", str(args.alpha),
        ]
        if batch:
            cmd.append("--batch-mode")
        if dqn_path.exists():
            cmd.extend(["--dqn-model", str(dqn_path)])

        print(f"\n>>> Evaluating {scenario} batch={batch}")
        print(f"    cmd: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=ROOT, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        if proc.returncode != 0:
            print(f"    FAILED (exit={proc.returncode})")
            sys.exit(1)
        print("    OK")


if __name__ == "__main__":
    main()
