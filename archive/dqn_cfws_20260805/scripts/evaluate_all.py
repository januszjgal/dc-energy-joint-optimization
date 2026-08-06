"""Run evaluate.py for all 4 scenarios
(US/Global x spatial-only/spatial+temporal) once
the full training sweep has completed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = [
    # (scenario, batch, ppo_model_stem, dqn_model_stem, dqn_flatidx_model_stem)
    ("env/scenarios/us_model.yaml",     False, "ppo_us_model",            "dqn_us_model",            "dqn_us_model_flatidx"),
    ("env/scenarios/us_model.yaml",     True,  "ppo_us_model_batch",      "dqn_us_model_batch",      "dqn_us_model_batch_flatidx"),
    ("env/scenarios/global_model.yaml", False, "ppo_global_model",        "dqn_global_model",        "dqn_global_model_flatidx"),
    ("env/scenarios/global_model.yaml", True,  "ppo_global_model_batch",  "dqn_global_model_batch",  "dqn_global_model_batch_flatidx"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.015)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--demand-charge-rate", type=float, default=0.0)
    parser.add_argument("--demand-charge-period-steps", type=int, default=None)
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / ".venv" / "Scripts" / "python.exe",
    )
    args = parser.parse_args()

    for scenario, batch, ppo_stem, dqn_stem, dqn_flatidx_stem in CONFIGS:
        suffix = "_demand_charge" if args.demand_charge_rate > 0.0 else ""
        ppo_path = args.models_dir / f"{ppo_stem}{suffix}.zip"
        dqn_path = args.models_dir / f"{dqn_stem}{suffix}.zip"
        dqn_flatidx_path = (
            args.models_dir / f"{dqn_flatidx_stem}{suffix}.zip"
        )
        mode = "spatial+temporal" if batch else "spatial-only"
        if not ppo_path.exists():
            print(f"SKIP {scenario} mode={mode}: missing {ppo_path}")
            continue

        cmd = [
            str(args.python),
            "evaluate.py",
            "--scenario", scenario,
            "--model", str(ppo_path),
            "--peak-penalty-weight", str(args.alpha),
            "--output-dir", str(args.output_dir),
        ]
        if args.demand_charge_rate > 0.0:
            cmd.extend([
                "--demand-charge-rate",
                str(args.demand_charge_rate),
            ])
        if args.demand_charge_period_steps is not None:
            cmd.extend([
                "--demand-charge-period-steps",
                str(args.demand_charge_period_steps),
            ])
        if batch:
            cmd.append("--batch-mode")
        if dqn_path.exists():
            cmd.extend(["--dqn-model", str(dqn_path)])
        if dqn_flatidx_path.exists():
            cmd.extend(["--dqn-flatidx-model", str(dqn_flatidx_path)])

        print(f"\n>>> Evaluating {scenario} mode={mode}")
        print(f"    cmd: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=ROOT, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        if proc.returncode != 0:
            print(f"    FAILED (exit={proc.returncode})")
            sys.exit(1)
        print("    OK")


if __name__ == "__main__":
    main()
