"""Run fixture jobs or integrated v6 pure-RL jobs without launching the final campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ramp_rl.evaluation import evaluate_checkpoint  # noqa: E402
from ramp_rl.campaign import plan_stage  # noqa: E402
from ramp_rl.runner import load_factory, run_training  # noqa: E402
from ramp_rl.schema import DEFAULT_PROTOCOL_PATH, load_protocol  # noqa: E402


def _default_window(split: str) -> str:
    handoff = ROOT / "output" / "energy_model_v3" / "ramp_v6" / "factory_manifest.json"
    if handoff.is_file():
        payload = json.loads(handoff.read_text(encoding="utf-8"))
        windows = sorted(payload["windows"][split])
        if windows:
            return windows[0]
    return "m-07-sealed-0000" if split == "validation" else "m-09-sealed-0000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "train", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--env-factory", default="env.ramp_v6.factory:make_fixture_env"
    )
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--epsilon-pct", type=float)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "ramp_rl_v6" / "fixture")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--window", action="append")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fixture-profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    if args.command == "plan":
        plan = {
            stage: [job.as_dict() for job in plan_stage(protocol, stage)]
            for stage in ("screen", "confirmation", "extension")
        }
        print(json.dumps({"campaign": protocol["campaign"], "jobs": plan}, indent=2, sort_keys=True))
        return
    factory = load_factory(args.env_factory)
    if args.command == "train":
        manifest = run_training(
            factory=factory,
            protocol=protocol,
            algorithm=args.algorithm,
            seed=args.seed,
            target_timesteps=args.timesteps,
            output_dir=args.output,
            n_envs=args.n_envs,
            resume=not args.no_resume,
            fixture_profile=args.fixture_profile,
            epsilon_pct=args.epsilon_pct,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    windows = args.window or [_default_window(args.split)]
    summary = evaluate_checkpoint(
        factory=factory,
        algorithm=args.algorithm,
        checkpoint_dir=args.output,
        split=args.split,
        seeds=[args.seed],
        windows=windows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
