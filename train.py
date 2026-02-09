"""Train a PPO agent on the Multi-DC environment using Stable-Baselines3.

Usage:
    python train.py --scenario env/scenarios/us_model.yaml --timesteps 200000
    python train.py --scenario env/scenarios/global_model.yaml --timesteps 200000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from env.data_loader import load_scenario
from env.multi_dc_env import MultiDCEnv


def make_env(scenario_path: Path, max_steps: int | None = None) -> MultiDCEnv:
    """Create a MultiDCEnv from a scenario config."""
    sites, power_model = load_scenario(scenario_path)
    return MultiDCEnv(
        sites=sites,
        power_model=power_model,
        max_steps=max_steps,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Multi-DC environment")
    parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Path to scenario YAML config",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max steps per episode (default: full trace length)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Hidden layer sizes (default: 128 128)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Steps per PPO update (default: 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained model (default: models/)",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run SB3 env checker before training",
    )
    args = parser.parse_args(argv)

    scenario_name = args.scenario.stem
    print(f"=== Training PPO on scenario: {scenario_name} ===")

    # Create environment
    env = make_env(args.scenario, args.max_steps)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of DCs: {env.n_dc}")
    print(f"Max steps per episode: {env.max_steps}")

    if args.check_env:
        print("Running environment check...")
        check_env(env, warn=True)
        print("Environment check passed.")

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        policy_kwargs=dict(net_arch=args.net_arch),
        verbose=1,
        seed=args.seed,
        tensorboard_log=None,  # Set to f"./tb_logs/{scenario_name}" if tensorboard is installed
    )

    print(f"\nTraining for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)

    # Save model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"ppo_{scenario_name}"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
