"""Train a DQN agent on the Multi-DC environment using Stable-Baselines3.

Following the CFWS (Zhao et al. 2024) methodology, this uses a Deep
Q-Network with a discretized action space.

Usage:
    python train_dqn.py --scenario env/scenarios/us_model.yaml --timesteps 500000
    python train_dqn.py --scenario env/scenarios/us_model.yaml --batch-mode --timesteps 500000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN

from env.data_loader import load_scenario
from env.discrete_wrapper import DiscretizedMultiDCEnv
from env.multi_dc_env import MultiDCEnv


def make_env(
    scenario_path: Path,
    max_steps: int | None = None,
    batch_enabled: bool = False,
    flexibility_factor: float = 1.0,
    deadline_penalty_weight: float = 2.0,
    urgency_horizon_steps: int = 12,
    memory_enabled: bool = False,
    dynamic_arrivals: bool = True,
    seed: int = 42,
    granularity: int = 5,
) -> DiscretizedMultiDCEnv:
    """Create a discretized MultiDCEnv from a scenario config."""
    sites, power_model, batch_config = load_scenario(
        scenario_path,
        batch_enabled=batch_enabled,
        dynamic_arrivals=dynamic_arrivals,
        seed=seed,
    )

    ff = flexibility_factor
    dp = deadline_penalty_weight
    uh = urgency_horizon_steps
    if batch_enabled and batch_config:
        ff = batch_config.get("flexibility_factor", ff)
        dp = batch_config.get("deadline_penalty_weight", dp)
        uh = batch_config.get("urgency_horizon_steps", uh)

    base_env = MultiDCEnv(
        sites=sites,
        power_model=power_model,
        max_steps=max_steps,
        batch_enabled=batch_enabled,
        flexibility_factor=ff,
        deadline_penalty_weight=dp,
        urgency_horizon_steps=uh,
        memory_enabled=memory_enabled,
    )

    return DiscretizedMultiDCEnv(base_env, granularity=granularity)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train DQN on Multi-DC environment")
    parser.add_argument(
        "--scenario", type=Path, required=True, help="Path to scenario YAML config"
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Max steps per episode"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer sizes",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=100_000, help="Replay buffer size"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.3,
        help="Fraction of timesteps for epsilon decay",
    )
    parser.add_argument(
        "--exploration-final-eps",
        type=float,
        default=0.05,
        help="Final exploration epsilon",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1000,
        help="Steps between target network updates",
    )
    parser.add_argument(
        "--granularity",
        type=int,
        default=5,
        help="Action discretization granularity (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models"), help="Model save directory"
    )
    parser.add_argument("--batch-mode", action="store_true", help="Enable batch mode")
    parser.add_argument(
        "--flexibility-factor", type=float, default=1.0, help="Deadline flexibility"
    )
    parser.add_argument(
        "--deadline-penalty", type=float, default=2.0, help="Deadline penalty weight"
    )
    parser.add_argument(
        "--memory", action="store_true", help="Enable memory constraints"
    )
    parser.add_argument(
        "--no-dynamic-arrivals",
        action="store_true",
        help="Disable dynamic batch arrivals",
    )
    args = parser.parse_args(argv)

    scenario_name = args.scenario.stem
    if args.batch_mode:
        scenario_name += "_batch"
    print(f"=== Training DQN on scenario: {scenario_name} ===")

    env = make_env(
        args.scenario,
        args.max_steps,
        batch_enabled=args.batch_mode,
        flexibility_factor=args.flexibility_factor,
        deadline_penalty_weight=args.deadline_penalty,
        memory_enabled=args.memory,
        dynamic_arrivals=not args.no_dynamic_arrivals,
        seed=args.seed,
        granularity=args.granularity,
    )
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space} ({env.n_actions} discrete actions)")
    print(f"Number of DCs: {env.env.n_dc}")
    print(f"Max steps per episode: {env.env.max_steps}")

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        target_update_interval=args.target_update_interval,
        policy_kwargs=dict(net_arch=args.net_arch),
        verbose=1,
        seed=args.seed,
        tensorboard_log=None,
    )

    print(f"\nTraining for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"dqn_{scenario_name}"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
