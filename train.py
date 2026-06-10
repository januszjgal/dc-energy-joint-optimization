"""Train a PPO agent on the Multi-DC environment using Stable-Baselines3.

Usage:
    python train.py --scenario env/scenarios/us_model.yaml --timesteps 200000
    python train.py --scenario env/scenarios/us_model.yaml --batch-mode --timesteps 200000
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
    peak_penalty_weight: float = 0.0,
    burst_aware: bool = False,
    batch_spatial_routing: bool = True,
) -> MultiDCEnv:
    """Create a MultiDCEnv from a scenario config."""
    sites, power_model, batch_config = load_scenario(
        scenario_path,
        batch_enabled=batch_enabled,
        dynamic_arrivals=dynamic_arrivals,
        seed=seed,
    )

    # Merge YAML batch config with CLI overrides (CLI takes precedence)
    ff = flexibility_factor
    dp = deadline_penalty_weight
    uh = urgency_horizon_steps
    if batch_enabled and batch_config:
        ff = batch_config.get("flexibility_factor", ff)
        dp = batch_config.get("deadline_penalty_weight", dp)
        uh = batch_config.get("urgency_horizon_steps", uh)

    return MultiDCEnv(
        sites=sites,
        power_model=power_model,
        max_steps=max_steps,
        batch_enabled=batch_enabled,
        flexibility_factor=ff,
        deadline_penalty_weight=dp,
        urgency_horizon_steps=uh,
        memory_enabled=memory_enabled,
        peak_penalty_weight=peak_penalty_weight,
        burst_aware=burst_aware,
        batch_spatial_routing=batch_spatial_routing,
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
    # Batch scheduling arguments
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Enable batch scheduling (temporal + spatial optimization)",
    )
    parser.add_argument(
        "--flexibility-factor",
        type=float,
        default=1.0,
        help="Deadline flexibility factor (default: 1.0). Higher = more slack.",
    )
    parser.add_argument(
        "--deadline-penalty",
        type=float,
        default=2.0,
        help="Penalty weight for batch deadline violations (default: 2.0)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable memory as a constraint dimension",
    )
    parser.add_argument(
        "--no-dynamic-arrivals",
        action="store_true",
        help="Disable dynamic batch arrivals (use static fraction split)",
    )
    parser.add_argument(
        "--peak-penalty-weight",
        type=float,
        default=0.0,
        help="Weight α on the grid demand-smoothing term: "
             "α × grid_mw² × net_demand_normalized[t]. Penalizes load "
             "concentrated during peak grid stress (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--burst-aware",
        action="store_true",
        help="Augment observation with per-DC burst_severity = current_arrival / "
             "rolling_24h_mean_arrival (batch mode only). Adds 1 dim per DC.",
    )
    parser.add_argument(
        "--no-batch-spatial-routing",
        dest="batch_spatial_routing",
        action="store_false",
        help="Disable spatial routing of drained batch work (batch mode only). "
             "Drained batch executes at its home DC; action space drops 3N->2N.",
    )
    args = parser.parse_args(argv)

    scenario_name = args.scenario.stem
    if args.batch_mode:
        scenario_name += "_batch"
    if args.burst_aware:
        scenario_name += "_burst"
    print(f"=== Training PPO on scenario: {scenario_name} ===")

    # Create environment
    env = make_env(
        args.scenario,
        args.max_steps,
        batch_enabled=args.batch_mode,
        flexibility_factor=args.flexibility_factor,
        deadline_penalty_weight=args.deadline_penalty,
        memory_enabled=args.memory,
        dynamic_arrivals=not args.no_dynamic_arrivals,
        seed=args.seed,
        peak_penalty_weight=args.peak_penalty_weight,
        burst_aware=args.burst_aware,
        batch_spatial_routing=args.batch_spatial_routing,
    )
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of DCs: {env.n_dc}")
    print(f"Max steps per episode: {env.max_steps}")
    if args.batch_mode:
        for site in env.sites:
            print(
                f"  {site.name}: batch_fraction={site.batch_fraction:.3f}, "
                f"mean_dur={site.batch_mean_duration_sec:.0f}s"
            )

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
        tensorboard_log=None,
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
