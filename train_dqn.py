"""Train a DQN agent on the Multi-DC environment using Stable-Baselines3.

Two discretization schemes are supported via --action-scheme:

  routing-grid (default):
      Stock SB3 DQN over a 759-action routing-fraction grid (4 DCs × 5
      levels deduped to 253 routing actions × 3 drain levels). See
      env/discrete_wrapper.py.

  cfws-style:
      48-action CFWS-style flattened-index scheme adapted to our env
      (Zhao et al. 2025, IEEE TSC 10(1)). Each action decodes to
      (src_dc, dst_dc, drain_level) via division/modulo; src==dst means
      "uniform allocation", else "migrate 15% from src to dst". See
      env/cfws_style_wrapper.py and thesis_overview.md §8.6.

Usage:
    python train_dqn.py --scenario env/scenarios/us_model.yaml --timesteps 500000
    python train_dqn.py --scenario env/scenarios/us_model.yaml --batch-mode --timesteps 500000
    python train_dqn.py --scenario env/scenarios/us_model.yaml --batch-mode \
        --action-scheme cfws-style --timesteps 500000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN

from env.cfws_style_wrapper import CFWSStyleDiscretizedEnv
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
    peak_penalty_weight: float = 0.0,
    demand_charge_rate: float = 0.0,
    demand_charge_period_steps: int = 288,
    action_scheme: str = "routing-grid",
    burst_aware: bool = False,
    batch_spatial_routing: bool = True,
) -> gym.Wrapper:
    """Create a discretized MultiDCEnv from a scenario config.

    action_scheme:
      'routing-grid'  -> DiscretizedMultiDCEnv (759 actions)
      'cfws-style'    -> CFWSStyleDiscretizedEnv (48 actions)
    """
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
        peak_penalty_weight=peak_penalty_weight,
        demand_charge_rate=demand_charge_rate,
        demand_charge_period_steps=demand_charge_period_steps,
        burst_aware=burst_aware,
        batch_spatial_routing=batch_spatial_routing,
    )

    if action_scheme == "cfws-style":
        return CFWSStyleDiscretizedEnv(base_env)
    if action_scheme == "routing-grid":
        return DiscretizedMultiDCEnv(base_env, granularity=granularity)
    raise ValueError(f"unknown action_scheme: {action_scheme}")


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
    parser.add_argument(
        "--peak-penalty-weight",
        type=float,
        default=0.0,
        help="Weight on peak-contribution penalty (default: 0.0)",
    )
    parser.add_argument(
        "--demand-charge-rate",
        type=float,
        default=0.0,
        help="Demand charge in $/kW-month, billed on the highest demand "
             "interval of each billing period (the real commercial tariff "
             "term). Default 0.0 = not in the reward; evaluation reports the "
             "charge at a reference rate either way",
    )
    parser.add_argument(
        "--demand-charge-period-steps",
        type=int,
        default=288,
        help="Billing window in steps (default 288 = daily). Monthly (8640) "
             "is the realistic tariff but far exceeds the gamma=0.99 credit "
             "horizon (~100 steps)",
    )
    parser.add_argument(
        "--action-scheme",
        choices=["routing-grid", "cfws-style"],
        default="routing-grid",
        help="Action discretization: routing-grid (759 actions, default) or "
             "cfws-style (48-action flattened-index, CFWS Zhao 2025 analog).",
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
             "Drained batch executes at its home DC. DQN ties batch routing to "
             "service routing, so the discrete action count is unchanged.",
    )
    args = parser.parse_args(argv)

    scenario_name = args.scenario.stem
    if args.batch_mode:
        scenario_name += "_batch"
    if args.action_scheme == "cfws-style":
        scenario_name += "_flatidx"
    if args.burst_aware:
        scenario_name += "_burst"
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
        peak_penalty_weight=args.peak_penalty_weight,
        demand_charge_rate=args.demand_charge_rate,
        demand_charge_period_steps=args.demand_charge_period_steps,
        action_scheme=args.action_scheme,
        burst_aware=args.burst_aware,
        batch_spatial_routing=args.batch_spatial_routing,
    )
    n_actions = getattr(env, "n_actions", env.action_space.n)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space} ({n_actions} discrete actions)")
    print(f"Action scheme: {args.action_scheme}")
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
