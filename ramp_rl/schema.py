"""Shared constants for the active pure-RL training contract."""

EXPECTED_SB3_VERSION = "2.9.0"

FORBIDDEN_TRAINING_INPUTS = (
    "teacher",
    "behavior_cloning",
    "demonstrations",
    "expert_replay",
    "warm_start",
    "analytic_economic_base",
    "mpc_actions",
    "offline_expert_labels",
    "optimizer_actions",
    "oracle_reward_shaping",
    "curriculum_trajectories",
)
