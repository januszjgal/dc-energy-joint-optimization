"""Load and validate the additive v6 pure-RL protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL_PATH = ROOT / "env" / "protocols" / "v6_pure_ramp_rl.yaml"
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


def normalized_sha256(path: Path) -> str:
    payload = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_protocol(protocol: dict[str, Any]) -> None:
    attribution = protocol["attribution"]
    _require(attribution["random_initialization_only"] is True, "random initialization is required")
    _require(attribution["safe_random_feasible_warmup_allowed"] is True, "safe random warmup must be allowed")
    for key in FORBIDDEN_TRAINING_INPUTS:
        _require(attribution[key] is False, f"pure-RL protocol requires attribution.{key}=false")

    training = protocol["training"]
    _require(training["stable_baselines3"] == EXPECTED_SB3_VERSION, "SB3 must be pinned to 2.9.0")
    _require(float(training["gamma"]) == 1.0, "finite-horizon ramp training requires gamma=1")
    _require(training["actual_terminals"] is True, "actual terminal states are required")
    _require(training["bootstrap_across_terminal"] is False, "terminal bootstrapping must be disabled")
    _require(int(training["history_hours"]) == 3, "v6 contract requires 3h real history")
    _require(int(training["terminal_tail_hours"]) == 3, "v6 contract requires a 3h terminal tail")
    _require(set(training["ramp_horizons_hours"]) == {1, 3}, "ramp horizons must be exactly 1h and 3h")
    _require(int(training["vectorized_environments"]) == 4, "integrated campaign requires four vector environments")

    algorithms = protocol["algorithms"]
    _require(float(algorithms["ppo"]["gamma"]) == 1.0, "PPO gamma must be 1")
    _require(0.0 < float(algorithms["ppo"]["gae_lambda"]) <= 1.0, "PPO GAE lambda is invalid")
    _require(float(algorithms["sac"]["gamma"]) == 1.0, "SAC gamma must be 1")
    _require(int(algorithms["sac"]["n_steps"]) >= 36, "SAC must cover the 3h delayed-credit horizon")

    split = protocol["data"]["split"]
    month_sets = [set(split[name]["months"]) for name in ("train", "validation", "test")]
    _require(not (month_sets[0] & month_sets[1] or month_sets[0] & month_sets[2] or month_sets[1] & month_sets[2]), "month splits overlap")
    _require(split["test"]["sealed"] is True, "test split must be sealed")
    _require(training["normalization"]["fit_split"] == "train", "normalization may only fit on train")
    _require(training["normalization"]["freeze_for"] == ["validation", "test"], "normalization freeze list is invalid")

    objective = protocol["multiobjective"]
    epsilons = {float(value) for value in objective["epsilon_sensitivity_pct"]}
    _require(epsilons == {0.0, 2.0, 5.0}, "epsilon sensitivities must be 0, 2, and 5 percent")
    _require(float(objective["primary_energy_budget_pct"]) == 2.0, "primary energy budget must be 2 percent")
    _require(objective["selection_data"] == "validation", "selection must use validation")
    _require(objective["test_used_for_selection"] is False, "test selection is prohibited")
    _require(protocol["success_gate"]["failure_is_publishable"] is True, "failure must be an allowed result")

    stages = protocol["campaign"]["stages"]
    _require(stages["screen"]["seeds"] == 3 and stages["screen"]["timesteps"] == 100_000, "screen stage must be 3x100k")
    _require(stages["confirmation"]["seeds"] == 5 and stages["confirmation"]["timesteps"] == 500_000, "confirmation must be 5x500k")
    _require(len(stages["screen"]["seed_values"]) == 3, "screen seed list must contain three seeds")
    _require(len(stages["confirmation"]["seed_values"]) == 5, "confirmation seed list must contain five seeds")
    _require(len(stages["extension"]["seed_values"]) == 5, "extension seed list must retain five seeds")
    _require(stages["extension"]["max_timesteps"] == 2_000_000, "extension cap must be 2M")
    _require(stages["extension"]["validation_curve_materiality_required"] is True, "2M extension needs validation materiality")


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid protocol document: {path}")
    validate_protocol(payload)
    payload["_path"] = str(path)
    payload["_sha256"] = normalized_sha256(path)
    return payload
