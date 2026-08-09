"""Load and validate the additive v6 pure-RL protocol."""

from __future__ import annotations

import hashlib
import json
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
    protocol_id = str(protocol["protocol"]["id"])
    is_v2 = protocol_id == "v6-ramp-pure-rl-preregistered-v2"
    _require(
        protocol_id
        in {
            "v6-ramp-pure-rl-preregistered-v1",
            "v6-ramp-pure-rl-preregistered-v2",
        },
        "unsupported ramp pure-RL campaign protocol",
    )
    environment = protocol["environment_protocol"]
    _require(
        environment["id"]
        == (
            "ramp-v6-pure-rl-frozen-v2"
            if is_v2
            else "ramp-v6-pure-rl-frozen-v1"
        ),
        "trainer must bind the frozen ramp-core protocol",
    )
    _require(
        environment["panel_schema"] == "env/protocols/v6_ramp_panel.schema.json",
        "trainer and ramp core must share one canonical panel schema",
    )
    _require(environment["cadence"] == "hourly UTC", "ramp core cadence must be hourly UTC")
    _require(
        environment["semantic_action_id"]
        == "ramp-v6-constraint-decoded-preferences-2n-plus-1-v1",
        "semantic action ID does not match the ramp-core decoder",
    )
    _require(
        environment["dimensions_for_n_sites"] == "2N+1"
        and [float(value) for value in environment["bounds"]] == [-6.0, 6.0],
        "semantic action shape/bounds do not match the ramp core",
    )

    attribution = protocol["attribution"]
    _require(attribution["random_initialization_only"] is True, "random initialization is required")
    _require(attribution["safe_random_feasible_warmup_allowed"] is True, "safe random warmup must be allowed")
    for key in FORBIDDEN_TRAINING_INPUTS:
        _require(attribution[key] is False, f"pure-RL protocol requires attribution.{key}=false")

    training = protocol["training"]
    _require(int(protocol["data"]["interval_minutes"]) == 60, "v6 ramp cadence is hourly")
    _require(training["stable_baselines3"] == EXPECTED_SB3_VERSION, "SB3 must be pinned to 2.9.0")
    _require(float(training["gamma"]) == 1.0, "finite-horizon ramp training requires gamma=1")
    _require(training["actual_terminals"] is True, "actual terminal states are required")
    _require(training["bootstrap_across_terminal"] is False, "terminal bootstrapping must be disabled")
    _require(int(training["history_hours"]) == 3, "v6 contract requires 3h real history")
    _require(int(training["terminal_tail_hours"]) == 3, "v6 contract requires a 3h terminal tail")
    _require(set(training["ramp_horizons_hours"]) == {1, 3}, "ramp horizons must be exactly 1h and 3h")
    _require(int(training["vectorized_environments"]) == 4, "integrated campaign requires four vector environments")
    _require(
        training["action_contract"] == environment["semantic_action_id"],
        "training action contract does not match the ramp environment",
    )
    _require(
        training["raw_redundant_projected_logits"] is False,
        "trainer may not reinterpret the bounded preference semantics",
    )

    algorithms = protocol["algorithms"]
    _require(float(algorithms["ppo"]["gamma"]) == 1.0, "PPO gamma must be 1")
    _require(0.0 < float(algorithms["ppo"]["gae_lambda"]) <= 1.0, "PPO GAE lambda is invalid")
    _require(float(algorithms["sac"]["gamma"]) == 1.0, "SAC gamma must be 1")
    _require(int(algorithms["sac"]["n_steps"]) >= 36, "SAC must cover the 3h delayed-credit horizon")
    if is_v2:
        _require(
            float(algorithms["ppo"]["learning_rate"]) == 1e-4
            and int(algorithms["ppo"]["n_epochs"]) == 5
            and float(algorithms["ppo"]["target_kl"]) == 0.02,
            "v2 PPO stabilization parameters are not frozen",
        )
        _require(
            training["allowed_seeds"] == [2701, 2702, 2703, 2704, 2705]
            and int(training["early_stopping_timesteps"]) == 100_000
            and int(training["effective_boundary_timesteps"]) == 110_592
            and training["normalization"]["reward"] is False,
            "v2 seed, stopping, or reward-normalization contract is not frozen",
        )

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
    confirmation_timesteps = 100_000 if is_v2 else 500_000
    _require(
        stages["confirmation"]["seeds"] == 5
        and stages["confirmation"]["timesteps"] == confirmation_timesteps,
        f"confirmation must be 5x{confirmation_timesteps}",
    )
    _require(len(stages["screen"]["seed_values"]) == 3, "screen seed list must contain three seeds")
    _require(len(stages["confirmation"]["seed_values"]) == 5, "confirmation seed list must contain five seeds")
    _require(len(stages["extension"]["seed_values"]) == 5, "extension seed list must retain five seeds")
    if is_v2:
        _require(
            stages["extension"]["enabled"] is False
            and stages["extension"]["max_timesteps"] == 100_000
            and stages["extension"]["validation_curve_materiality_required"]
            is False,
            "v2 must freeze 100k early stopping and disable extension",
        )
        _require(
            stages["screen"]["seed_values"] == [2701, 2702, 2703]
            and stages["confirmation"]["seed_values"]
            == [2701, 2702, 2703, 2704, 2705],
            "v2 seed sets are not frozen",
        )
    else:
        _require(stages["extension"]["max_timesteps"] == 2_000_000, "extension cap must be 2M")
        _require(stages["extension"]["validation_curve_materiality_required"] is True, "2M extension needs validation materiality")
    _require(
        protocol["campaign"]["final_campaign_blocked_until"]
        == ["energy-model-v3-ramp-panel"],
        "the real campaign must be blocked only on the energy-model-v3 ramp panel",
    )


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    path = path.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid protocol document: {path}")
    validate_protocol(payload)
    environment_path = ROOT / payload["environment_protocol"]["path"]
    environment_payload = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    if environment_payload["protocol"]["id"] != payload["environment_protocol"]["id"]:
        raise ValueError("ramp-core protocol ID does not match the trainer binding")
    if (
        environment_payload["protocol"]["trainer_campaign_protocol"]
        != path.relative_to(ROOT).as_posix()
    ):
        raise ValueError("ramp-core protocol does not point back to this campaign")
    if (
        environment_payload["data"]["panel_schema"]
        != payload["environment_protocol"]["panel_schema"]
    ):
        raise ValueError("ramp-core and trainer panel schema references differ")
    if (
        environment_payload["action"]["semantic_action_id"]
        != payload["environment_protocol"]["semantic_action_id"]
    ):
        raise ValueError("ramp-core and trainer semantic action IDs differ")
    panel_schema_path = ROOT / payload["environment_protocol"]["panel_schema"]
    component_hashes = {
        "campaign": normalized_sha256(path),
        "environment": normalized_sha256(environment_path),
        "panel_schema": normalized_sha256(panel_schema_path),
    }
    payload["_path"] = str(path)
    payload["_document_sha256"] = component_hashes["campaign"]
    payload["_environment_protocol_path"] = str(environment_path)
    payload["_panel_schema_path"] = str(panel_schema_path)
    payload["_component_sha256"] = component_hashes
    payload["_sha256"] = hashlib.sha256(
        json.dumps(component_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload
