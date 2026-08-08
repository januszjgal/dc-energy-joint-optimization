"""Narrow integration contract for the separately developed v6 environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

CONTRACT_VERSION = "ramp-v6-semantic-action-v1"
REQUIRED_STEP_INFO = (
    "ramp_h1_adjusted",
    "ramp_h3_adjusted",
    "incremental_ramp_impact",
    "energy_cost",
    "service_unserved",
    "batch_unfinished",
    "batch_expired",
    "certificate_violations",
    "emergency_feasibility",
    "semantic_adjustment_l2",
    "action_provenance",
)


class RampContractError(RuntimeError):
    """Raised when an environment violates the v6 trainer boundary."""


@dataclass(frozen=True)
class EnvRequest:
    split: str
    seed: int
    rank: int = 0
    window_id: str | None = None
    training: bool = False
    epsilon_pct: float = 2.0
    lagrangian_multiplier: float = 0.0


@runtime_checkable
class RampEnvironmentFactory(Protocol):
    def __call__(self, request: EnvRequest) -> gym.Env: ...


class RampEnvAdapter(gym.Wrapper):
    """Validate semantic actions, chronological context, and true terminals."""

    def __init__(self, env: gym.Env, request: EnvRequest):
        super().__init__(env)
        self.request = request
        self.episode_context: dict[str, Any] = {}
        contract_fn = getattr(env.unwrapped, "ramp_rl_contract", None)
        if not callable(contract_fn):
            raise RampContractError("v6 environment must expose ramp_rl_contract()")
        contract = contract_fn()
        if not isinstance(contract, Mapping):
            raise RampContractError("ramp_rl_contract() must return a mapping")
        self.contract = dict(contract)
        self._validate_static_contract()

    def _validate_static_contract(self) -> None:
        if self.contract.get("version") != CONTRACT_VERSION:
            raise RampContractError("unsupported ramp environment contract version")
        if self.contract.get("semantic_feasible_action") is not True:
            raise RampContractError("trainer requires the environment's semantic feasible action")
        if self.contract.get("raw_redundant_projected_logits") is not False:
            raise RampContractError("raw redundant projected logits are prohibited")
        if int(self.contract.get("history_hours", -1)) != 3:
            raise RampContractError("environment must supply 3h of real warm history")
        if int(self.contract.get("terminal_tail_hours", -1)) != 3:
            raise RampContractError("environment must settle a 3h terminal tail")
        if self.contract.get("actual_terminal") is not True:
            raise RampContractError("finite windows must use actual terminal states")
        if int(self.contract.get("decision_steps", 0)) <= 0:
            raise RampContractError("environment must declare a fixed positive decision_steps")
        if not isinstance(self.action_space, spaces.Box):
            raise RampContractError("SB3 PPO/SAC integration requires a bounded Box semantic action")
        if not np.isfinite(self.action_space.low).all() or not np.isfinite(self.action_space.high).all():
            raise RampContractError("semantic action bounds must be finite")
        if self.request.training and not callable(
            getattr(self.env.unwrapped, "set_lagrangian_multiplier", None)
        ):
            raise RampContractError("training environment must support frozen Lagrangian updates")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        requested = dict(options or {})
        requested["split"] = self.request.split
        if self.request.window_id is not None:
            requested["window_id"] = self.request.window_id
        observation, info = self.env.reset(seed=seed, options=requested)
        context = info.get("episode_context")
        if not isinstance(context, Mapping):
            raise RampContractError("reset info must contain episode_context")
        if context.get("split") != self.request.split:
            raise RampContractError("environment returned the wrong data split")
        if int(context.get("history_hours", -1)) != 3 or int(context.get("terminal_tail_hours", -1)) != 3:
            raise RampContractError("episode context is missing the required history or tail")
        if context.get("forecast_model") in (None, "") or context.get("forecast_vintage") in (None, ""):
            raise RampContractError("forecast model and vintage identity are required")
        if not isinstance(context.get("source_hashes"), Mapping) or not context["source_hashes"]:
            raise RampContractError("source/data hashes are required")
        if context.get("future_realized_features_exposed") is not False:
            raise RampContractError("environment exposes realized future information")
        if self.request.training and self.request.split != "train":
            raise RampContractError("training is restricted to the train split")
        self.episode_context = dict(context)
        return np.asarray(observation, dtype=np.float32), dict(info)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        semantic_action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(semantic_action):
            raise RampContractError("policy emitted an out-of-bounds semantic action")
        observation, reward, terminated, truncated, info = self.env.step(semantic_action)
        missing = [key for key in REQUIRED_STEP_INFO if key not in info]
        if missing:
            raise RampContractError(f"step info missing required evidence fields: {missing}")
        provenance = str(info["action_provenance"])
        if self.request.training and provenance != "agent_semantic":
            raise RampContractError("environment reported a non-agent training action")
        if not self.request.training and provenance not in {
            "agent_semantic",
            "evaluation_status_quo",
            "evaluation_oracle_bound",
        }:
            raise RampContractError("environment reported an unknown evaluation action source")
        if truncated:
            raise RampContractError("chronological windows must terminate, not truncate")
        if terminated:
            if info.get("tail_complete") is not True or info.get("actual_terminal") is not True:
                raise RampContractError("terminal transition must include the completed 3h tail")
            expected_tail_steps = int(
                60 * int(self.contract["terminal_tail_hours"])
                / int(self.contract.get("interval_minutes", 5))
            )
            for key in (
                "terminal_tail_ramp_h1_adjusted",
                "terminal_tail_ramp_h3_adjusted",
                "terminal_tail_incremental_ramp_impact",
            ):
                values = info.get(key)
                if not isinstance(values, (list, tuple)) or len(values) != expected_tail_steps:
                    raise RampContractError(f"terminal transition must expose {expected_tail_steps} values for {key}")
        info = dict(info)
        info["ramp_episode_context"] = dict(self.episode_context)
        return np.asarray(observation, dtype=np.float32), float(reward), bool(terminated), False, dict(info)

    def set_lagrangian_multiplier(self, value: float) -> None:
        setter = getattr(self.env.unwrapped, "set_lagrangian_multiplier", None)
        if not callable(setter):
            raise RampContractError("environment does not support Lagrangian updates")
        setter(float(value))

    def evaluation_action(self, name: str) -> np.ndarray:
        if self.request.training or self.request.split == "train":
            raise RampContractError("status-quo/oracle actions are evaluation-only")
        action_fn = getattr(self.env.unwrapped, "evaluation_action", None)
        if not callable(action_fn):
            raise RampContractError("environment does not expose evaluation_action()")
        action = np.asarray(action_fn(name), dtype=np.float32)
        if not self.action_space.contains(action):
            raise RampContractError("evaluation controller returned an invalid semantic action")
        return action
