"""Small trainer boundary for the hourly ramp environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


CONTRACT_VERSION = "ramp-v7-semantic-action-v3"
SEMANTIC_ACTION_ID = "ramp-v6-constraint-decoded-preferences-2n-plus-1-v1"
HISTORY_HOURS = 4
REQUIRED_STEP_INFO = (
    "ramp_h1_adjusted",
    "ramp_h3_adjusted",
    "abs_adjusted_ramp_h1_fraction_s_per_hour_by_market",
    "abs_adjusted_ramp_h3_fraction_s_per_hour_by_market",
    "physical_ramp_market_order",
    "incremental_ramp_impact",
    "service_unserved",
    "batch_unfinished",
    "batch_expired",
    "certificate_violations",
    "emergency_feasibility",
    "semantic_adjustment_l2",
    "semantic_adjustment_applied",
    "semantic_adjustment_coordinate_id",
    "semantic_adjustment_units",
    "terminal_work",
)


class RampContractError(RuntimeError):
    """Raised when an environment violates the trainer boundary."""


@dataclass(frozen=True)
class EnvRequest:
    split: str
    seed: int
    rank: int = 0
    window_id: str | None = None
    training: bool = False


@runtime_checkable
class RampEnvironmentFactory(Protocol):
    def __call__(self, request: EnvRequest) -> gym.Env: ...


class RampEnvAdapter(gym.Wrapper):
    """Validate semantic actions and finite chronological episodes."""

    def __init__(self, env: gym.Env, request: EnvRequest):
        super().__init__(env)
        self.request = request
        self.episode_context: dict[str, Any] = {}
        contract_fn = getattr(env.unwrapped, "ramp_rl_contract", None)
        if not callable(contract_fn):
            raise RampContractError("environment must expose ramp_rl_contract()")
        contract = contract_fn()
        if not isinstance(contract, Mapping):
            raise RampContractError("ramp_rl_contract() must return a mapping")
        self.contract = dict(contract)
        self._validate_static_contract()

    def _validate_static_contract(self) -> None:
        if self.contract.get("version") != CONTRACT_VERSION:
            raise RampContractError("unsupported ramp environment contract version")
        if self.contract.get("semantic_feasible_action") is not True:
            raise RampContractError("trainer requires semantic feasible actions")
        if self.contract.get("semantic_action_id") != SEMANTIC_ACTION_ID:
            raise RampContractError("environment returned the wrong semantic action ID")
        if int(self.contract.get("history_hours", -1)) != HISTORY_HOURS:
            raise RampContractError(
                f"environment must supply {HISTORY_HOURS}h of warm history"
            )
        if int(self.contract.get("terminal_tail_hours", -1)) != 0:
            raise RampContractError("continuous episodes must score every slot")
        if int(self.contract.get("grid_observation_lag_hours", -1)) != 1:
            raise RampContractError(
                "the hour-t decision must observe grid rows through hour t-1 only"
            )
        if self.contract.get("actual_terminal") is not True:
            raise RampContractError("finite windows must use actual terminal states")
        if int(self.contract.get("decision_steps", 0)) <= 0:
            raise RampContractError("environment must declare decision steps")
        if not isinstance(self.action_space, spaces.Box):
            raise RampContractError("PPO integration requires a bounded Box action")
        if not np.isfinite(self.action_space.low).all() or not np.isfinite(self.action_space.high).all():
            raise RampContractError("semantic action bounds must be finite")
        if tuple(self.contract.get("action_shape", ())) != self.action_space.shape:
            raise RampContractError("declared action shape does not match the environment")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        requested = dict(options or {})
        requested["split"] = self.request.split
        if self.request.window_id is not None:
            requested["window_id"] = self.request.window_id
        observation, info = self.env.reset(seed=seed, options=requested)
        context = info.get("episode_context")
        if not isinstance(context, Mapping) or context.get("split") != self.request.split:
            raise RampContractError("reset returned the wrong episode context")
        if context.get("future_realized_features_exposed") is not False:
            raise RampContractError("environment exposes realized future information")
        if self.request.training and self.request.split != "train":
            raise RampContractError("training is restricted to the train split")
        self.episode_context = dict(context)
        return np.asarray(observation, dtype=np.float32), dict(info)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        semantic_action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(semantic_action):
            raise RampContractError("policy emitted an out-of-bounds semantic action")
        observation, reward, terminated, truncated, info = self.env.step(semantic_action)
        missing = [key for key in REQUIRED_STEP_INFO if key not in info]
        if missing:
            raise RampContractError(f"step info missing fields: {missing}")
        if truncated:
            raise RampContractError("chronological windows must terminate, not truncate")
        if terminated and info.get("actual_terminal") is not True:
            raise RampContractError("terminal transition must be the actual episode end")
        info = dict(info)
        info["ramp_episode_context"] = dict(self.episode_context)
        return np.asarray(observation, dtype=np.float32), float(reward), bool(terminated), False, info

    def evaluation_action(self, name: str) -> np.ndarray:
        if self.request.training or self.request.split == "train":
            raise RampContractError("status-quo action is evaluation-only")
        action_fn = getattr(self.env.unwrapped, "evaluation_action", None)
        if not callable(action_fn):
            raise RampContractError("environment does not expose evaluation_action()")
        action = np.asarray(action_fn(name), dtype=np.float32)
        if not self.action_space.contains(action):
            raise RampContractError("evaluation controller returned an invalid action")
        return action
