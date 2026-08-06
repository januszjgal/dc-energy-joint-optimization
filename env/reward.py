"""Configuration for post-v2 PPO reward experiments.

The frozen v2 environment keeps its legacy constructor behavior when no
``RewardConfig`` is supplied.  V3 development runs use this immutable config so
reward semantics can be hashed and reproduced independently of CLI defaults.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    """Training-only reward controls for completion-guarded batch PPO."""

    service_backlog_weight: float
    batch_completion_weight: float
    evaluation_service_backlog_weight: float = 1000.0
    evaluation_batch_completion_weight: float = 1000.0
    reward_scale: float = 1e-4
    subtract_idle_cost: bool = False
    urgency_potential_weight: float = 0.0

    def validate(self, economic_floor: float) -> None:
        values = {
            "service_backlog_weight": self.service_backlog_weight,
            "batch_completion_weight": self.batch_completion_weight,
            "evaluation_service_backlog_weight": (
                self.evaluation_service_backlog_weight
            ),
            "evaluation_batch_completion_weight": (
                self.evaluation_batch_completion_weight
            ),
            "reward_scale": self.reward_scale,
            "urgency_potential_weight": self.urgency_potential_weight,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.service_backlog_weight < economic_floor:
            raise ValueError(
                "service_backlog_weight must be at least the modeled economic "
                f"floor ${economic_floor:,.2f}; got {self.service_backlog_weight}"
            )
        if self.batch_completion_weight < economic_floor:
            raise ValueError(
                "batch_completion_weight must be at least the modeled economic "
                f"floor ${economic_floor:,.2f}; got {self.batch_completion_weight}"
            )
        if self.evaluation_service_backlog_weight < economic_floor:
            raise ValueError(
                "evaluation_service_backlog_weight must be at least the modeled "
                f"economic floor ${economic_floor:,.2f}; got "
                f"{self.evaluation_service_backlog_weight}"
            )
        if self.evaluation_batch_completion_weight < economic_floor:
            raise ValueError(
                "evaluation_batch_completion_weight must be at least the modeled "
                f"economic floor ${economic_floor:,.2f}; got "
                f"{self.evaluation_batch_completion_weight}"
            )
        if self.reward_scale <= 0.0:
            raise ValueError(
                f"reward_scale must be positive, got {self.reward_scale}"
            )
        if self.urgency_potential_weight < 0.0:
            raise ValueError(
                "urgency_potential_weight must be non-negative, got "
                f"{self.urgency_potential_weight}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-serializable representation."""
        return asdict(self)
