"""Causal one-step feasibility projection for joint service/batch control."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from env.workload_generator import BatchPool


@dataclass(frozen=True)
class SafetyConfig:
    """Frozen assumptions and optional constraints for the v4 projector."""

    service_envelope_total: float = 2.25
    batch_arrival_envelope_total: float = 1.0
    future_fleet_capacity_total: float = 4.0
    envelope_id: str = "ad-rounded-envelope-v1"
    envelope_scope: str = "a-d-development-only"
    negative_demand_flush: bool = False
    negative_demand_flush_price_ceiling: float | None = None
    max_grid_mw: tuple[float, ...] | None = None
    max_upward_ramp_mw: tuple[float, ...] | None = None
    tolerance: float = 1e-9

    @property
    def guaranteed_carried_batch_capacity(self) -> float:
        return (
            self.future_fleet_capacity_total
            - self.service_envelope_total
            - self.batch_arrival_envelope_total
        )

    def validate(self, n_dc: int) -> None:
        if not self.envelope_id.strip() or not self.envelope_scope.strip():
            raise ValueError(
                "envelope_id and envelope_scope must be non-empty"
            )
        numeric = {
            "service_envelope_total": self.service_envelope_total,
            "batch_arrival_envelope_total": (
                self.batch_arrival_envelope_total
            ),
            "future_fleet_capacity_total": (
                self.future_fleet_capacity_total
            ),
            "tolerance": self.tolerance,
        }
        for name, value in numeric.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative, got {value}"
                )
        if self.guaranteed_carried_batch_capacity < -self.tolerance:
            raise ValueError(
                "service and batch-arrival envelopes exceed future fleet "
                "capacity"
            )
        for name, values in (
            ("max_grid_mw", self.max_grid_mw),
            ("max_upward_ramp_mw", self.max_upward_ramp_mw),
        ):
            if values is None:
                continue
            if len(values) != n_dc:
                raise ValueError(
                    f"{name} must have {n_dc} values, got {len(values)}"
                )
            if any(
                not math.isfinite(value) or value < 0.0
                for value in values
            ):
                raise ValueError(
                    f"{name} values must be finite and non-negative"
                )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyProjectionResult:
    service: np.ndarray
    origin_batch: np.ndarray
    destination_batch: np.ndarray
    transport: np.ndarray
    service_fractions: np.ndarray
    drain_rates: np.ndarray
    batch_fractions: np.ndarray
    mandatory_by_origin: np.ndarray
    mandatory_total: float
    desired_service: np.ndarray
    desired_origin_batch: np.ndarray
    desired_destination_batch: np.ndarray
    effective_capacity: np.ndarray
    residual_capacity: np.ndarray
    projection_l2: float
    intervened: bool
    binding_deadline_step: int | None
    minimum_deadline_slack: float
    service_capacity_slack: float
    batch_capacity_slack: float
    negative_flush_active: bool
    exact_zero_drain_count: int
    exact_full_drain_count: int


class SafetyInfeasibleError(RuntimeError):
    """Raised when a hard safety guarantee cannot be satisfied."""

    def __init__(self, certificate: dict[str, Any]):
        self.certificate = certificate
        super().__init__(
            "Safety projection infeasible: "
            f"{certificate.get('reason', 'unknown')}"
        )


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - float(np.max(values))
    exp_values = np.exp(shifted)
    return exp_values / float(exp_values.sum())


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def project_capped_simplex(
    desired: np.ndarray,
    total: float,
    upper: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Euclidean projection onto ``sum(x)=total, 0<=x<=upper``."""
    desired = np.asarray(desired, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if desired.shape != upper.shape:
        raise ValueError("desired and upper must have the same shape")
    if np.any(upper < -tolerance):
        raise ValueError("upper bounds must be non-negative")
    total = float(total)
    capacity = float(upper.sum())
    if total < -tolerance or total > capacity + tolerance:
        raise ValueError(
            f"simplex total {total} outside [0, {capacity}]"
        )
    if total <= tolerance:
        return np.zeros_like(desired)
    if capacity - total <= tolerance:
        return upper.copy()

    low = float(np.min(desired - upper))
    high = float(np.max(desired))
    for _ in range(100):
        midpoint = 0.5 * (low + high)
        projected = np.clip(desired - midpoint, 0.0, upper)
        if float(projected.sum()) > total:
            low = midpoint
        else:
            high = midpoint
    result = np.clip(desired - high, 0.0, upper)
    residual = total - float(result.sum())
    if abs(residual) > tolerance:
        if residual > 0.0:
            room = upper - result
            for index in np.argsort(-room):
                delta = min(residual, float(room[index]))
                result[index] += delta
                residual -= delta
                if residual <= tolerance:
                    break
        else:
            for index in np.argsort(-result):
                delta = min(-residual, float(result[index]))
                result[index] -= delta
                residual += delta
                if residual >= -tolerance:
                    break
    if not math.isclose(
        float(result.sum()), total, rel_tol=0.0, abs_tol=1e-8
    ):
        raise RuntimeError("capped-simplex projection did not conserve total")
    return result


def project_box_sum_range(
    desired: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    minimum_total: float,
    maximum_total: float,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Project onto box bounds with a permitted interval for the total."""
    desired = np.asarray(desired, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if not (desired.shape == lower.shape == upper.shape):
        raise ValueError("desired/lower/upper shapes must match")
    if np.any(lower < -tolerance) or np.any(upper < lower - tolerance):
        raise ValueError("invalid box bounds")
    lower_total = float(lower.sum())
    upper_total = float(upper.sum())
    minimum_total = max(float(minimum_total), lower_total)
    maximum_total = min(float(maximum_total), upper_total)
    if minimum_total > maximum_total + tolerance:
        raise ValueError("empty box/sum feasible set")
    clipped = np.clip(desired, lower, upper)
    current_total = float(clipped.sum())
    if current_total < minimum_total - tolerance:
        return lower + project_capped_simplex(
            desired - lower,
            minimum_total - lower_total,
            upper - lower,
            tolerance=tolerance,
        )
    if current_total > maximum_total + tolerance:
        return lower + project_capped_simplex(
            desired - lower,
            maximum_total - lower_total,
            upper - lower,
            tolerance=tolerance,
        )
    return clipped


def exact_transport(
    origin: np.ndarray,
    destination: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Construct a deterministic non-negative flow with exact row/column sums."""
    rows = np.asarray(origin, dtype=np.float64).copy()
    columns = np.asarray(destination, dtype=np.float64).copy()
    if not math.isclose(
        float(rows.sum()),
        float(columns.sum()),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("origin and destination totals differ")
    flow = np.zeros((len(rows), len(columns)), dtype=np.float64)
    i = 0
    j = 0
    while i < len(rows) and j < len(columns):
        amount = min(float(rows[i]), float(columns[j]))
        if amount > tolerance:
            flow[i, j] = amount
            rows[i] -= amount
            columns[j] -= amount
        if rows[i] <= tolerance:
            i += 1
        if j < len(columns) and columns[j] <= tolerance:
            j += 1
    if (
        np.max(np.abs(flow.sum(axis=1) - origin), initial=0.0)
        > 1e-8
        or np.max(
            np.abs(flow.sum(axis=0) - destination), initial=0.0
        )
        > 1e-8
    ):
        raise RuntimeError("transport construction lost work")
    return flow


def _mandatory_edf_by_origin(
    pools: Sequence[BatchPool],
    desired_origin: np.ndarray,
    current_step: int,
    max_steps: int,
    guaranteed_future_capacity: float,
    *,
    tolerance: float,
) -> tuple[np.ndarray, float, int | None]:
    entries: list[tuple[int, int, float]] = []
    for origin, pool in enumerate(pools):
        for entry in pool.entries:
            if entry.cpu_demand <= tolerance:
                continue
            if entry.deadline_step <= current_step:
                raise ValueError(
                    "unsalvageable entry reached mandatory EDF projection"
                )
            entries.append(
                (
                    min(int(entry.deadline_step), int(max_steps)),
                    origin,
                    float(entry.cpu_demand),
                )
            )
    if not entries:
        return np.zeros(len(pools), dtype=np.float64), 0.0, None

    desired = np.asarray(desired_origin, dtype=np.float64)
    mandatory = np.zeros(len(pools), dtype=np.float64)
    binding: int | None = None
    for deadline in sorted({entry[0] for entry in entries}):
        due_by_origin = np.zeros(len(pools), dtype=np.float64)
        for entry_deadline, origin, amount in entries:
            if entry_deadline <= deadline:
                due_by_origin[origin] += amount
        due = float(due_by_origin.sum())
        future_slots = max(deadline - current_step - 1, 0)
        required = max(
            0.0,
            due - guaranteed_future_capacity * future_slots,
        )
        protected = float(
            np.minimum(mandatory, due_by_origin).sum()
        )
        deficit = max(0.0, required - protected)
        if deficit <= tolerance:
            continue
        room = np.maximum(due_by_origin - mandatory, 0.0)
        if float(room.sum()) < deficit - tolerance:
            raise RuntimeError(
                "deadline-prefix projection has insufficient urgent work"
            )
        # Minimal squared deviation from the PPO origin preference for the
        # additional amount needed by this nested deadline prefix.
        mandatory += project_capped_simplex(
            desired - mandatory,
            deficit,
            room,
            tolerance=tolerance,
        )
        if deficit > tolerance:
            binding = deadline
    return mandatory, float(mandatory.sum()), binding


def _minimum_deadline_slack(
    pools: Sequence[BatchPool],
    origin_drain: np.ndarray,
    current_step: int,
    max_steps: int,
    guaranteed_future_capacity: float,
    *,
    tolerance: float,
) -> float:
    remaining_entries: list[tuple[int, float]] = []
    for origin, pool in enumerate(pools):
        drain_left = float(origin_drain[origin])
        for entry in sorted(
            pool.entries, key=lambda item: item.deadline_step
        ):
            amount = float(entry.cpu_demand)
            take = min(amount, drain_left)
            amount -= take
            drain_left -= take
            if amount > tolerance:
                remaining_entries.append(
                    (min(entry.deadline_step, max_steps), amount)
                )
        if drain_left > 1e-8:
            raise RuntimeError("projected drain exceeds origin pool")
    if not remaining_entries:
        return math.inf
    slack = math.inf
    for deadline in sorted({entry[0] for entry in remaining_entries}):
        due = sum(
            amount
            for entry_deadline, amount in remaining_entries
            if entry_deadline <= deadline
        )
        future_slots = max(deadline - current_step - 1, 0)
        slack = min(
            slack,
            guaranteed_future_capacity * future_slots - due,
        )
    return float(slack)


def project_joint_action(
    raw_action: np.ndarray,
    pools: Sequence[BatchPool],
    *,
    current_step: int,
    max_steps: int,
    total_service: float,
    current_batch_arrival: float,
    effective_capacity: np.ndarray,
    net_demand: np.ndarray,
    price: np.ndarray,
    config: SafetyConfig,
) -> SafetyProjectionResult:
    """Project one decoded joint action into the hard feasible set."""
    n_dc = len(pools)
    config.validate(n_dc)
    action = np.asarray(raw_action, dtype=np.float64)
    if action.shape != (3 * n_dc,):
        raise ValueError(
            f"expected action shape {(3 * n_dc,)}, got {action.shape}"
        )
    capacity = np.asarray(effective_capacity, dtype=np.float64)
    if capacity.shape != (n_dc,) or np.any(capacity < 0.0):
        raise ValueError("invalid effective capacities")
    tolerance = config.tolerance
    if total_service > config.service_envelope_total + tolerance:
        raise SafetyInfeasibleError(
            {
                "reason": "service_envelope_exceeded",
                "step": current_step,
                "observed_service": float(total_service),
                "service_envelope": config.service_envelope_total,
            }
        )
    if (
        current_batch_arrival
        > config.batch_arrival_envelope_total + tolerance
    ):
        raise SafetyInfeasibleError(
            {
                "reason": "batch_arrival_envelope_exceeded",
                "step": current_step,
                "observed_batch_arrival": float(current_batch_arrival),
                "batch_arrival_envelope": (
                    config.batch_arrival_envelope_total
                ),
            }
        )
    fleet_capacity = float(capacity.sum())
    if total_service > fleet_capacity + tolerance:
        raise SafetyInfeasibleError(
            {
                "reason": "service_capacity_deficit",
                "step": current_step,
                "required_service": float(total_service),
                "available_capacity": fleet_capacity,
                "deficit": float(total_service - fleet_capacity),
            }
        )

    service_pref = softmax(action[:n_dc])
    raw_drain_rates = sigmoid(action[n_dc : 2 * n_dc])
    batch_pref = softmax(action[2 * n_dc :])
    desired_service = service_pref * total_service
    service = project_capped_simplex(
        desired_service,
        total_service,
        capacity,
        tolerance=tolerance,
    )
    residual = capacity - service

    pool_totals = np.array(
        [pool.total_demand for pool in pools],
        dtype=np.float64,
    )
    desired_origin = raw_drain_rates * pool_totals
    guaranteed_capacity = max(
        0.0, config.guaranteed_carried_batch_capacity
    )
    try:
        mandatory, mandatory_total, binding = (
            _mandatory_edf_by_origin(
                pools,
                desired_origin,
                current_step,
                max_steps,
                guaranteed_capacity,
                tolerance=tolerance,
            )
        )
    except ValueError as exc:
        raise SafetyInfeasibleError(
            {
                "reason": "pre_action_deadline_miss",
                "step": current_step,
                "detail": str(exc),
            }
        ) from exc

    batch_capacity = float(residual.sum())
    if mandatory_total > batch_capacity + tolerance:
        raise SafetyInfeasibleError(
            {
                "reason": "mandatory_batch_capacity_deficit",
                "step": current_step,
                "mandatory_batch": mandatory_total,
                "available_batch_capacity": batch_capacity,
                "deficit": mandatory_total - batch_capacity,
                "binding_deadline_step": binding,
            }
        )

    flush_mask = np.asarray(net_demand, dtype=np.float64) < 0.0
    if config.negative_demand_flush_price_ceiling is not None:
        flush_mask &= (
            np.asarray(price, dtype=np.float64)
            <= config.negative_demand_flush_price_ceiling
        )
    flush_active = bool(
        config.negative_demand_flush and np.any(flush_mask)
    )
    minimum_total = mandatory_total
    if flush_active:
        minimum_total = max(
            minimum_total,
            min(
                float(pool_totals.sum()),
                float(residual[flush_mask].sum()),
            ),
        )
    maximum_total = min(float(pool_totals.sum()), batch_capacity)
    try:
        origin_batch = project_box_sum_range(
            desired_origin,
            mandatory,
            pool_totals,
            minimum_total,
            maximum_total,
            tolerance=tolerance,
        )
    except ValueError as exc:
        raise SafetyInfeasibleError(
            {
                "reason": "origin_drain_projection_empty",
                "step": current_step,
                "mandatory_batch": mandatory_total,
                "available_batch_capacity": batch_capacity,
                "detail": str(exc),
            }
        ) from exc

    total_batch = float(origin_batch.sum())
    desired_destination = batch_pref * total_batch
    if flush_active and total_batch > tolerance:
        negative_target = min(
            total_batch, float(residual[flush_mask].sum())
        )
        negative_upper = residual * flush_mask.astype(np.float64)
        negative_desired = desired_destination * flush_mask
        negative_placement = project_capped_simplex(
            negative_desired,
            negative_target,
            negative_upper,
            tolerance=tolerance,
        )
        remaining_target = total_batch - negative_target
        destination_batch = negative_placement
        if remaining_target > tolerance:
            remaining_upper = residual - negative_placement
            destination_batch = (
                destination_batch
                + project_capped_simplex(
                    desired_destination - negative_placement,
                    remaining_target,
                    remaining_upper,
                    tolerance=tolerance,
                )
            )
    else:
        destination_batch = project_capped_simplex(
            desired_destination,
            total_batch,
            residual,
            tolerance=tolerance,
        )

    transport = exact_transport(
        origin_batch,
        destination_batch,
        tolerance=tolerance,
    )
    service_fractions = np.divide(
        service,
        total_service,
        out=np.zeros_like(service),
        where=total_service > tolerance,
    )
    drain_rates = np.divide(
        origin_batch,
        pool_totals,
        out=np.zeros_like(origin_batch),
        where=pool_totals > tolerance,
    )
    batch_fractions = np.divide(
        destination_batch,
        total_batch,
        out=np.zeros_like(destination_batch),
        where=total_batch > tolerance,
    )
    desired_destination_for_distance = batch_pref * total_batch
    projection_l2 = float(
        math.sqrt(
            float(np.square(service - desired_service).sum())
            + float(np.square(origin_batch - desired_origin).sum())
            + float(
                np.square(
                    destination_batch
                    - desired_destination_for_distance
                ).sum()
            )
        )
    )
    minimum_slack = _minimum_deadline_slack(
        pools,
        origin_batch,
        current_step,
        max_steps,
        guaranteed_capacity,
        tolerance=tolerance,
    )
    if minimum_slack < -1e-7:
        raise RuntimeError(
            "projected drain violates cumulative deadline feasibility"
        )
    return SafetyProjectionResult(
        service=service,
        origin_batch=origin_batch,
        destination_batch=destination_batch,
        transport=transport,
        service_fractions=service_fractions,
        drain_rates=drain_rates,
        batch_fractions=batch_fractions,
        mandatory_by_origin=mandatory,
        mandatory_total=mandatory_total,
        desired_service=desired_service,
        desired_origin_batch=desired_origin,
        desired_destination_batch=desired_destination_for_distance,
        effective_capacity=capacity,
        residual_capacity=residual,
        projection_l2=projection_l2,
        intervened=projection_l2 > 1e-8,
        binding_deadline_step=binding,
        minimum_deadline_slack=minimum_slack,
        service_capacity_slack=fleet_capacity - total_service,
        batch_capacity_slack=batch_capacity - total_batch,
        negative_flush_active=flush_active,
        exact_zero_drain_count=int(np.count_nonzero(drain_rates == 0.0)),
        exact_full_drain_count=int(np.count_nonzero(drain_rates == 1.0)),
    )
