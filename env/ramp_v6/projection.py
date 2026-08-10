"""Constraint-only v6 action decoder: no economic or forecast objective."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from env.ramp_v6.models import EDFQueue


SEMANTIC_ADJUSTMENT_COORDINATE_ID = (
    "decoded-work-allocation-service-batch-total-destination-2n-plus-1-v1"
)
SEMANTIC_ADJUSTMENT_UNITS = "compute_work_units_per_hourly_decision"


class InfeasibleActionError(RuntimeError):
    """Raised with a hard, inspectable feasibility certificate."""

    def __init__(self, reason: str, **details: float):
        self.certificate = {"reason": reason, **details}
        super().__init__(f"v6 action infeasible: {reason}")


@dataclass(frozen=True)
class ProjectedAction:
    service: np.ndarray
    batch_by_origin: np.ndarray
    batch_by_destination: np.ndarray
    transport: np.ndarray
    mandatory_batch: float
    requested_batch: float
    requested_service: np.ndarray
    requested_batch_by_destination: np.ndarray

    def requested_semantic_allocation(self) -> np.ndarray:
        """Return the unconstrained request in decoded 2N+1 work coordinates."""
        return np.concatenate(
            (
                np.asarray(self.requested_service, dtype=np.float64),
                np.asarray([self.requested_batch], dtype=np.float64),
                np.asarray(
                    self.requested_batch_by_destination,
                    dtype=np.float64,
                ),
            )
        )

    def projected_semantic_allocation(self) -> np.ndarray:
        """Return executed work in the same decoded 2N+1 coordinates."""
        return np.concatenate(
            (
                np.asarray(self.service, dtype=np.float64),
                np.asarray(
                    [float(self.batch_by_destination.sum())],
                    dtype=np.float64,
                ),
                np.asarray(self.batch_by_destination, dtype=np.float64),
            )
        )

    def semantic_adjustment_l2(self, tolerance: float = 1e-12) -> float:
        """Euclidean decoder adjustment in compute-work units, not logit units."""
        value = float(
            np.linalg.norm(
                self.projected_semantic_allocation()
                - self.requested_semantic_allocation()
            )
        )
        return 0.0 if value <= tolerance else value


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - float(np.max(values))
    exp_values = np.exp(values)
    return exp_values / float(exp_values.sum())


def bounded_fraction(value: float, bound: float = 6.0) -> float:
    """Map a bounded scalar to [0, 1] with exact endpoints."""
    return (float(np.clip(value, -bound, bound)) + bound) / (2.0 * bound)


def project_capped_simplex(
    desired: np.ndarray,
    total: float,
    upper: np.ndarray,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Euclidean projection onto sum(x)=total with 0<=x<=upper."""
    desired = np.asarray(desired, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if desired.shape != upper.shape:
        raise ValueError("desired and upper shapes differ")
    capacity = float(upper.sum())
    if total < -tolerance or total > capacity + tolerance:
        raise InfeasibleActionError(
            "capped_simplex_total", requested=float(total), capacity=capacity
        )
    if total <= tolerance:
        return np.zeros_like(upper)
    if capacity - total <= tolerance:
        return upper.copy()
    low = float(np.min(desired - upper))
    high = float(np.max(desired))
    for _ in range(100):
        midpoint = (low + high) / 2.0
        candidate = np.clip(desired - midpoint, 0.0, upper)
        if float(candidate.sum()) > total:
            low = midpoint
        else:
            high = midpoint
    result = np.clip(desired - high, 0.0, upper)
    residual = float(total - result.sum())
    for index in np.argsort(-(upper - result) if residual > 0 else -result):
        if abs(residual) <= tolerance:
            break
        if residual > 0.0:
            delta = min(residual, float(upper[index] - result[index]))
        else:
            delta = -min(-residual, float(result[index]))
        result[index] += delta
        residual -= delta
    if not math.isclose(float(result.sum()), total, abs_tol=1e-8):
        raise RuntimeError("capped simplex failed exact conservation")
    return result


def exact_transport(origin: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Deterministic northwest-corner transport with exact endpoints."""
    supply = np.asarray(origin, dtype=np.float64).copy()
    demand = np.asarray(destination, dtype=np.float64).copy()
    if not math.isclose(float(supply.sum()), float(demand.sum()), abs_tol=1e-8):
        raise ValueError("transport endpoints must have equal totals")
    flow = np.zeros((len(supply), len(demand)), dtype=np.float64)
    i = j = 0
    while i < len(supply) and j < len(demand):
        value = min(float(supply[i]), float(demand[j]))
        flow[i, j] += value
        supply[i] -= value
        demand[j] -= value
        if supply[i] <= 1e-12:
            i += 1
        if demand[j] <= 1e-12:
            j += 1
    if not np.allclose(flow.sum(axis=1), origin, atol=1e-8):
        raise RuntimeError("transport failed origin conservation")
    if not np.allclose(flow.sum(axis=0), destination, atol=1e-8):
        raise RuntimeError("transport failed destination conservation")
    return flow


def project_action(
    raw_action: np.ndarray,
    service_total: float,
    capacity: np.ndarray,
    queue: EDFQueue,
    current_step: int,
    final_step: int,
    n_origins: int,
    guaranteed_future_batch_capacity_by_deadline: dict[int, float] | None = None,
) -> ProjectedAction:
    """Decode preferences into the unique hard-feasible work movement."""
    n_sites = len(capacity)
    action = np.asarray(raw_action, dtype=np.float64)
    if action.shape != (2 * n_sites + 1,):
        raise ValueError(f"action must have shape {(2 * n_sites + 1,)}")
    if service_total > float(capacity.sum()) + 1e-9:
        raise InfeasibleActionError(
            "service_capacity_deficit",
            service=float(service_total),
            capacity=float(capacity.sum()),
        )
    requested_service = softmax(action[:n_sites]) * service_total
    service = project_capped_simplex(
        requested_service,
        service_total,
        capacity,
    )
    residual = capacity - service

    queue_total = queue.total
    mandatory = queue.due_by(current_step + 1)
    future_capacity = guaranteed_future_batch_capacity_by_deadline or {}
    for deadline in queue.deadlines:
        due = queue.due_by(deadline)
        available = max(float(future_capacity.get(deadline, 0.0)), 0.0)
        mandatory = max(mandatory, due - available)
    if current_step == final_step:
        mandatory = queue_total
    if mandatory > float(residual.sum()) + 1e-9:
        raise InfeasibleActionError(
            "deadline_capacity_deficit",
            mandatory=float(mandatory),
            residual_capacity=float(residual.sum()),
        )
    maximum = min(queue_total, float(residual.sum()))
    requested = bounded_fraction(float(action[n_sites])) * queue_total
    batch_total = float(np.clip(requested, mandatory, maximum))
    destination_weights = softmax(action[n_sites + 1 :])
    requested_batch_destination = destination_weights * requested
    destination_pref = destination_weights * batch_total
    batch_destination = project_capped_simplex(
        destination_pref, batch_total, residual
    )
    batch_origin = queue.drain(batch_total, n_origins)
    transport = exact_transport(batch_origin, batch_destination)
    return ProjectedAction(
        service=service,
        batch_by_origin=batch_origin,
        batch_by_destination=batch_destination,
        transport=transport,
        mandatory_batch=float(mandatory),
        requested_batch=float(requested),
        requested_service=requested_service,
        requested_batch_by_destination=requested_batch_destination,
    )
