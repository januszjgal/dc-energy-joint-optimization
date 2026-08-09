"""Load the frozen additive v6 protocol without touching v2-v5 loaders."""

from __future__ import annotations

from pathlib import Path

import yaml

from env.ramp_v6.models import RampProtocol


DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "protocols"
    / "v6_ramp_pure_rl.yaml"
)


def load_ramp_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> RampProtocol:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = payload["protocol"]
    objective = payload["ramp_objective"]
    robust = objective.get("robust_market_non_harm", {})
    shaping = objective.get("potential_shaping", {})
    anti_gaming = payload["anti_gaming"]
    decoder = payload["action"]["decoder"]
    protocol = RampProtocol(
        protocol_id=root["id"],
        history_hours=int(anti_gaming["warm_history_hours"]),
        terminal_tail_hours=int(anti_gaming["terminal_tail_hours"]),
        ramp_weights={
            int(key): float(value)
            for key, value in objective["weights"].items()
        },
        tail_weight=float(objective["residual_tail"]["weight"]),
        ramp_reward_scale=1.0,
        worst_market_harm_weight=float(robust.get("weight", 0.0)),
        worst_market_temperature=float(robust.get("temperature", 1e-6)),
        anticipatory_potential_scale=float(shaping.get("scale", 0.0)),
        cost_budget_fraction=float(
            objective["energy_cost"][
                "default_budget_fraction_over_status_quo"
            ]
        ),
        guaranteed_batch_capacity_fraction=float(
            decoder["guaranteed_batch_capacity_fraction"]
        ),
        service_envelope_fraction_of_fleet=float(
            decoder["service_envelope_fraction_of_fleet"]
        ),
        batch_arrival_envelope_fraction_of_fleet=float(
            decoder["batch_arrival_envelope_fraction_of_fleet"]
        ),
    )
    protocol.validate()
    if objective["scalar_reward"]["id"] not in {
        "ramp-v6-pure-rl-scalar-v1",
        "ramp-v6-pure-rl-scalar-v2",
    }:
        raise ValueError("unexpected v6 scalar reward interface")
    if robust and (
        robust.get("id") != "smooth-positive-log-mean-exp-v1"
        or float(robust.get("weight", 0.0)) <= 0.0
        or float(robust.get("temperature", 0.0)) <= 0.0
        or robust.get("post_hoc_tolerance") is not False
    ):
        raise ValueError("v2 robust market non-harm objective is invalid")
    if shaping and (
        shaping.get("id") != "causal-forecast-queue-power-potential-v1"
        or float(shaping.get("gamma", -1.0)) != 1.0
        or float(shaping.get("terminal_potential", 1.0)) != 0.0
        or float(shaping.get("scale", 0.0)) <= 0.0
        or shaping.get("realized_future_inputs") is not False
        or shaping.get("policy_invariant_finite_horizon") is not True
    ):
        raise ValueError("v2 potential shaping must be causal and policy invariant")
    if not anti_gaming["sliding_windows_no_wrap"]:
        raise ValueError("v6 requires no-wrap sliding windows")
    if anti_gaming["terminal_tail_new_arrivals"]:
        raise ValueError("v6 terminal tail must prohibit new arrivals")
    return protocol
