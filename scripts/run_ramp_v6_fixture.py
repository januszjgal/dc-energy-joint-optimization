"""Run deterministic ramp-v6 evaluation and write isolated evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.ramp_v6.environment import RampAwareEnv
from env.ramp_v6.evaluation import (
    compare_to_status_quo,
    flat_preference_policy,
    no_proxy_summary,
    run_episode,
    status_quo_policy,
    summarize,
)
from env.ramp_v6.fixture import load_fixture
from env.ramp_v6.reward import closed_window_terms


DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "ramp_v6"
DEFAULT_OUTPUT = ROOT / "output" / "ramp_v6_fixture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate additive ramp-v6 deterministic fixtures."
    )
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scale-multiplier",
        type=float,
        default=1.0,
        help=(
            "Scale site rated power, compute capacity, arrivals, and warm power "
            "together (1=100 MW/site fixture, 10=1 GW/site stress)."
        ),
    )
    return parser.parse_args()


def evaluate_fixture(
    fixture_root: Path,
    scale_multiplier: float,
) -> dict[str, Any]:
    panel, sites, workload, stats, protocol = load_fixture(
        fixture_root, scale_multiplier=scale_multiplier
    )
    candidate_env = RampAwareEnv(panel, sites, workload, stats, protocol)
    candidate_reward, candidate_history = run_episode(
        candidate_env, flat_preference_policy
    )
    candidate = summarize(candidate_history, total_reward=candidate_reward)

    status_env = RampAwareEnv(panel, sites, workload, stats, protocol)
    status_reward, status_history = run_episode(
        status_env, status_quo_policy
    )
    status = summarize(status_history, total_reward=status_reward)
    files = (
        fixture_root / "canonical_panel.csv",
        fixture_root / "fixture.json",
        ROOT / "env" / "protocols" / "v6_ramp_pure_rl.yaml",
        ROOT / "env" / "protocols" / "v6_ramp_panel.schema.json",
    )
    return {
        "evidence_id": "ramp-v6-fixture-oracle-v1",
        "long_rl_campaign_run": False,
        "scale_multiplier": scale_multiplier,
        "site_count": len(sites),
        "market_count": len(panel.markets),
        "integration_contract": {
            "panel_columns": list(panel.frame.columns),
            "action_schema": list(candidate_env.action_schema),
            "observation_schema": list(candidate_env.observation_schema),
            "stats_id": stats.stats_id,
        },
        "candidate_flat_preferences": candidate,
        "status_quo": status,
        "no_proxy": no_proxy_summary(candidate),
        "candidate_vs_status_quo": compare_to_status_quo(
            candidate, status, protocol.cost_budget_fraction
        ),
        "oracle_sanity": oracle_sanity(),
        "input_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in files
        },
    }


def oracle_sanity() -> dict[str, Any]:
    constant = closed_window_terms(110.0, 100.0, 25.0, 25.0, 200.0, 1, 0.02)
    credit = closed_window_terms(110.0, 100.0, 10.0, 15.0, 200.0, 1, 0.02)
    overshoot = closed_window_terms(110.0, 100.0, 40.0, 10.0, 200.0, 1, 0.02)
    return {
        "constant_power_incremental_impact": (
            constant.incremental_squared_impact
        ),
        "lower_power_during_up_ramp_incremental_impact": (
            credit.incremental_squared_impact
        ),
        "overshoot_incremental_impact": overshoot.incremental_squared_impact,
        "identities_pass": (
            abs(constant.incremental_squared_impact) <= 1e-12
            and credit.incremental_squared_impact < 0.0
            and overshoot.incremental_squared_impact > 0.0
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def render_report(evidence: dict[str, Any]) -> str:
    candidate = evidence["candidate_flat_preferences"]
    status = evidence["status_quo"]
    comparison = evidence["candidate_vs_status_quo"]
    safety = candidate["workload_safety"]
    return "\n".join(
        [
            "# Ramp v6 fixture/oracle evidence",
            "",
            "This is a deterministic implementation sanity check, not a long RL campaign.",
            "",
            f"- Protocol: `{candidate['protocol_id']}`",
            f"- Sites / markets: {evidence['site_count']} / {evidence['market_count']}",
            f"- Physical scale multiplier: {evidence['scale_multiplier']:.3f}",
            f"- Oracle identities pass: {evidence['oracle_sanity']['identities_pass']}",
            f"- Candidate macro incremental ramp impact: {candidate['macro_incremental_ramp_impact_mean']:.9f}",
            f"- Status-quo macro incremental ramp impact: {status['macro_incremental_ramp_impact_mean']:.9f}",
            f"- Candidate DA cost: ${candidate['da_energy_cost_usd']:.2f}",
            f"- Status-quo DA cost: ${status['da_energy_cost_usd']:.2f}",
            f"- Cost-budget compliant: {comparison['cost_budget_compliant']}",
            f"- Terminal batch queue: {safety['terminal_batch_queue']:.12f}",
            f"- Batch conservation error: {safety['batch_conservation_error']:.12f}",
            f"- Terminal-tail arrivals: {safety['terminal_tail_new_arrivals']:.12f}",
            "",
            "Per-market 1h/3h native and adjusted maximum/p95 metrics, tail burden, "
            "P/S, P/D, cost, and forecast-error strata are in `evidence.json`.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    evidence = evaluate_fixture(args.fixture_root, args.scale_multiplier)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "REPORT.md").write_text(
        render_report(evidence), encoding="utf-8"
    )
    print(args.output_root / "evidence.json")


if __name__ == "__main__":
    main()
