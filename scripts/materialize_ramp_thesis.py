"""Materialize the final ramp thesis from hash-verified V4R evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ramp_rl.v4r_thesis import (  # noqa: E402
    DEFAULT_CANONICAL,
    EXPECTED_CANONICAL_SHA256,
    EXPECTED_PROTOCOL_ID,
    EXPECTED_PROTOCOL_SHA256,
    EXPECTED_SOURCE_COMMIT,
    build_claim_ledger,
    load_verified_evidence,
)
from scripts.validate_ramp_thesis import validate_text  # noqa: E402


DEFAULT_SOURCE = ROOT / "thesis_ramp_v6.md"
DEFAULT_OUTPUT = ROOT / "thesis_paper.md"
EXPECTED_TEMPLATE_SHA256 = (
    "ba215bbf0cd2645591d8a1a9e02e32aef64407662b1de2c13a73506ee88abd44"
)
MARKET_LABELS = {
    "CAISO_NP15": "CAISO NP15",
    "ERCOT_LZ_NORTH": "ERCOT North",
    "ISONE_NEMA": "ISO-NE NEMA",
    "MISO_MINN_HUB": "MISO Minnesota Hub",
    "NYISO_NYC_J": "NYISO Zone J",
    "SPP_NORTH_HUB": "SPP North Hub",
}


def fmt(value: Any, digits: int = 10) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"expected numeric evidence, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"non-finite evidence value: {value!r}")
    return f"{value:.{digits}g}"


def pct_change_from_ratio(value: float) -> str:
    return f"{100 * (value - 1):+.3f}%"


def render_split_table(evidence: dict[str, Any]) -> str:
    rows = [
        "| Split | Episodes | Mean incremental ramp impact | DA cost ratio | Cost change | Every market negative | Strict gates |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for label, result in (
        ("February validation", evidence["validation"]["result"]),
        ("March-April sealed test", evidence["test"]["result"]),
    ):
        rows.append(
            "| {label} | {episodes} | {impact} | {ratio} | {change} | true | true |".format(
                label=label,
                episodes=result["episode_count"],
                impact=fmt(result["mean_incremental_ramp_impact"], 12),
                ratio=fmt(result["energy_cost_ratio"], 12),
                change=pct_change_from_ratio(result["energy_cost_ratio"]),
            )
        )
    return "\n".join(rows)


def render_market_table(evidence: dict[str, Any]) -> str:
    validation = evidence["validation"]["result"]["per_market_macro"]
    test = evidence["test"]["result"]["per_market_macro"]
    rows = [
        "| Evaluated market | Validation impact | Sealed-test impact | Test direction |",
        "|---|---:|---:|---|",
    ]
    for market in MARKET_LABELS:
        rows.append(
            f"| {MARKET_LABELS[market]} | {fmt(validation[market], 11)} | "
            f"{fmt(test[market], 11)} | counter-ramp |"
        )
    return "\n".join(rows)


def render_physical_table(evidence: dict[str, Any]) -> str:
    rows = [
        "| Split | 1 h p95 | 1 h max | 3 h p95 | 3 h max |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, result in (
        ("Validation", evidence["validation"]["result"]),
        ("Sealed test", evidence["test"]["result"]),
    ):
        rows.append(
            "| {label} | {h1p95} | {h1max} | {h3p95} | {h3max} |".format(
                label=label,
                h1p95=fmt(result["ramp_h1_adjusted_p95"], 8),
                h1max=fmt(result["ramp_h1_adjusted_max"], 8),
                h3p95=fmt(result["ramp_h3_adjusted_p95"], 8),
                h3max=fmt(result["ramp_h3_adjusted_max"], 8),
            )
        )
    return "\n".join(rows)


def render_behavior_table(evidence: dict[str, Any]) -> str:
    rows = [
        "| Split | Deferrable pre-service | Status-quo ramp power | Policy ramp power | Reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, result in (
        ("Validation", evidence["validation"]["result"]),
        ("Sealed test", evidence["test"]["result"]),
    ):
        audit = result["behavior_audit"]
        reduction = 100 * (1 - audit["policy_ramp_power"] / audit["status_quo_ramp_power"])
        rows.append(
            "| {label} | {pre} | {status} | {policy} | {reduction:.3f}% |".format(
                label=label,
                pre=fmt(audit["deferrable_pre_service"], 9),
                status=fmt(audit["status_quo_ramp_power"], 10),
                policy=fmt(audit["policy_ramp_power"], 10),
                reduction=reduction,
            )
        )
    return "\n".join(rows)


def render_robustness_table(evidence: dict[str, Any]) -> str:
    rows = [
        "| Post-selection analysis | Scope | Ramp impact | DA cost ratio | Every market negative | Strict gates |",
        "|---|---|---:|---:|---|---|",
    ]
    for key, scope in (
        ("one_gw_total", "1 GW fleet-total scale sensitivity"),
        ("c_h_overlapping", "c-h overlapping/non-independent robustness"),
    ):
        result = evidence["robustness"][key]["result"]
        rows.append(
            f"| `{key}` | {scope} | {fmt(result['mean_incremental_ramp_impact'], 12)} | "
            f"{fmt(result['energy_cost_ratio'], 12)} | true | true |"
        )
    return "\n".join(rows)


def render_tokens(evidence: dict[str, Any]) -> dict[str, str]:
    validation = evidence["validation"]["result"]
    test = evidence["test"]["result"]
    ledger = build_claim_ledger(evidence)
    ledger_text = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    ledger_sha = hashlib.sha256(ledger_text.encode("utf-8")).hexdigest()
    return {
        "PROTOCOL_ID": f"`{EXPECTED_PROTOCOL_ID}`",
        "PROTOCOL_SHA256": f"`{EXPECTED_PROTOCOL_SHA256}`",
        "SOURCE_COMMIT": f"`{EXPECTED_SOURCE_COMMIT}`",
        "CANONICAL_EVIDENCE_SHA256": f"`{EXPECTED_CANONICAL_SHA256}`",
        "CLAIM_LEDGER_SHA256": f"`{ledger_sha}`",
        "SPLIT_RESULTS_TABLE": render_split_table(evidence),
        "PER_MARKET_TABLE": render_market_table(evidence),
        "PHYSICAL_RAMP_TABLE": render_physical_table(evidence),
        "BEHAVIOR_TABLE": render_behavior_table(evidence),
        "ROBUSTNESS_TABLE": render_robustness_table(evidence),
        "VALIDATION_DAY_BOOTSTRAP": (
            f"95% day-block interval "
            f"[{fmt(validation['bootstrap_by_day']['lower_95'], 11)}, "
            f"{fmt(validation['bootstrap_by_day']['upper_95'], 11)}] "
            f"from {validation['bootstrap_by_day']['draws']} draws"
        ),
        "TEST_DAY_BOOTSTRAP": (
            f"95% day-block interval "
            f"[{fmt(test['bootstrap_by_day']['lower_95'], 11)}, "
            f"{fmt(test['bootstrap_by_day']['upper_95'], 11)}] "
            f"from {test['bootstrap_by_day']['draws']} draws"
        ),
        "TEST_MONTH_BOOTSTRAP": (
            f"95% month-block interval "
            f"[{fmt(test['bootstrap_by_month']['lower_95'], 11)}, "
            f"{fmt(test['bootstrap_by_month']['upper_95'], 11)}] "
            f"from {test['bootstrap_by_month']['draws']} draws"
        ),
        "FINAL_VERDICT": "`sealed_test_success`",
    }


def materialize_text(source: str, evidence: dict[str, Any]) -> str:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != EXPECTED_TEMPLATE_SHA256:
        raise ValueError(
            "source does not match the frozen V4R thesis template; "
            f"expected {EXPECTED_TEMPLATE_SHA256}, got {digest}"
        )
    begin = source.index("<!-- data-result-contract-begin -->")
    end = source.index("<!-- data-result-contract-end -->")
    replacements = render_tokens(evidence)
    for name in replacements:
        token = f"{{{{CANONICAL_V4R:{name}}}}}"
        if token not in source:
            raise ValueError(f"source does not contain required token {token}")
        if not begin < source.index(token) < end:
            raise ValueError(f"canonical result token is outside contract block: {token}")
    for name, rendered in replacements.items():
        source = source.replace(f"{{{{CANONICAL_V4R:{name}}}}}", rendered)
    errors = validate_text(source, allow_placeholders=False, evidence=evidence)
    if errors:
        raise ValueError("; ".join(errors))
    return source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--results", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = load_verified_evidence(args.results)
    text = materialize_text(args.source.read_text(encoding="utf-8"), evidence)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
