"""Validate the final V4R thesis and its generated-result boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / "thesis_ramp_v6.md"
PLACEHOLDER_PATTERN = re.compile(r"\{\{CANONICAL_V4R:([A-Z0-9_]+)\}\}")
REQUIRED_PLACEHOLDERS = {
    "PROTOCOL_ID",
    "PROTOCOL_SHA256",
    "SOURCE_COMMIT",
    "HASH_CONTRACT_ID",
    "CANONICAL_REPRESENTATION",
    "SEALED_CANONICAL_EVIDENCE_SHA256",
    "CHECKPOINT_RECOVERY_SHA256",
    "CANONICAL_EVIDENCE_SHA256",
    "CORRECTION_CLASSIFICATION",
    "SPLIT_RESULTS_TABLE",
    "PER_MARKET_TABLE",
    "STATUS_QUO_COMPARISON_TABLE",
    "PHYSICAL_RAMP_TABLE",
    "DECODER_ADJUSTMENT_TABLE",
    "COST_COMPARISON_TABLE",
    "BEHAVIOR_TABLE",
    "ROBUSTNESS_TABLE",
    "VALIDATION_DAY_BOOTSTRAP",
    "TEST_DAY_BOOTSTRAP",
    "TEST_MONTH_BOOTSTRAP",
    "FINAL_VERDICT",
}
REQUIRED_SCOPE_PHRASES = {
    "wholesale day-ahead LMP": "wholesale-price scope",
    "not a retail electricity bill": "retail-bill disclaimer",
    "market-level": "market geography scope",
    "PJM DOM / Northern Virginia was not evaluated": "Virginia exclusion",
    "price-taking": "scale limitation",
    "reconstructed causal forecasts": "forecast provenance",
    "No market series is interpolated": "no-interpolation rule",
    "March-April 2026 sealed test": "sealed-test identity",
    "c-h overlapping/non-independent": "overlapping robustness label",
    "new binary identity": "recovery identity",
    "no new V4R training": "V4R training distinction",
    "same-balancing-authority EIA bulk fallback": "physical-source fallback scope",
    "provenance-hash-chain-only": "provenance supersession scope",
    "Git-blob bytes": "Git-blob identity scope",
    "working-tree bytes": "working-tree identity scope",
    "price-taking scale sensitivity": "1 GW sensitivity scope",
    "five of six markets": "status-quo comparison scope",
    "MISO is the exception": "status-quo exception",
    "not a second sealed generalization test": "post-hoc correction scope",
    "Demand charges, retail tariffs, and facility bill savings are explicitly out of scope": "demand-charge scope",
    "no untrained/random PPO comparator": "RL-necessity limitation",
    "unavailable from the original sealed traces": "physical erratum boundary",
    "literal-zero decoder field is invalid placeholder telemetry": "decoder erratum boundary",
    "March-April row is not a sealed test": "replay naming boundary",
    "This comparator erratum requires no policy replay": "trace-derived comparator boundary",
    "34,848 market-hours": "market-hour count",
    "114 training daily windows": "factory window count",
    "186 scalar values": "observation dimension",
    "1,620 ensemble decisions": "controller decision count",
    "2, 1, 2, 3, 4, and 3 hours": "a-f deadline horizons",
    "The `dede685` worktree executes only": "training/verifier context separation",
    "original live_v3 training manifests": "stable verifier evidence context",
}
PROHIBITED_CLAIMS = (
    re.compile(r"\bc-h\b.{0,40}\b(?:(?<!non-)independent|holdout|out-of-fold)\b", re.I),
    re.compile(r"\bPJM\b.{0,25}\b(?:evaluated|result|improved|reduced)\b", re.I),
    re.compile(r"\bV4R\b.{0,40}\b(?:trained|training run|randomly initialized)\b", re.I),
    re.compile(r"\bretail (?:cost|bill) savings?\b", re.I),
    re.compile(r"\b(?:validation|February).{0,80}(?:pooled|combined|blended).{0,40}(?:test|March-April)\b", re.I),
    re.compile(r"\bcounter-ramp(?:ing)?\b", re.I),
    re.compile(r"\b(?:six )?independent markets?\b", re.I),
    re.compile(r"\bb1742a2e753d9a899be256667c80679cbfcf2da4cf6056a6b471d42e66ee7b30\b", re.I),
    re.compile(r"\|\s*Original V4\s*\|\s*(?:0(?:\.0+)?|[-+]?\d)", re.I),
)


def validate_text(
    text: str,
    *,
    allow_placeholders: bool,
    evidence: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    normalized_text = re.sub(r"\s+", " ", text)
    tokens = set(PLACEHOLDER_PATTERN.findall(text))
    if allow_placeholders:
        missing = REQUIRED_PLACEHOLDERS - tokens
        extra = tokens - REQUIRED_PLACEHOLDERS
        if missing:
            errors.append("missing required canonical placeholders: " + ", ".join(sorted(missing)))
        if extra:
            errors.append("unsupported canonical placeholders: " + ", ".join(sorted(extra)))
    elif tokens:
        errors.append("unresolved canonical placeholders: " + ", ".join(sorted(tokens)))
    for phrase, purpose in REQUIRED_SCOPE_PHRASES.items():
        if phrase not in normalized_text:
            errors.append(f"missing {purpose}: {phrase!r}")
    for pattern in PROHIBITED_CLAIMS:
        if match := pattern.search(normalized_text):
            errors.append(f"unsupported claim: {match.group(0)!r}")
    if text.count("data-result-contract-begin") != 1:
        errors.append("generated-result contract must have exactly one begin marker")
    if text.count("data-result-contract-end") != 1:
        errors.append("generated-result contract must have exactly one end marker")
    if evidence is not None:
        if evidence.get("sealed_test_open_count") != 1:
            errors.append("sealed test must have exactly one opening")
        if evidence.get("test", {}).get("result", {}).get("success_gate", {}).get("passed") is not True:
            errors.append("unsupported final success claim")
        if not tokens:
            from scripts.materialize_ramp_thesis import render_tokens

            try:
                begin = text.index("<!-- data-result-contract-begin -->")
                end = text.index("<!-- data-result-contract-end -->")
            except ValueError:
                errors.append("missing generated-result contract markers")
            else:
                template = DEFAULT_SOURCE.read_text(encoding="utf-8")
                template_begin = template.index("<!-- data-result-contract-begin -->")
                template_end = template.index("<!-- data-result-contract-end -->")
                contract = template[template_begin : template_end + len("<!-- data-result-contract-end -->")]
                for name, rendered in render_tokens(evidence).items():
                    contract = contract.replace(f"{{{{CANONICAL_V4R:{name}}}}}", rendered)
                actual = text[begin : end + len("<!-- data-result-contract-end -->")]
                if actual != contract:
                    errors.append("generated-result contract differs from canonical V4R evidence")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--allow-placeholders", action="store_true")
    parser.add_argument("--results", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = None
    if not args.allow_placeholders or args.results:
        from ramp_rl.v4r_corrected_thesis import (
            DEFAULT_CANONICAL,
            load_verified_evidence,
        )

        evidence = load_verified_evidence(
            (args.results or DEFAULT_CANONICAL).resolve()
        )
    text = args.source.read_text(encoding="utf-8")
    errors = validate_text(
        text,
        allow_placeholders=args.allow_placeholders,
        evidence=evidence,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Validated {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
