"""Validate the ramp-aware thesis draft and its result/claim boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "thesis_ramp_v6.md"
PLACEHOLDER_PATTERN = re.compile(r"\{\{CANONICAL_V2:([A-Z0-9_]+)\}\}")
REQUIRED_PLACEHOLDERS = {
    "CAMPAIGN_STATUS",
    "VALIDATION_RESULTS_TABLE",
    "SELECTION_DECISION",
    "SEALED_TEST_STATUS",
    "TEST_RESULTS_TABLE",
    "STATISTICAL_SUMMARY",
    "FINAL_VERDICT",
}
REQUIRED_SCOPE_PHRASES = {
    "wholesale day-ahead LMP": "wholesale-price scope",
    "not a retail electricity bill": "retail-bill disclaimer",
    "market-level": "market geography scope",
    "PJM DOM / Northern Virginia was not evaluated": "Virginia exclusion",
    "price-taking 1 GW": "scale stress scope",
    "reconstructed causal forecasts": "forecast provenance",
    "No market series is interpolated": "no-interpolation rule",
    "March-April 2026 sealed test": "sealed-test identity",
}
PROHIBITED_RESULT_CLAIMS = (
    re.compile(r"\bv2 (?:achieved|improved|reduced|increased|outperformed)\b", re.I),
    re.compile(r"\bv2 results? (?:show|shows|showed|demonstrate|demonstrates)\b", re.I),
)


def validate_text(text: str, *, allow_placeholders: bool) -> list[str]:
    errors: list[str] = []
    normalized_text = re.sub(r"\s+", " ", text)
    tokens = set(PLACEHOLDER_PATTERN.findall(text))
    if allow_placeholders:
        missing = REQUIRED_PLACEHOLDERS - tokens
        if missing:
            errors.append(
                "missing required canonical placeholders: "
                + ", ".join(sorted(missing))
            )
    if tokens and not allow_placeholders:
        errors.append(
            "unresolved canonical placeholders: " + ", ".join(sorted(tokens))
        )
    for phrase, purpose in REQUIRED_SCOPE_PHRASES.items():
        if phrase not in normalized_text:
            errors.append(f"missing {purpose}: {phrase!r}")
    for pattern in PROHIBITED_RESULT_CLAIMS:
        if match := pattern.search(normalized_text):
            errors.append(f"unbound v2 result claim: {match.group(0)!r}")
    if "data-result-contract-begin" not in text or "data-result-contract-end" not in text:
        errors.append("missing generated-result contract markers")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Validate a draft whose canonical v2 results are intentionally pending.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = args.source.read_text(encoding="utf-8")
    errors = validate_text(text, allow_placeholders=args.allow_placeholders)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Validated {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
