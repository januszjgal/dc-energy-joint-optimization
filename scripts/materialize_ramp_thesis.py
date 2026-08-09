"""Materialize pending v2 thesis sections from canonical machine evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.validate_ramp_thesis import validate_text
except ModuleNotFoundError:
    from validate_ramp_thesis import validate_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "thesis_ramp_v6.md"
DEFAULT_OUTPUT = ROOT / "build" / "thesis_ramp_v6_materialized.md"
EXPECTED_SCHEMA = "ramp-v6-thesis-results-v1"
EXPECTED_PROTOCOL_COMMIT = "4d47a9a"
EXPECTED_TEMPLATE_SHA256 = (
    "14855e7553ab8be21e94f71a3e7cbef43e3e0e3070a9e7955f571d2ba76d05d4"
)
TOP_LEVEL_KEYS = {
    "schema_version",
    "canonical",
    "generated",
    "protocol_commit",
    "source_commit",
    "campaign_status",
    "selection_split",
    "validation_rows",
    "selection",
    "sealed_test",
    "test_rows",
    "statistics",
    "verdict_code",
}
ROW_KEYS = {
    "candidate_id",
    "seed_count",
    "mean_incremental_ramp_impact",
    "mean_da_cost_ratio",
    "exact_safety_all_seeds",
    "all_strict_gates_pass",
}
SELECTION_KEYS = {"selected", "candidate_id", "failed_gates"}
SEALED_TEST_KEYS = {"opened", "selection_or_tuning_used"}
STATISTICS_KEYS = {
    "optimizer_seed_count",
    "day_units",
    "month_units",
    "interval_method",
}
ALLOWED_STATUSES = {
    "validation_blocker_no_selected_protocol",
    "validation_selected_test_pending",
    "campaign_complete",
}
VERDICTS_BY_STATUS = {
    "validation_blocker_no_selected_protocol": {
        "validation_blocker_no_selected_protocol"
    },
    "validation_selected_test_pending": {"validation_selected_test_pending"},
    "campaign_complete": {"sealed_test_success", "sealed_test_failure"},
}


def require(mapping: dict[str, Any], key: str, expected_type: type) -> Any:
    value = mapping.get(key)
    if expected_type in (bool, int, float) and isinstance(value, bool):
        if expected_type is not bool:
            raise ValueError(f"{key} must be {expected_type.__name__}")
    if not isinstance(value, expected_type):
        raise ValueError(f"{key} must be {expected_type.__name__}")
    return value


def require_exact_keys(
    mapping: dict[str, Any],
    expected: set[str],
    *,
    field: str,
) -> None:
    extra = set(mapping) - expected
    missing = expected - set(mapping)
    if extra or missing:
        raise ValueError(
            f"{field} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def require_nonnegative_int(mapping: dict[str, Any], key: str) -> int:
    value = require(mapping, key, int)
    if value < 0:
        raise ValueError(f"{key} must be nonnegative")
    return value


def fmt_number(value: Any, digits: int = 8) -> str:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"Expected numeric value, got {value!r}")
    return f"{value:.{digits}g}"


def verify_git_commit(commit: str, *, field: str) -> None:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"{field} does not resolve to a commit in this repository")


def render_result_table(rows: list[Any], *, label: str) -> str:
    if not rows:
        return f"No {label.lower()} result rows are authorized in this package."
    rendered = [
        "| Candidate | Seeds | Mean ramp impact | Mean DA cost ratio | Exact safety | All strict gates |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("validation_rows entries must be objects")
        require_exact_keys(row, ROW_KEYS, field=f"{label} row")
        candidate_id = require(row, "candidate_id", str)
        if not candidate_id:
            raise ValueError(f"{label} candidate_id must be non-empty")
        seed_count = require(row, "seed_count", int)
        if seed_count < 1:
            raise ValueError(f"{label} seed_count must be at least one")
        rendered.append(
            "| {candidate} | {seeds} | {ramp} | {cost} | {safety} | {gates} |".format(
                candidate=candidate_id,
                seeds=seed_count,
                ramp=fmt_number(row.get("mean_incremental_ramp_impact")),
                cost=fmt_number(row.get("mean_da_cost_ratio")),
                safety=str(require(row, "exact_safety_all_seeds", bool)).lower(),
                gates=str(require(row, "all_strict_gates_pass", bool)).lower(),
            )
        )
    return "\n".join(rendered)


def render_tokens(payload: dict[str, Any]) -> dict[str, str]:
    require_exact_keys(payload, TOP_LEVEL_KEYS, field="top-level")
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError(f"schema_version must be {EXPECTED_SCHEMA}")
    if payload.get("canonical") is not True:
        raise ValueError("canonical must be true")
    if payload.get("generated") is not True:
        raise ValueError("generated must be true")
    commit = require(payload, "protocol_commit", str)
    if re.fullmatch(r"4d47a9a[0-9a-f]{0,33}", commit) is None:
        raise ValueError(f"protocol_commit must identify {EXPECTED_PROTOCOL_COMMIT}")
    verify_git_commit(commit, field="protocol_commit")
    source_commit = require(payload, "source_commit", str)
    if re.fullmatch(r"[0-9a-f]{7,40}", source_commit) is None:
        raise ValueError("source_commit must be a 7-40 character lowercase git hash")
    verify_git_commit(source_commit, field="source_commit")
    if require(payload, "selection_split", str) != "validation":
        raise ValueError("selection_split must be validation")

    validation_rows = require(payload, "validation_rows", list)
    if not validation_rows:
        raise ValueError("validation_rows must contain at least one candidate")
    validation_candidate_ids = [
        require(row, "candidate_id", str)
        for row in validation_rows
        if isinstance(row, dict)
    ]
    if len(validation_candidate_ids) != len(set(validation_candidate_ids)):
        raise ValueError("validation candidate_id values must be unique")
    test_rows = require(payload, "test_rows", list)
    sealed = require(payload, "sealed_test", dict)
    require_exact_keys(sealed, SEALED_TEST_KEYS, field="sealed_test")
    opened = require(sealed, "opened", bool)
    selection = require(payload, "selection", dict)
    require_exact_keys(selection, SELECTION_KEYS, field="selection")
    selected = require(selection, "selected", bool)
    if opened and not selected:
        raise ValueError("sealed test cannot be opened without a selected protocol")
    if require(sealed, "selection_or_tuning_used", bool):
        raise ValueError("sealed test cannot be used for selection or tuning")
    if opened != bool(test_rows):
        raise ValueError("test_rows must be non-empty exactly when sealed test is opened")

    status = require(payload, "campaign_status", str)
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"unsupported campaign_status: {status}")
    if status == "validation_blocker_no_selected_protocol" and (selected or opened):
        raise ValueError("validation blocker cannot select a protocol or open test")
    if status == "validation_selected_test_pending" and (not selected or opened):
        raise ValueError("test-pending status requires selection with unopened test")
    if status == "campaign_complete" and (not selected or not opened):
        raise ValueError("campaign_complete requires selection and opened test")
    if status == "campaign_complete" and len(test_rows) != 1:
        raise ValueError("campaign_complete requires exactly one selected test row")
    failed = selection.get("failed_gates", [])
    if not isinstance(failed, list) or not all(isinstance(item, str) for item in failed):
        raise ValueError("selection.failed_gates must be an array of strings")
    candidate_id = require(selection, "candidate_id", str)
    if selected:
        candidates = {
            require(row, "candidate_id", str): row
            for row in validation_rows
            if isinstance(row, dict)
        }
        if not candidate_id:
            raise ValueError("selected candidate_id must be non-empty")
        if candidate_id not in candidates:
            raise ValueError("selected candidate_id must appear in validation_rows")
        if require(candidates[candidate_id], "all_strict_gates_pass", bool) is not True:
            raise ValueError("selected candidate must pass all strict validation gates")
        if failed:
            raise ValueError("selected protocol cannot retain failed validation gates")
        for row in test_rows:
            if not isinstance(row, dict):
                raise ValueError("test_rows entries must be objects")
            if require(row, "candidate_id", str) != candidate_id:
                raise ValueError(
                    "every sealed-test row must identify the selected candidate"
                )
    elif candidate_id:
        raise ValueError("unselected candidate_id must be empty")
    decision = (
        f"Selected protocol: `{candidate_id}`."
        if selected
        else "No protocol selected. Failed gates: "
        + (", ".join(f"`{item}`" for item in failed) or "none recorded")
        + "."
    )
    sealed_status = (
        "The sealed test was opened once after validation-only selection and was "
        "not used for selection or tuning."
        if opened
        else "The March-April 2026 sealed test remained unopened."
    )

    statistics = require(payload, "statistics", dict)
    require_exact_keys(statistics, STATISTICS_KEYS, field="statistics")
    optimizer_seed_count = require_nonnegative_int(
        statistics,
        "optimizer_seed_count",
    )
    if optimizer_seed_count < 1:
        raise ValueError("optimizer_seed_count must be at least one")
    day_units = require_nonnegative_int(statistics, "day_units")
    month_units = require_nonnegative_int(statistics, "month_units")
    interval_method = require(statistics, "interval_method", str)
    if not interval_method:
        raise ValueError("interval_method must be non-empty")
    summary = (
        f"Optimizer seeds: {optimizer_seed_count}. "
        f"Day-bootstrap unit count: {day_units}. "
        f"Month-bootstrap unit count: {month_units}. "
        f"Interval method: {interval_method}."
    )
    verdict = require(payload, "verdict_code", str)
    if verdict not in VERDICTS_BY_STATUS[status]:
        raise ValueError(
            f"verdict_code {verdict!r} is invalid for campaign_status {status!r}"
        )
    if status == "campaign_complete":
        test_passed = require(test_rows[0], "all_strict_gates_pass", bool)
        if verdict == "sealed_test_success" and not test_passed:
            raise ValueError("sealed_test_success requires all strict test gates")
        if verdict == "sealed_test_failure" and test_passed:
            raise ValueError("sealed_test_failure requires at least one failed test gate")
    return {
        "CAMPAIGN_STATUS": f"`{status}`",
        "VALIDATION_RESULTS_TABLE": render_result_table(
            validation_rows,
            label="Validation",
        ),
        "SELECTION_DECISION": decision,
        "SEALED_TEST_STATUS": sealed_status,
        "TEST_RESULTS_TABLE": render_result_table(test_rows, label="Sealed test"),
        "STATISTICAL_SUMMARY": summary,
        "FINAL_VERDICT": f"`{verdict}`",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    replacements = render_tokens(payload)
    text = args.source.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != EXPECTED_TEMPLATE_SHA256:
        raise ValueError(
            "source does not match the frozen ramp-thesis template; "
            f"expected {EXPECTED_TEMPLATE_SHA256}, got {digest}"
        )
    begin = text.index("<!-- data-result-contract-begin -->")
    end = text.index("<!-- data-result-contract-end -->")
    for name, rendered in replacements.items():
        token = f"{{{{CANONICAL_V2:{name}}}}}"
        if token not in text:
            raise ValueError(f"Source does not contain required token {token}")
        if not begin < text.index(token) < end:
            raise ValueError(f"Canonical result token is outside contract block: {token}")
        text = text.replace(token, rendered)
    errors = validate_text(text, allow_placeholders=False)
    if errors:
        raise ValueError("; ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
