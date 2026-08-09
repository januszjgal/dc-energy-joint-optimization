"""Audit one explicit ramp-v6 result manifest and build thesis result artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ramp_rl.final_aggregation import audit_manifest, generate_package  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate the evidence package without writing report artifacts.",
    )
    args = parser.parse_args()
    evidence = audit_manifest(args.manifest, args.artifact_root)
    if args.audit_only:
        print(
            json.dumps(
                {
                    "passed": True,
                    "label": evidence.manifest["label"],
                    "result_status": evidence.manifest["result_status"],
                },
                sort_keys=True,
            )
        )
        return
    print(json.dumps(generate_package(evidence, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
