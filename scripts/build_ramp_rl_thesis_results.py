"""Audit ramp-v6 evidence and build hash-bound thesis result artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ramp_rl.final_aggregation import audit_manifest, generate_package  # noqa: E402
from ramp_rl.v4r_thesis import (  # noqa: E402
    DEFAULT_CANONICAL,
    DEFAULT_OUTPUT,
    load_verified_evidence,
    write_publication_package,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--v4r",
        action="store_true",
        help="Build the final V4R package from canonical recovered-policy evidence.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate the evidence package without writing report artifacts.",
    )
    args = parser.parse_args()
    if args.v4r:
        canonical = (args.manifest or DEFAULT_CANONICAL).resolve()
        evidence = load_verified_evidence(canonical)
        if args.audit_only:
            print(
                json.dumps(
                    {
                        "passed": True,
                        "protocol_id": evidence["protocol_id"],
                        "canonical_evidence_sha256": evidence["_verified"][
                            "canonical_sha256"
                        ],
                        "sealed_test_open_count": evidence["sealed_test_open_count"],
                    },
                    sort_keys=True,
                )
            )
            return
        print(
            json.dumps(
                write_publication_package(
                    evidence,
                    (args.output or DEFAULT_OUTPUT).resolve(),
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.manifest is None:
        parser.error("manifest is required unless --v4r is used")
    if args.output is None:
        parser.error("--output is required for legacy manifest packages")
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
