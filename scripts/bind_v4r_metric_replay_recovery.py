"""Create or verify the tracked V4R metric-replay recovery manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ramp_rl.provenance import canonical_json_file_sha256  # noqa: E402
from ramp_rl.v4r_metric_replay_recovery import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RECOVERY_ROOT,
    build_metric_replay_recovery_manifest,
    verify_metric_replay_recovery_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    bind = subparsers.add_parser("bind")
    bind.add_argument("--raw-verification", type=Path, required=True)
    bind.add_argument(
        "--recovery-root",
        type=Path,
        default=DEFAULT_RECOVERY_ROOT,
    )
    bind.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    check = subparsers.add_parser("verify")
    check.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    check.add_argument("--require-binaries", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "bind":
        manifest = build_metric_replay_recovery_manifest(
            raw_verification_path=args.raw_verification,
            recovery_root=args.recovery_root,
            output_root=args.output_root,
        )
        path = args.output_root.resolve() / DEFAULT_MANIFEST.name
    else:
        path = args.manifest.resolve()
        manifest = verify_metric_replay_recovery_manifest(
            path,
            require_binaries=args.require_binaries,
        )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": path.relative_to(ROOT).as_posix(),
                "canonical_json_sha256": canonical_json_file_sha256(path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
