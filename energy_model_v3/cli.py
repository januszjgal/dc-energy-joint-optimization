"""Command-line interface for v3 provenance, probes, and fail-closed builds."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .adapters import ADAPTERS, get_adapter
from .builder import MARKETS, blocked_preflight, build_panel
from .calendar import make_calendar
from .contract import ContractError

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "energy_model_v3"
DEFAULT_OUTPUT = ROOT / "output" / "energy_model_v3"
DEFAULT_SCENARIO = (
    ROOT / "env" / "scenarios" / "us_six_market_v3_primary.yaml"
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def command_describe(args: argparse.Namespace) -> int:
    output = Path(args.output)
    descriptors = {
        market: [
            descriptor.to_dict()
            for descriptor in get_adapter(market).descriptors()
        ]
        for market in MARKETS
    }
    write_json(
        output,
        {
            "schema_version": "energy-model-v3",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": descriptors,
        },
    )
    return 0


def command_probe(args: argparse.Namespace) -> int:
    capabilities = [
        get_adapter(market).probe(timeout=args.timeout).to_dict()
        for market in MARKETS
    ]
    write_json(Path(args.output), capabilities)
    unavailable = [item for item in capabilities if item["status"] != "AVAILABLE"]
    return 2 if unavailable else 0


def command_preflight(args: argparse.Namespace) -> int:
    capability_path = Path(args.capabilities)
    capabilities = (
        json.loads(capability_path.read_text(encoding="utf-8"))
        if capability_path.exists()
        else [
            get_adapter(market).capability().to_dict() for market in MARKETS
        ]
    )
    report = blocked_preflight(Path(args.data_root), capabilities)
    write_json(Path(args.output), report)
    return 2 if report["status"] == "BLOCKED" else 0


def command_build(args: argparse.Namespace) -> int:
    scenario = Path(args.scenario)
    calendar = make_calendar()
    index = pd.date_range(
        calendar.start_utc, calendar.end_utc, freq="1h", inclusive="left"
    )
    dc_power: dict[str, pd.Series] = {}
    scenario_data = __import__("yaml").safe_load(
        scenario.read_text(encoding="utf-8")
    )
    for site in scenario_data["sites"]:
        power_path = (
            Path(args.data_root)
            / "modeled_dc_power"
            / f"{site['market']}.csv"
        )
        if not power_path.exists():
            raise ContractError(f"missing modeled DC power: {power_path}")
        power = pd.read_csv(power_path)
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(power["interval_start_utc"], utc=True)
        )
        if not timestamps.equals(index):
            raise ContractError(
                f"{site['market']} modeled DC power index is not exact"
            )
        dc_power[site["market"]] = pd.to_numeric(
            power["dc_power_mw"], errors="raise"
        )
    build_panel(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        scenario_path=scenario,
        dc_power=dc_power,
        calendar=calendar,
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Independent six-market US energy model v3"
    )
    commands = result.add_subparsers(dest="command", required=True)
    describe = commands.add_parser("describe-sources")
    describe.add_argument(
        "--output",
        default=str(DEFAULT_DATA / "provenance" / "source_contract.json"),
    )
    describe.set_defaults(func=command_describe)
    probe = commands.add_parser("probe")
    probe.add_argument("--timeout", type=int, default=30)
    probe.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT / "capabilities.json"),
    )
    probe.set_defaults(func=command_probe)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--data-root", default=str(DEFAULT_DATA))
    preflight.add_argument(
        "--capabilities",
        default=str(DEFAULT_OUTPUT / "capabilities.json"),
    )
    preflight.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT / "blocked-capability-report.json"),
    )
    preflight.set_defaults(func=command_preflight)
    build = commands.add_parser("build")
    build.add_argument("--data-root", default=str(DEFAULT_DATA))
    build.add_argument("--output-root", default=str(DEFAULT_OUTPUT / "panel"))
    build.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    build.set_defaults(func=command_build)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ContractError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
