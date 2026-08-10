"""Build the canonical closeout evidence for the live ramp-RL campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit(revision: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", revision], cwd=ROOT, text=True
    ).strip()


def _market_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    impacts: dict[str, list[float]] = defaultdict(list)
    failed_seeds: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        for market, value in row["per_market_macro"].items():
            impacts[market].append(float(value))
            if float(value) >= 0.0:
                failed_seeds[market].append(int(row["seed"]))
    return {
        market: {
            "mean_incremental_ramp_impact": mean(values),
            "all_seeds_improve": not failed_seeds[market],
            "failed_seeds": failed_seeds[market],
        }
        for market, values in sorted(impacts.items())
    }


def _report(evidence: dict[str, Any]) -> str:
    confirmation = evidence["confirmation"]
    lines = [
        "# Six-market pure-RL campaign closeout",
        "",
        f"**Result:** `{evidence['campaign_status']}`. PPO remained safe and under "
        "the energy budget at 500k, but three of five seeds failed preregistered "
        "validation behavior or per-market ramp gates. No protocol qualified for "
        "sealed-test access.",
        "",
        "## Validation result",
        "",
        "| Stage | Algorithm | Seeds | Mean ramp impact | Mean cost ratio | Exact safety | All gates |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for stage in ("screen", "confirmation"):
        aggregate = evidence[stage]["aggregate"]
        lines.append(
            f"| {stage} | PPO | {aggregate['seed_count']} | "
            f"{aggregate['mean_incremental_ramp_impact']:.12g} | "
            f"{aggregate['mean_energy_cost_ratio']:.10f} | "
            f"{aggregate['all_safety_pass']} | "
            f"{aggregate['all_success_gates_pass']} |"
        )
    lines.extend(
        [
            "",
            f"The 500k validation curve changed by "
            f"**{confirmation['extension_decision']['validation_improvement_pct']:.4f}%** "
            f"against a required **{confirmation['extension_decision']['required_pct']:.1f}%** "
            "improvement, so extension toward 2M was rejected.",
            "",
            "## Confirmation seeds",
            "",
            "| Seed | Ramp impact | Cost ratio | Safety | Success | Failed gates |",
            "|---:|---:|---:|---|---|---|",
        ]
    )
    for row in confirmation["rows"]:
        lines.append(
            f"| {row['seed']} | {row['mean_incremental_ramp_impact']:.12g} | "
            f"{row['energy_cost_ratio']:.10f} | {row['safety_pass']} | "
            f"{row['success_gate_pass']} | "
            f"{', '.join(row['failed_gates']) or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Per-market confirmation ramp",
            "",
            "| Market | Five-seed mean impact | All seeds improve | Failed seeds |",
            "|---|---:|---|---|",
        ]
    )
    for market, row in confirmation["per_market"].items():
        failed = ", ".join(str(seed) for seed in row["failed_seeds"]) or "-"
        lines.append(
            f"| {market} | {row['mean_incremental_ramp_impact']:.12g} | "
            f"{row['all_seeds_improve']} | {failed} |"
        )
    lines.extend(
        [
            "",
            "Cost, exact safety, and behavior are reported at the six-market episode "
            "aggregate because the frozen evaluator did not persist per-market "
            "decompositions for those fields. No values were inferred or synthesized.",
            "",
            "## Sealed test and robustness",
            "",
            "- Mar-Apr 2026 sealed test opened: **false**.",
            "- 1 GW scaling robustness run: **false**.",
            "- Overlapping c-h robustness run: **false**.",
            "",
            "Both robustness studies are post-selection analyses. Running them without "
            "a qualifying base protocol would violate the frozen campaign order.",
            "",
            "## Reproduction commands",
            "",
            "```powershell",
            *evidence["commands"],
            "```",
            "",
            "## Integrity",
            "",
            f"- Live panel manifest SHA-256: `{evidence['frozen_inputs']['live_panel_manifest_sha256']}`",
            f"- Raw acquisition manifest SHA-256: `{evidence['frozen_inputs']['raw_acquisition_manifest_sha256']}`",
            f"- Factory manifest SHA-256: `{evidence['artifacts']['factory_manifest']['sha256']}`",
            f"- Screen summary SHA-256: `{evidence['artifacts']['screen_summary']['sha256']}`",
            f"- Confirmation summary SHA-256: `{evidence['artifacts']['confirmation_summary']['sha256']}`",
            f"- Confirmation evidence index SHA-256: `{evidence['artifacts']['confirmation_evidence_index']['sha256']}`",
            "",
            "All selection and stopping decisions used September-January training and "
            "February validation only. The March-April test split remains sealed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/ramp_rl_v6/live/final_results.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "output/ramp_rl_v6/live/final_report.md",
    )
    args = parser.parse_args()

    factory_path = ROOT / "output/energy_model_v3/ramp_v6/factory_manifest.json"
    screen_path = ROOT / "output/ramp_rl_v6/live/screen_summary.json"
    confirmation_path = ROOT / "output/ramp_rl_v6/live/confirmation_summary.json"
    confirmation_index_path = (
        ROOT / "output/ramp_rl_v6/live/confirmation_evidence_index.json"
    )
    screen = _read(screen_path)
    confirmation = _read(confirmation_path)
    index = _read(confirmation_index_path)
    if not index["passed"]:
        raise RuntimeError("confirmation evidence index did not pass")
    if confirmation["promotion_decision"]["promoted_algorithms"]:
        raise RuntimeError("closeout builder is only valid for an unselected campaign")
    if confirmation["extension_decision"]["allowed"]:
        raise RuntimeError("closeout builder cannot suppress an allowed extension")

    artifacts = {
        "factory_manifest": {
            "path": str(factory_path.relative_to(ROOT)),
            "sha256": _sha256(factory_path),
        },
        "screen_summary": {
            "path": str(screen_path.relative_to(ROOT)),
            "sha256": _sha256(screen_path),
        },
        "confirmation_summary": {
            "path": str(confirmation_path.relative_to(ROOT)),
            "sha256": _sha256(confirmation_path),
        },
        "confirmation_evidence_index": {
            "path": str(confirmation_index_path.relative_to(ROOT)),
            "sha256": _sha256(confirmation_index_path),
        },
    }
    commands = [
        "python scripts\\build_energy_v3_ramp_factory.py",
        "python scripts\\run_ramp_rl_v6.py train --env-factory "
        "env.ramp_v6.factory:make_energy_model_v3_env --algorithm <ppo|sac> "
        "--seed <2601|2602|2603> --timesteps 100000 --n-envs 4 "
        "--output models\\ramp_rl_v6\\live\\screen\\<algorithm>\\<seed>",
        "python scripts\\run_ramp_rl_v6.py train --env-factory "
        "env.ramp_v6.factory:make_energy_model_v3_env --algorithm ppo "
        "--seed <2601|2602|2603|2604|2605> --timesteps 500000 --n-envs 4 "
        "--output models\\ramp_rl_v6\\live\\confirmation\\ppo\\<seed>",
        "$m = Get-Content output\\energy_model_v3\\ramp_v6\\factory_manifest.json "
        "-Raw | ConvertFrom-Json; $a = @('scripts\\run_ramp_rl_v6.py', "
        "'evaluate', '--env-factory', "
        "'env.ramp_v6.factory:make_energy_model_v3_env', '--algorithm', 'ppo', "
        "'--seed', '<seed>', '--output', "
        "'models\\ramp_rl_v6\\live\\confirmation\\ppo\\<seed>', '--split', "
        "'validation'); foreach ($w in ($m.windows.validation.PSObject.Properties.Name "
        "| Sort-Object)) { $a += '--window'; $a += $w }; & python @a",
        "python scripts\\summarize_ramp_rl_stage.py --validation-root "
        "output\\ramp_rl_v6\\live\\confirmation --expected-seeds 5 "
        "--stage confirmation --previous-summary "
        "output\\ramp_rl_v6\\live\\screen_summary.json --output "
        "output\\ramp_rl_v6\\live\\confirmation_summary.json",
        "python scripts\\build_ramp_rl_evidence_v6.py "
        "models\\ramp_rl_v6\\live\\confirmation\\ppo\\2601 "
        "models\\ramp_rl_v6\\live\\confirmation\\ppo\\2602 "
        "models\\ramp_rl_v6\\live\\confirmation\\ppo\\2603 "
        "models\\ramp_rl_v6\\live\\confirmation\\ppo\\2604 "
        "models\\ramp_rl_v6\\live\\confirmation\\ppo\\2605 "
        "--output output\\ramp_rl_v6\\live\\confirmation_evidence_index.json",
    ]
    evidence = {
        "schema_version": "ramp-pure-rl-final-results-v1",
        "campaign_status": "validation_blocker_no_selected_protocol",
        "frozen_inputs": {
            "live_panel_manifest_sha256": (
                "489cb39c61c19952fe90b213c1ea20e4f5eb904563425ec8b751a15ceed446df"
            ),
            "raw_acquisition_manifest_sha256": (
                "64fd78254dabefa8d525f3e43144b2a50bc12c3050bad21efcc8836e3d11b7e7"
            ),
            "train_months": ["2025-09", "2025-10", "2025-11", "2025-12", "2026-01"],
            "validation_months": ["2026-02"],
            "sealed_test_months": ["2026-03", "2026-04"],
        },
        "commits": {
            "factory": _commit("a14efd4"),
            "factory_cache": _commit("aec5331"),
            "screen": _commit("7b21499"),
            "confirmation": _commit("3391440"),
        },
        "screen": {
            "aggregate": screen["aggregates"]["ppo"],
            "promotion_decision": screen["promotion_decision"],
        },
        "confirmation": {
            "aggregate": confirmation["aggregates"]["ppo"],
            "rows": confirmation["rows"],
            "per_market": _market_summary(confirmation["rows"]),
            "promotion_decision": confirmation["promotion_decision"],
            "extension_decision": confirmation["extension_decision"],
            "validation_curve": confirmation["validation_curve"],
        },
        "sealed_test": {
            "opened": False,
            "reason": "no protocol passed five-seed confirmation",
        },
        "robustness": {
            "one_gw_total_run": False,
            "overlapping_c_h_run": False,
            "reason": "post-selection robustness requires a finalized base protocol",
        },
        "metric_scope": {
            "per_market": ["incremental_ramp_impact"],
            "six_market_aggregate": ["energy_cost", "safety", "behavior"],
            "unsynthesized_missing_decompositions": True,
        },
        "artifacts": artifacts,
        "commands": commands,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.report.write_text(_report(evidence), encoding="utf-8")


if __name__ == "__main__":
    main()
