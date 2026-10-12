"""Build the data of every split: frozen statistics, calendar, and panels.

    python scripts/build_data_splits.py

1. Final statistics, fitted on all twelve 2025 months, into
   ``data/four_market_2025/frozen_stats.json``.
2. The calendar of development months, cross-validation folds, and the test
   month, into ``data/four_market_2025/calendar.json``.
3. Final forecasts (fitted on all of 2025) and S_m into the committed 2025
   panels, and the May 2026 test panel built with the same fixed weights.
4. For each cross-validation fold, statistics and forecasts fitted on its nine
   training months, with panels for all twelve months, under
   ``output/four_market_joint_v3/cv-kX/data/``. These are generated, not
   committed; the factory builder calls ``write_fold_data`` when they are
   missing.

The raw grid columns (demand, wind, solar, net load) are the same in every
split; only S_m and the forecast columns differ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from energy_model_v3.splits import (  # noqa: E402
    CV_FOLDS, CV_SPECS, DEVELOPMENT_MONTHS, TEST_FLEET, TEST_MONTHS, spec_months,
    frozen_stats,
)
from env.ramp_v6 import clock  # noqa: E402
from env.ramp_v6.models import HISTORY_HOURS  # noqa: E402
from scripts import build_causal_forecasts as forecasts  # noqa: E402

ARTIFACT_NAMESPACE = "four_market_joint_v3"
DATA_ROOT = ROOT / "data" / "four_market_2025"
CALENDAR_PATH = DATA_ROOT / "calendar.json"
FROZEN_STATS_PATH = DATA_ROOT / "frozen_stats.json"
MARKETS = ("CAISO", "MISO", "SPP", "ISONE")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def month_hours(month_id: str) -> int:
    return clock.month_hours(month_id)


def development_panel_path(month_id: str) -> Path:
    return DATA_ROOT / "months" / month_id / "canonical_panel.csv"


def test_panel_path(month_id: str) -> Path:
    return forecasts.TEST_ROOT / "months" / month_id / "canonical_panel.csv"


def fold_data_root(spec: str) -> Path:
    if spec not in CV_SPECS:
        raise ValueError(f"{spec!r} is not a cross-validation fold")
    return ROOT / "output" / ARTIFACT_NAMESPACE / spec / "data"


def calendar_payload() -> dict[str, Any]:
    def entry(month_id: str, panel: Path, cells: tuple[str, ...]) -> dict[str, Any]:
        return {
            "window_id": month_id,
            "month_id": month_id,
            "month": int(month_id[-2:]),
            "hours": month_hours(month_id),
            "history_hours": HISTORY_HOURS,
            "panel_path": _relative(panel),
            "workload_cells": list(cells),
        }

    return {
        "markets": list(MARKETS),
        "hour_label_convention": "panel timestamps are hour-beginning UTC",
        "month_clock": (
            "months follow the Google trace's clock, a fixed UTC-7; each month "
            "starts at 07:00 UTC on its first day"
        ),
        "episode_unit": (
            "one continuous month on the trace clock; queues and power history "
            "reset only at the month boundary"
        ),
        "history_hours": HISTORY_HOURS,
        "frozen_stats_path": _relative(FROZEN_STATS_PATH),
        "forecast_model": forecasts.VINTAGE_ID,
        "development": [
            entry(month, development_panel_path(month), ("a", "b", "c", "d"))
            for month in DEVELOPMENT_MONTHS
        ],
        "cv_folds": {fold: list(months) for fold, months in CV_FOLDS.items()},
        "test": [
            entry(month, test_panel_path(month), tuple(cell for _, cell in TEST_FLEET))
            for month in TEST_MONTHS
        ],
    }


def write_final_data() -> dict[str, Any]:
    """Final statistics, calendar, committed panel forecasts, and the test panel."""
    panel = forecasts.load_hourly_series()
    stats = frozen_stats(panel, DEVELOPMENT_MONTHS, "four-market-2025-development-stats-v2")
    _write_json(FROZEN_STATS_PATH, stats)
    _write_json(CALENDAR_PATH, calendar_payload())
    models, fallbacks = forecasts.fit_forecasts(panel, DEVELOPMENT_MONTHS)
    touched = forecasts.write_development_panels(
        forecasts.predict(panel, models, fallbacks), stats["gross_q95_mw"]
    )
    test_path = forecasts.write_test_panel(
        forecasts.predict(forecasts.load_test_series(), models, fallbacks), stats["gross_q95_mw"]
    )
    return {"development_panels": touched, "test_panel": _relative(test_path),
            "fallbacks": [list(item[:3]) for item in fallbacks]}


def write_fold_data(spec: str) -> dict[str, Any]:
    """Statistics and forecasts fitted on one fold's training months only."""
    root = fold_data_root(spec)
    train = spec_months(spec)["train"]
    panel = forecasts.load_hourly_series()
    stats = frozen_stats(panel, train, f"four-market-2025-{spec}-stats-v2")
    _write_json(root / "frozen_stats.json", stats)
    models, fallbacks = forecasts.fit_forecasts(panel, train)
    lookup = forecasts.predict(panel, models, fallbacks).set_index(["timestamp_utc", "market_id"])
    vintage = f"{forecasts.VINTAGE_ID}+{spec}"
    panels = {}
    for month in DEVELOPMENT_MONTHS:
        destination = root / "months" / month / "canonical_panel.csv"
        forecasts.write_forecast_columns(
            development_panel_path(month), destination, lookup, vintage, stats["gross_q95_mw"]
        )
        panels[month] = _relative(destination)
    return {"frozen_stats_path": _relative(root / "frozen_stats.json"), "panels": panels,
            "forecast_model": vintage, "fallbacks": [list(item[:3]) for item in fallbacks]}


def main() -> int:
    print(json.dumps({"final": write_final_data()}, indent=2))
    for spec in CV_SPECS:
        result = write_fold_data(spec)
        print(f"{spec}: {len(result['panels'])} panels, fallbacks {result['fallbacks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
