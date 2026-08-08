"""Common-calendar discovery and sealed split rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .contract import ContractError

CANDIDATE_START = pd.Timestamp("2025-09-01T00:00:00Z")
CANDIDATE_END = pd.Timestamp("2026-05-01T00:00:00Z")


@dataclass(frozen=True)
class StudyCalendar:
    start_utc: str
    end_utc: str
    train_start_utc: str
    train_end_utc: str
    validation_start_utc: str
    validation_end_utc: str
    test_start_utc: str
    test_end_utc: str
    rows: int
    discovery: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_calendar(
    start: pd.Timestamp = CANDIDATE_START,
    end: pd.Timestamp = CANDIDATE_END,
    *,
    discovery: str = "locked_candidate",
) -> StudyCalendar:
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    months = pd.date_range(start, end, freq="MS", inclusive="left")
    if len(months) != 8 or start.day != 1 or end.day != 1:
        raise ContractError("v3 calendar must contain eight complete UTC months")
    train_end = months[5]
    validation_end = months[6]
    return StudyCalendar(
        start_utc=start.isoformat(),
        end_utc=end.isoformat(),
        train_start_utc=start.isoformat(),
        train_end_utc=train_end.isoformat(),
        validation_start_utc=train_end.isoformat(),
        validation_end_utc=validation_end.isoformat(),
        test_start_utc=validation_end.isoformat(),
        test_end_utc=end.isoformat(),
        rows=len(pd.date_range(start, end, freq="1h", inclusive="left")),
        discovery=discovery,
    )


def complete_months(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    utc = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    if not utc.is_unique:
        raise ContractError("coverage index has duplicates")
    result: set[pd.Timestamp] = set()
    for period in utc.tz_localize(None).to_period("M").unique():
        start = pd.Timestamp(period.start_time, tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        expected = pd.date_range(start, end, freq="1h", inclusive="left")
        actual = utc[(utc >= start) & (utc < end)]
        if actual.equals(expected):
            result.add(start)
    return result


def discover_latest_calendar(
    coverage: dict[str, pd.DatetimeIndex],
    *,
    retrieved_at: pd.Timestamp,
) -> StudyCalendar:
    if not coverage:
        raise ContractError("no market coverage supplied")
    intersection: set[pd.Timestamp] | None = None
    for market, index in coverage.items():
        months = complete_months(index)
        if not months:
            raise ContractError(f"{market} has no complete hourly UTC month")
        intersection = months if intersection is None else intersection & months
    common = sorted(intersection or set())
    cutoff = pd.Timestamp(retrieved_at).tz_convert("UTC") - pd.Timedelta(days=90)
    candidates: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start in common:
        end = start + pd.offsets.MonthBegin(8)
        needed = {
            start + pd.offsets.MonthBegin(offset) for offset in range(8)
        }
        if needed.issubset(common) and end <= cutoff:
            candidates.append((start, end))
    if not candidates:
        raise ContractError(
            "all-six intersection lacks eight consecutive complete months "
            "ending at least 90 days before retrieval"
        )
    start, end = max(candidates, key=lambda item: item[1])
    return make_calendar(start, end, discovery="latest_complete_intersection")
