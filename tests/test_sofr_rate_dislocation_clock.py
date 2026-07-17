from __future__ import annotations

import csv
import gzip
from datetime import datetime, timezone
from pathlib import Path

import pytest

from training import sofr_rate_dislocation_clock as clock


def test_rate_parser_is_exact_decimal_integer_bp() -> None:
    assert clock._parse_rate_bp("5.00") == 500
    assert clock._parse_rate_bp("0.01") == 1
    with pytest.raises(ValueError, match="integer basis point"):
        clock._parse_rate_bp("5.001")


def test_exact_rank_ties_and_direct_state_flip_create_events() -> None:
    rows = []
    start = datetime(2020, 1, 1, 20, tzinfo=timezone.utc)
    for rate_bp in [100, 101, 102, 103, 105, 103]:
        rows.append(
            clock.SourceRow(
                effective_date=(start.date()),
                available_at=start,
                rate_bp=rate_bp,
            )
        )
        start = start.replace(day=start.day + 1)
    cfg = clock.ClockConfig(
        lookback_observations=3,
        lower_rank_twice_numerator_max=0,
        upper_rank_twice_numerator_min=6,
        execution_delay_bars=1,
        hold_bars=1,
    )
    events = clock.build_events(rows, cfg)
    assert [(event.delta_bp, event.rank_twice_numerator, event.side) for event in events] == [
        (2, 6, -1),
        (-2, 0, 1),
    ]


def test_global_nonoverlap_ignores_events_instead_of_queueing() -> None:
    rows = clock.read_source()
    cfg = clock.ClockConfig(hold_bars=10_000_000)
    events = clock.build_events(rows, cfg)
    assert len(events) == 1


def test_frozen_clock_replays_exact_counts_sides_and_concentration() -> None:
    events = clock.build_events(clock.read_source())
    expected = {
        "train": ("2021-01-01", "2023-01-01", 48, 31, 17, 5, 5 / 48),
        "2021": ("2021-01-01", "2022-01-01", 12, 8, 4, 4, 4 / 12),
        "2022": ("2022-01-01", "2023-01-01", 36, 23, 13, 5, 5 / 36),
        "2023": ("2023-01-01", "2024-01-01", 40, 20, 20, 5, 5 / 40),
        "2023_h1": ("2023-01-01", "2023-07-01", 18, 9, 9, 4, 4 / 18),
        "2023_h2": ("2023-07-01", "2024-01-01", 21, 11, 10, 5, 5 / 21),
    }
    for start, end, count, long, short, month_count, month_share in expected.values():
        summary = clock.event_summary(events, start, end)
        assert summary["count"] == count
        assert summary["long"] == long
        assert summary["short"] == short
        assert summary["max_single_month_count"] == month_count
        assert summary["max_single_month_share"] == month_share


def test_written_clock_is_deterministic_and_contains_no_outcome(tmp_path: Path) -> None:
    cfg = clock.ClockConfig(output=str(tmp_path / "clock.csv.gz"))
    first = clock.write_clock(cfg)
    first_bytes = Path(first["output"]).read_bytes()
    second = clock.write_clock(cfg)
    assert Path(second["output"]).read_bytes() == first_bytes
    assert first == second
    assert first["outcomes_opened"] is False
    assert clock.read_event_ledger(first["output"]) == clock.build_events(
        clock.read_source()
    )
    with gzip.open(first["output"], "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == clock.OUTPUT_COLUMNS
        assert not {"return", "price", "cagr", "drawdown"}.intersection(
            reader.fieldnames or []
        )
