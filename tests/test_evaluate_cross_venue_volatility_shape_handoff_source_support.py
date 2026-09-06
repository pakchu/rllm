from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import itertools
import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pytest

from training import cross_venue_volatility_shape_handoff as mechanism
from training import evaluate_cross_venue_volatility_shape_handoff_source_support as s


UTC = timezone.utc
HOUR = timedelta(hours=1)


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _format(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _candle(
    open_: str = "100",
    high: str = "112",
    low: str = "99",
    close: str = "110",
) -> mechanism.Candle:
    return mechanism.Candle.from_tokens(
        open=open_,
        high=high,
        low=low,
        close=close,
    )


def _joined_hour(
    close_time: datetime,
    *,
    bvol: mechanism.Candle | None = None,
    dvol: mechanism.Candle | None = None,
    valid: bool = True,
) -> mechanism.JoinedHour:
    return mechanism.JoinedHour(
        close_time=close_time,
        available_at=close_time,
        bvol=_candle() if bvol is None and valid else bvol,
        dvol=_candle("100", "101", "97", "98") if dvol is None and valid else dvol,
        source_valid=valid,
    )


def _event(
    entry: datetime,
    *,
    control: str = mechanism.PRIMARY,
    side: mechanism.Side = "LONG",
    serial: int = 0,
) -> mechanism.ScheduledEvent:
    signal = entry - timedelta(minutes=5)
    return mechanism.ScheduledEvent(
        signal_id=f"CVVH-432|{control}|synthetic-{serial:04d}",
        control=control,
        signal_time=signal,
        available_at=signal,
        entry_time=entry,
        exit_time=entry + mechanism.HOLD_TIME,
        side=side,
    )


def _write_gzip_csv(
    path: Path,
    header: Sequence[str],
    rows: Iterable[Mapping[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(header),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _small_calendar() -> s.SourceCalendar:
    start = _time("2024-01-01T00:00:00Z")
    return s.SourceCalendar(
        start=start,
        end=start + timedelta(hours=3),
        mechanism_end=start + timedelta(hours=3),
    )


def _bvol_row(
    date: datetime,
    *,
    valid: bool = True,
    source_rows: str = "3600",
    complete: str = "true",
    reason: str | None = None,
) -> dict[str, str]:
    available = date + HOUR
    row = {
        "date": _format(date),
        "feature_available_time_utc": _format(available),
        "trade_earliest_time_utc": _format(available),
        "open": "100",
        "high": "112",
        "low": "99",
        "close": "110",
        "source_rows": source_rows,
        "source_complete": complete,
        "feature_valid": "true" if valid else "false",
        "feature_invalid_reason": reason if reason is not None else ("ok" if valid else "gap"),
    }
    if not valid:
        for name in ("open", "high", "low", "close"):
            row[name] = ""
    return row


def _dvol_row(date: datetime) -> dict[str, str]:
    return {
        "date": _format(date),
        "close_time": _format(date + HOUR),
        "open": "100",
        "high": "101",
        "low": "97",
        "close": "98",
    }


def _write_small_sources(
    root: Path,
    *,
    bvol_rows: Sequence[Mapping[str, str]] | None = None,
    dvol_rows: Sequence[Mapping[str, str]] | None = None,
    bvol_header: Sequence[str] | None = None,
    dvol_header: Sequence[str] | None = None,
) -> tuple[s.Config, s.SourceCalendar]:
    calendar = _small_calendar()
    bvol = list(
        bvol_rows
        if bvol_rows is not None
        else [
            _bvol_row(calendar.start),
            _bvol_row(calendar.start + HOUR, valid=False),
            _bvol_row(calendar.start + 2 * HOUR),
        ]
    )
    dvol = list(
        dvol_rows
        if dvol_rows is not None
        else [
            _dvol_row(calendar.start + index * HOUR)
            for index in range(calendar.dvol_rows)
        ]
    )
    cfg = replace(
        s.Config(),
        bvol="synthetic/bvol.csv.gz",
        dvol="synthetic/dvol.csv.gz",
    )
    _write_gzip_csv(
        root / cfg.bvol,
        bvol_header
        or s.prereg.SOURCE_ARTIFACTS["binance_btc_bvol_hourly"]["header"],
        bvol,
    )
    _write_gzip_csv(
        root / cfg.dvol,
        dvol_header
        or s.prereg.SOURCE_ARTIFACTS["deribit_btc_dvol_hourly"]["header"],
        dvol,
    )
    return cfg, calendar


def _brute_force_matching(
    left: Sequence[datetime],
    right: Sequence[datetime],
    tolerance: timedelta,
) -> tuple[tuple[datetime, datetime], ...]:
    """Independent oracle: enumerate every one-to-one edge assignment."""

    tolerance_us = (
        (tolerance.days * 86_400 + tolerance.seconds) * 1_000_000
        + tolerance.microseconds
    )
    candidates: list[
        tuple[int, int, tuple[tuple[datetime, datetime], ...]]
    ] = []
    for size in range(min(len(left), len(right)) + 1):
        for left_subset in itertools.combinations(left, size):
            for right_subset in itertools.combinations(right, size):
                for right_assignment in itertools.permutations(right_subset):
                    pairs = tuple(sorted(zip(left_subset, right_assignment)))
                    lags = [
                        abs(
                            (
                                (a - b).days * 86_400
                                + (a - b).seconds
                            )
                            * 1_000_000
                            + (a - b).microseconds
                        )
                        for a, b in pairs
                    ]
                    if all(lag <= tolerance_us for lag in lags):
                        candidates.append((-size, sum(lags), pairs))
    return min(candidates)[2]


def _support_primary() -> tuple[mechanism.ScheduledEvent, ...]:
    entries: list[datetime] = []
    entries.extend(
        _time("2023-09-01T00:00:00Z") + index * timedelta(days=8)
        for index in range(3)
    )
    entries.extend(
        _time("2023-12-01T00:00:00Z") + index * timedelta(days=3)
        for index in range(9)
    )
    for month in range(1, 7):
        entries.extend(
            datetime(2024, month, day, tzinfo=UTC) for day in (2, 12)
        )
    for month in range(7, 13):
        count = 6 if month == 7 else 3
        entries.extend(
            datetime(2024, month, 2 + index * 4, tzinfo=UTC)
            for index in range(count)
        )
    for month in range(1, 11):
        entries.extend(
            datetime(2025, month, day, tzinfo=UTC) for day in (2, 10, 18)
        )
    for month in range(1, 6):
        entries.extend(
            datetime(2026, month, day, tzinfo=UTC) for day in (2, 10, 18)
        )
    assert len(entries) == 90

    events: list[mechanism.ScheduledEvent] = []
    for index, entry in enumerate(entries):
        if index < 45:
            side = "LONG" if index in set(range(0, 42, 3)) else "SHORT"
        elif index < 75:
            side = "LONG" if (index - 45) % 4 == 0 else "SHORT"
        else:
            side = "LONG" if (index - 75) % 4 == 0 else "SHORT"
        events.append(_event(entry, side=side, serial=index))
    return tuple(sorted(events, key=lambda row: row.entry_time))


def _all_support_clocks(
    primary: Sequence[mechanism.ScheduledEvent] | None = None,
) -> dict[str, tuple[mechanism.ScheduledEvent, ...]]:
    primary_clock = tuple(primary or _support_primary())
    clocks: dict[str, tuple[mechanism.ScheduledEvent, ...]] = {
        mechanism.PRIMARY: primary_clock
    }
    for offset, control in enumerate(mechanism.INDEPENDENT_CONTROLS, start=2):
        clocks[control] = tuple(
            replace(
                row,
                signal_id=f"{row.signal_id}|{control}",
                control=control,
                signal_time=row.signal_time + timedelta(days=offset),
                available_at=row.available_at + timedelta(days=offset),
                entry_time=row.entry_time + timedelta(days=offset),
                exit_time=row.exit_time + timedelta(days=offset),
            )
            for row in primary_clock
        )
    for control in mechanism.PARENT_SET_CONTROLS:
        clocks[control] = ()
    return clocks


def _fake_identity() -> dict[str, Any]:
    seal = {"synthetic.py": {"git_blob": "a" * 40, "sha256": "b" * 64}}
    return {
        "repository": {"commit": "c" * 40, "tree": "d" * 40},
        "preregistered_protocol_seal_hash": "e" * 64,
        "evaluator_seal": seal,
        "evaluator_seal_hash": s.prereg.canonical_hash(seal),
    }


def _patch_authoritative_preclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "load_registration", lambda repo, cfg: {})
    monkeypatch.setattr(
        s,
        "evaluator_identity",
        lambda repo, registration: _fake_identity(),
    )
    monkeypatch.setattr(
        s,
        "validate_source_bindings",
        lambda repo, cfg: s.CapturedSources(
            bindings={
                "synthetic": {
                    "path": cfg.bvol,
                    "sha256": "e" * 64,
                }
            },
            bvol_compressed=b"synthetic-bvol",
            dvol_compressed=b"synthetic-dvol",
        ),
    )


def _directory_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_matching_matches_independent_exhaustive_oracle() -> None:
    base = _time("2024-01-01T00:00:00Z")
    points = tuple(base + index * HOUR for index in range(5))
    for left_size in range(1, 4):
        for right_size in range(1, 4):
            for left in itertools.combinations(points, left_size):
                for right in itertools.combinations(points, right_size):
                    for tolerance_hours in range(3):
                        tolerance = timedelta(hours=tolerance_hours)
                        assert s.deterministic_one_to_one_pairs(
                            left, right, tolerance
                        ) == _brute_force_matching(left, right, tolerance)


def test_matching_uses_lexicographically_smallest_equal_score_pair() -> None:
    base = _time("2024-01-01T00:00:00Z")
    left = (base, base + 3 * HOUR)
    right = (base + 2 * HOUR, base + 4 * HOUR)
    assert s.deterministic_one_to_one_pairs(left, right, HOUR) == (
        (base + 3 * HOUR, base + 2 * HOUR),
    )


def test_matching_includes_exact_tolerance_and_excludes_one_microsecond_more() -> None:
    base = _time("2024-01-01T00:00:00Z")
    right = base + HOUR
    assert s.deterministic_one_to_one_pairs((base,), (right,), HOUR) == (
        (base, right),
    )
    assert s.deterministic_one_to_one_pairs(
        (base,), (right + timedelta(microseconds=1),), HOUR
    ) == ()


@pytest.mark.parametrize("side", ["left", "right"])
def test_matching_rejects_duplicate_clocks(side: str) -> None:
    base = _time("2024-01-01T00:00:00Z")
    left = (base, base) if side == "left" else (base,)
    right = (base, base) if side == "right" else (base,)
    with pytest.raises(s.SourceSupportTerminalError, match="duplicates"):
        s.deterministic_one_to_one_pairs(left, right, HOUR)


def test_source_timestamp_pins_naive_tokens_and_rejects_nonzero_offsets() -> None:
    expected = _time("2024-01-01T00:00:00Z")
    assert s._timestamp("2024-01-01 00:00:00") == expected
    assert s._timestamp("2024-01-01T00:00:00+00:00") == expected
    with pytest.raises(s.SourceSupportTerminalError, match="not UTC"):
        s._timestamp("2024-01-01T00:00:00+09:00")


def test_small_source_grid_preserves_invalid_hour_without_fill(tmp_path: Path) -> None:
    cfg, calendar = _write_small_sources(tmp_path)
    rows, diagnostics = s.load_joined_sources_for_calendar(
        tmp_path, cfg, calendar
    )
    assert [row.close_time for row in rows] == [
        calendar.start + HOUR,
        calendar.start + 2 * HOUR,
    ]
    assert rows[0].base_valid is True
    assert rows[1].base_valid is False
    assert rows[1].bvol is None
    assert rows[1].dvol is not None
    assert diagnostics["joined_valid_rows_before_full_end"] == 1
    assert diagnostics["fills_imputations_tolerance_or_nearest"] == 0


def test_small_source_grid_excludes_mechanism_cutoff_and_dvol_padding(
    tmp_path: Path,
) -> None:
    cfg, calendar = _write_small_sources(tmp_path)
    rows, diagnostics = s.load_joined_sources_for_calendar(
        tmp_path, cfg, calendar
    )
    assert all(row.close_time < calendar.mechanism_end for row in rows)
    assert diagnostics[
        "dvol_rows_after_required_close_before_source_end_filter"
    ] == 2
    assert diagnostics["dvol"]["rows"] == 4


@pytest.mark.parametrize(
    "source,header",
    [
        (
            "bvol",
            list(
                reversed(
                    s.prereg.SOURCE_ARTIFACTS["binance_btc_bvol_hourly"]["header"]
                )
            ),
        ),
        (
            "dvol",
            s.prereg.SOURCE_ARTIFACTS["deribit_btc_dvol_hourly"]["header"][:-1],
        ),
    ],
)
def test_small_source_loader_rejects_schema_drift(
    tmp_path: Path,
    source: str,
    header: Sequence[str],
) -> None:
    if source == "bvol":
        cfg, calendar = _write_small_sources(
            tmp_path,
            bvol_header=header,
        )
    else:
        cfg, calendar = _write_small_sources(
            tmp_path,
            dvol_header=header,
        )
    with pytest.raises(s.SourceSupportTerminalError, match="header drift"):
        s.load_joined_sources_for_calendar(tmp_path, cfg, calendar)


@pytest.mark.parametrize("mutation", ["short", "gap", "availability", "close"])
def test_small_source_loader_rejects_grid_or_clock_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    calendar = _small_calendar()
    bvol = [
        _bvol_row(calendar.start + index * HOUR)
        for index in range(calendar.bvol_rows)
    ]
    dvol = [
        _dvol_row(calendar.start + index * HOUR)
        for index in range(calendar.dvol_rows)
    ]
    expected = "row count"
    if mutation == "short":
        bvol.pop()
    elif mutation == "gap":
        bvol[1]["date"] = _format(calendar.start + 2 * HOUR)
        expected = "hourly grid"
    elif mutation == "availability":
        bvol[1]["feature_available_time_utc"] = _format(
            calendar.start + 2 * HOUR + timedelta(seconds=1)
        )
        expected = "availability"
    else:
        dvol[1]["close_time"] = _format(
            calendar.start + 2 * HOUR + timedelta(seconds=1)
        )
        expected = "close clock"
    cfg, calendar = _write_small_sources(
        tmp_path, bvol_rows=bvol, dvol_rows=dvol
    )
    with pytest.raises(s.SourceSupportTerminalError, match=expected):
        s.load_joined_sources_for_calendar(tmp_path, cfg, calendar)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"source_rows": "3599", "complete": "true"}, "completeness"),
        ({"source_rows": "3601"}, "completeness"),
        ({"source_rows": "+3600"}, "canonical"),
        ({"reason": "gap"}, "valid row"),
    ],
)
def test_bvol_loader_rejects_invalid_valid_row_contract(
    tmp_path: Path,
    changes: Mapping[str, str],
    match: str,
) -> None:
    calendar = _small_calendar()
    rows = [
        _bvol_row(calendar.start + index * HOUR)
        for index in range(calendar.bvol_rows)
    ]
    rows[1] = _bvol_row(
        calendar.start + HOUR,
        source_rows=changes.get("source_rows", "3600"),
        complete=changes.get("complete", "true"),
        reason=changes.get("reason"),
    )
    cfg, calendar = _write_small_sources(tmp_path, bvol_rows=rows)
    with pytest.raises((s.SourceSupportTerminalError, ValueError), match=match):
        s.load_bvol(tmp_path / cfg.bvol, calendar)


@pytest.mark.parametrize(
    "field,value",
    [
        ("open", "0"),
        ("high", "99"),
        ("low", "111"),
        ("close", "NaN"),
    ],
)
def test_bvol_loader_rejects_invalid_ohlc(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    calendar = _small_calendar()
    rows = [
        _bvol_row(calendar.start + index * HOUR)
        for index in range(calendar.bvol_rows)
    ]
    rows[0][field] = value
    cfg, calendar = _write_small_sources(tmp_path, bvol_rows=rows)
    with pytest.raises(ValueError):
        s.load_bvol(tmp_path / cfg.bvol, calendar)


def test_join_rejects_missing_exact_dvol_clock_without_nearest_fill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, calendar = _write_small_sources(tmp_path)
    real_load_dvol = s.load_dvol

    def missing_exact(
        path: Path,
        selected_calendar: s.SourceCalendar,
    ) -> tuple[dict[datetime, mechanism.Candle], dict[str, Any]]:
        rows, diagnostics = real_load_dvol(path, selected_calendar)
        del rows[selected_calendar.start + HOUR]
        rows[selected_calendar.start + HOUR + timedelta(seconds=1)] = _candle()
        return rows, diagnostics

    monkeypatch.setattr(s, "load_dvol", missing_exact)
    with pytest.raises(s.SourceSupportTerminalError, match="missing DVOL"):
        s.load_joined_sources_for_calendar(tmp_path, cfg, calendar)


def test_preclaim_compressed_capture_is_the_only_bytes_later_decoded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_cfg, calendar = _write_small_sources(tmp_path)
    bvol_raw = (tmp_path / synthetic_cfg.bvol).read_bytes()
    dvol_raw = (tmp_path / synthetic_cfg.dvol).read_bytes()
    bvol_header, bvol_header_hash = s._gzip_header_from_compressed(
        bvol_raw,
        synthetic_cfg.bvol,
    )
    dvol_header, dvol_header_hash = s._gzip_header_from_compressed(
        dvol_raw,
        synthetic_cfg.dvol,
    )
    metadata = {
        "synthetic/bvol-manifest.json": b'{"synthetic":"bvol"}\n',
        "synthetic/dvol-summary.json": b'{"synthetic":"dvol"}\n',
    }
    for relative, raw in metadata.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    specs = {
        "binance_btc_bvol_hourly": {
            "path": synthetic_cfg.bvol,
            "sha256": hashlib.sha256(bvol_raw).hexdigest(),
            "header": bvol_header,
            "header_line_sha256": bvol_header_hash,
        },
        "binance_btc_bvol_manifest": {
            "path": "synthetic/bvol-manifest.json",
            "sha256": hashlib.sha256(
                metadata["synthetic/bvol-manifest.json"]
            ).hexdigest(),
        },
        "deribit_btc_dvol_hourly": {
            "path": synthetic_cfg.dvol,
            "sha256": hashlib.sha256(dvol_raw).hexdigest(),
            "header": dvol_header,
            "header_line_sha256": dvol_header_hash,
        },
        "deribit_btc_dvol_summary": {
            "path": "synthetic/dvol-summary.json",
            "sha256": hashlib.sha256(
                metadata["synthetic/dvol-summary.json"]
            ).hexdigest(),
        },
    }
    monkeypatch.setattr(s.prereg, "SOURCE_ARTIFACTS", specs)
    cfg = replace(
        synthetic_cfg,
        bvol_manifest="synthetic/bvol-manifest.json",
        dvol_summary="synthetic/dvol-summary.json",
    )
    captured = s.validate_source_bindings(tmp_path, cfg)
    (tmp_path / cfg.bvol).write_bytes(b"replacement-bvol")
    (tmp_path / cfg.dvol).write_bytes(b"replacement-dvol")

    bvol, _ = s.load_bvol_compressed(
        captured.bvol_compressed,
        label=cfg.bvol,
        calendar=calendar,
    )
    dvol, _ = s.load_dvol_compressed(
        captured.dvol_compressed,
        label=cfg.dvol,
        calendar=calendar,
    )
    joined, diagnostics = s.join_loaded_sources(bvol, dvol, calendar)
    assert len(joined) == 2
    assert diagnostics["fills_imputations_tolerance_or_nearest"] == 0
    assert captured.bvol_compressed == bvol_raw
    assert captured.dvol_compressed == dvol_raw


def test_canonical_snapshot_decoder_rechecks_retained_hashes() -> None:
    captured = s.CapturedSources(
        bindings={},
        bvol_compressed=b"wrong-bvol",
        dvol_compressed=b"wrong-dvol",
    )
    with pytest.raises(
        s.SourceSupportTerminalError,
        match="retained compressed snapshot drift",
    ):
        s.load_captured_sources(captured)


def test_evaluate_clock_support_passes_exact_frozen_count_boundaries() -> None:
    report = s.evaluate_clock_support(
        _all_support_clocks(),
        {"synthetic": True},
        {"byte_identical": True},
    )
    assert report["support_statistics"]["selection"]["total"] == 45
    assert report["support_statistics"]["future25"]["total"] == 30
    assert report["support_statistics"]["future26"]["total"] == 15
    assert report["support_statistics"]["selection"]["LONG"] == 14
    assert report["support_statistics"]["future25"]["LONG"] == 8
    assert report["support_statistics"]["future26"]["LONG"] == 4
    assert report["support_statistics"]["selection"]["maximum_month_share"][
        "numerator"
    ] == 1
    assert report["support_statistics"]["selection"]["maximum_month_share"][
        "denominator"
    ] == 5
    assert report["passed"] is True


def test_support_gate_accepts_exactly_ninety_day_maximum_gap() -> None:
    primary = list(_support_primary())
    replacements = (
        _time("2023-08-17T00:00:00Z"),
        _time("2023-08-25T00:00:00Z"),
        _time("2023-09-02T00:00:00Z"),
    )
    for index, entry in enumerate(replacements):
        primary[index] = replace(
            primary[index],
            signal_time=entry - timedelta(minutes=5),
            available_at=entry - timedelta(minutes=5),
            entry_time=entry,
            exit_time=entry + mechanism.HOLD_TIME,
        )
    primary.sort(key=lambda row: row.entry_time)
    report = s.evaluate_clock_support(
        _all_support_clocks(primary),
        {"synthetic": True},
        {"byte_identical": True},
    )
    assert report["maximum_accepted_entry_gap_seconds"] == 90 * 86_400
    assert (
        report["checks"]["maximum_accepted_entry_gap_at_most_90_days"]
        is True
    )
    assert report["passed"] is True


def test_support_gate_accepts_exactly_twelve_event_same_side_run() -> None:
    primary = list(_support_primary())
    primary[:12] = [replace(row, side="LONG") for row in primary[:12]]
    primary[12] = replace(primary[12], side="SHORT")
    report = s.evaluate_clock_support(
        _all_support_clocks(primary),
        {"synthetic": True},
        {"byte_identical": True},
    )
    assert report["maximum_same_side_run"] == 12
    assert report["checks"]["maximum_same_side_run_at_most_12"] is True
    assert report["passed"] is True


def test_evaluate_clock_support_fails_one_below_future26_total() -> None:
    primary = list(_support_primary())
    primary.pop()
    report = s.evaluate_clock_support(
        _all_support_clocks(primary),
        {"synthetic": True},
        {"byte_identical": True},
    )
    assert report["support_statistics"]["future26"]["total"] == 14
    assert report["checks"]["future26_total_min_15"] is False
    assert report["passed"] is False


def test_structural_jaccard_rejects_exact_nine_tenths() -> None:
    start = _time("2024-01-01T00:00:00Z")
    primary = tuple(
        _event(start + index * timedelta(days=4), serial=index)
        for index in range(10)
    )
    control = tuple(
        replace(
            row,
            signal_id=f"{row.signal_id}|control",
            control=mechanism.DERIBIT_LED,
        )
        for row in primary[:9]
    )
    result = s.structural_control_novelty(primary, control)
    assert result["exact_entry_jaccard"]["numerator"] == 9
    assert result["exact_entry_jaccard"]["denominator"] == 10
    assert (
        result["checks"]["exact_entry_jaccard_strictly_below_9_over_10"]
        is False
    )


def test_structural_matching_rejects_exact_nineteen_twentieths() -> None:
    start = _time("2024-01-01T00:00:00Z")
    primary = tuple(
        _event(start + index * timedelta(days=5), serial=index)
        for index in range(20)
    )
    shifted: list[mechanism.ScheduledEvent] = []
    for index, row in enumerate(primary):
        delay = HOUR if index < 19 else timedelta(days=2)
        shifted.append(
            replace(
                row,
                signal_id=f"{row.signal_id}|control",
                control=mechanism.DERIBIT_LED,
                signal_time=row.signal_time + delay,
                available_at=row.available_at + delay,
                entry_time=row.entry_time + delay,
                exit_time=row.exit_time + delay,
            )
        )
    result = s.structural_control_novelty(primary, shifted)
    share = result["one_to_one_24h"]["maximum_matched_share"]
    assert share["numerator"] == 19
    assert share["denominator"] == 20
    assert (
        result["checks"][
            "one_to_one_24h_max_matched_share_strictly_below_19_over_20"
        ]
        is False
    )


def test_support_gate_rejects_thirteen_event_same_side_run() -> None:
    primary = list(_support_primary())
    primary[:13] = [
        replace(row, side="LONG")
        for row in primary[:13]
    ]
    report = s.evaluate_clock_support(
        _all_support_clocks(primary),
        {"synthetic": True},
        {"byte_identical": True},
    )
    assert report["maximum_same_side_run"] == 13
    assert report["checks"]["maximum_same_side_run_at_most_12"] is False
    assert report["passed"] is False


def test_support_gate_rejects_more_than_ninety_day_entry_gap() -> None:
    primary = list(_support_primary())
    for index, entry in enumerate(
        (
            _time("2023-06-21T00:00:00Z"),
            _time("2023-06-29T00:00:00Z"),
            _time("2023-07-07T00:00:00Z"),
        )
    ):
        duration = primary[index].exit_time - primary[index].entry_time
        primary[index] = replace(
            primary[index],
            signal_time=entry - timedelta(minutes=5),
            available_at=entry - timedelta(minutes=5),
            entry_time=entry,
            exit_time=entry + duration,
        )
    primary.sort(key=lambda row: row.entry_time)
    report = s.evaluate_clock_support(
        _all_support_clocks(primary),
        {"synthetic": True},
        {"byte_identical": True},
    )
    assert report["maximum_accepted_entry_gap_seconds"] > 90 * 86_400
    assert (
        report["checks"]["maximum_accepted_entry_gap_at_most_90_days"]
        is False
    )
    assert report["passed"] is False


def test_clock_stats_include_exact_boundaries_and_exclude_crossings() -> None:
    start = _time("2024-01-01T00:00:00Z")
    end = start + timedelta(days=10)
    contained = _event(start, serial=1)
    contained = replace(contained, exit_time=end)
    left_crossing = replace(
        _event(start - timedelta(seconds=1), serial=2),
        exit_time=start + timedelta(days=1),
    )
    right_crossing = replace(
        _event(end - timedelta(days=1), serial=3),
        exit_time=end + timedelta(seconds=1),
    )
    stats = s._clock_stats(
        (contained, left_crossing, right_crossing), start, end
    )
    assert stats["total"] == 1
    assert stats["boundary_crossing_excluded"] == 2


@pytest.mark.parametrize(
    "gap,expected",
    [
        (timedelta(days=90), 90 * 86_400),
        (timedelta(days=90, seconds=1), 90 * 86_400 + 1),
    ],
)
def test_maximum_gap_preserves_exact_second_boundary(
    gap: timedelta,
    expected: int,
) -> None:
    start = _time("2024-01-01T00:00:00Z")
    assert s._maximum_gap_seconds(
        (_event(start, serial=1), _event(start + gap, serial=2))
    ) == expected


@pytest.mark.parametrize("count", [12, 13])
def test_maximum_same_side_run_preserves_exact_event_boundary(count: int) -> None:
    start = _time("2024-01-01T00:00:00Z")
    events = tuple(
        _event(start + index * timedelta(days=2), serial=index)
        for index in range(count)
    )
    assert s._maximum_same_side_run(events) == count


def test_append_invariance_is_byte_identical_after_future_rows() -> None:
    start = s.SELECTION_END - timedelta(hours=5)
    prefix = tuple(
        _joined_hour(start + index * HOUR)
        for index in range(5)
    )
    future = (
        _joined_hour(s.SELECTION_END, valid=False),
        _joined_hour(s.SELECTION_END + HOUR),
        _joined_hour(s.SELECTION_END + 3 * HOUR),
    )
    prefix_result = s.append_invariance(prefix)
    appended_result = s.append_invariance((*prefix, *future))
    assert appended_result["byte_identical"] is True
    assert appended_result["future_append_differences"] == 0
    assert appended_result["full_trace_sha256"] == prefix_result["full_trace_sha256"]
    assert appended_result["prefix_trace_sha256"] == prefix_result[
        "prefix_trace_sha256"
    ]


def test_append_invariance_compares_full_computation_not_prefix_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = (
        _joined_hour(s.SELECTION_END - 2 * HOUR),
        _joined_hour(s.SELECTION_END - HOUR),
    )
    future = _joined_hour(s.SELECTION_END)
    controls = (*mechanism.OWN_CLOCKS, *mechanism.PARENT_SET_CONTROLS)

    def future_sensitive(
        rows: Sequence[mechanism.JoinedHour],
    ) -> dict[str, tuple[mechanism.ScheduledEvent, ...]]:
        has_future = any(row.close_time >= s.SELECTION_END for row in rows)
        output: dict[str, tuple[mechanism.ScheduledEvent, ...]] = {}
        for serial, control in enumerate(controls):
            event = _event(
                s.SELECTION_END - timedelta(days=2),
                control=control,
                serial=serial,
            )
            output[control] = (event,) if has_future else ()
        return output

    monkeypatch.setattr(s, "build_all_clocks", future_sensitive)
    result = s.append_invariance((*prefix, future))
    assert result["byte_identical"] is False
    assert result["future_append_differences"] == 1


def test_publish_bundle_is_byte_deterministic_and_canonically_ordered(
    tmp_path: Path,
) -> None:
    controls = (*mechanism.OWN_CLOCKS, *mechanism.PARENT_SET_CONTROLS)
    start = _time("2024-01-01T00:00:00Z")
    forward = {
        control: (
            _event(start, control=control, serial=1),
            _event(start + timedelta(days=2), control=control, serial=2),
        )
        for control in controls
    }
    reverse = {
        control: tuple(reversed(forward[control]))
        for control in reversed(controls)
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = {"passed": True, "support_manifest_hash": "synthetic"}
    s.publish_bundle(first, report, forward)
    s.publish_bundle(second, report, reverse)
    assert _directory_bytes(first) == _directory_bytes(second)

    primary_raw = (first / "primary.csv.gz").read_bytes()
    assert int.from_bytes(primary_raw[4:8], "little") == 0
    assert primary_raw[3] & 0x08 == 0
    lines = gzip.decompress(primary_raw).decode("utf-8").splitlines()
    assert lines[0].split(",") == s.CLOCK_HEADER
    assert lines[1].split(",")[5] < lines[2].split(",")[5]
    payload = json.loads((first / "report.json").read_text(encoding="utf-8"))
    artifact = payload["clock_artifacts"][mechanism.PRIMARY]
    assert artifact["sha256"] == hashlib.sha256(primary_raw).hexdigest()
    assert artifact["bytes"] == len(primary_raw)
    assert artifact["rows"] == 2
    core = {
        key: value for key, value in payload.items()
        if key != "bundle_manifest_hash"
    }
    assert payload["bundle_manifest_hash"] == s.prereg.canonical_hash(core)


def test_publish_bundle_never_overwrites_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    marker = output / "marker"
    marker.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        s.publish_bundle(output, {}, _all_support_clocks())
    assert marker.read_bytes() == b"original"


def test_publish_bundle_cleans_staging_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "bundle"
    writes = 0
    real_write = s._write_file

    def fail_second_write(path: Path, raw: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("synthetic write failure")
        real_write(path, raw)

    monkeypatch.setattr(s, "_write_file", fail_second_write)
    with pytest.raises(OSError, match="synthetic write failure"):
        s.publish_bundle(output, {}, _all_support_clocks())
    assert not output.exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_publish_bundle_no_replace_survives_destination_creation_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "bundle"
    real_rename = s._rename_directory_once

    def create_destination_then_rename(
        source: Path,
        destination: Path,
    ) -> None:
        destination.mkdir()
        real_rename(source, destination)

    monkeypatch.setattr(
        s,
        "_rename_directory_once",
        create_destination_then_rename,
    )
    with pytest.raises(FileExistsError):
        s.publish_bundle(output, {}, _all_support_clocks())
    assert output.is_dir()
    assert not any(output.iterdir())
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_publish_bundle_marks_post_rename_fsync_error_as_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "bundle"
    real_fsync = s._fsync_directory

    def fail_only_after_rename(path: Path) -> None:
        if path == output.parent and output.exists():
            raise OSError("synthetic post-rename fsync failure")
        real_fsync(path)

    monkeypatch.setattr(s, "_fsync_directory", fail_only_after_rename)
    with pytest.raises(
        s.SourceSupportPublicationCommittedError,
        match="parent fsync failed",
    ):
        s.publish_bundle(output, {}, _all_support_clocks())
    assert (output / "report.json").is_file()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_attempt_claim_binds_every_configured_path_and_hash() -> None:
    cfg = s.Config()
    payload = s.attempt_claim_payload(
        cfg=cfg,
        identity=_fake_identity(),
        sources={"synthetic": {"sha256": "f" * 64}},
    )
    assert payload["attempt_claim"] == cfg.attempt_claim
    assert payload["output_directory"] == cfg.output_directory
    assert payload["failure_receipt"] == cfg.failure_receipt
    assert payload["preregistration"]["path"] == cfg.preregistration
    assert payload["source_calendar"]["bvol_expected_rows"] == 26_568
    assert payload["source_calendar"]["dvol_expected_rows"] == 26_569
    assert payload["source_transport"] == {
        "hash_header_and_snapshot_use_same_compressed_bytes": True,
        "compressed_snapshots_retained_before_claim": True,
        "value_rows_decompressed_only_after_claim": True,
        "source_paths_reopened_after_claim": False,
    }
    core = {key: value for key, value in payload.items() if key != "claim_hash"}
    assert payload["claim_hash"] == s.prereg.canonical_hash(core)


@pytest.mark.parametrize(
    "field",
    [
        "bvol",
        "bvol_manifest",
        "dvol",
        "dvol_summary",
        "preregistration",
        "attempt_claim",
        "output_directory",
        "failure_receipt",
    ],
)
def test_authoritative_config_rejects_each_noncanonical_path(field: str) -> None:
    cfg = replace(s.Config(), **{field: f"synthetic/{field}"})
    with pytest.raises(s.SourceSupportTerminalError, match=field):
        s.validate_config(cfg)


def test_evaluator_closure_seals_import_and_pytest_bootstraps() -> None:
    assert "training/__init__.py" in s.EVALUATOR_PATHS
    assert "tests/conftest.py" in s.EVALUATOR_PATHS


def test_preregistered_protocol_seal_rejects_later_committed_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "synthetic/frozen.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("frozen = True\n", encoding="utf-8")
    blob = "a" * 40
    seal = {
        relative: {
            "git_blob": blob,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    }
    registration = {
        "repository": {
            "protocol_seal": seal,
            "protocol_seal_hash": s.prereg.canonical_hash(seal),
        }
    }
    monkeypatch.setattr(s.prereg, "_run_git", lambda repo, *args: blob)
    assert (
        s.validate_preregistered_protocol_seal(tmp_path, registration)
        == registration["repository"]["protocol_seal_hash"]
    )
    path.write_text("frozen = False\n", encoding="utf-8")
    with pytest.raises(
        s.SourceSupportTerminalError,
        match="protocol path drift",
    ):
        s.validate_preregistered_protocol_seal(tmp_path, registration)


def test_authoritative_claim_exists_before_module_loader_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = s.Config()
    _patch_authoritative_preclaim(monkeypatch)
    observed: dict[str, Any] = {}

    def loader(
        captured: s.CapturedSources,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        claim_path = tmp_path / cfg.attempt_claim
        observed.update(json.loads(claim_path.read_text(encoding="utf-8")))
        return (), {"synthetic": True}

    monkeypatch.setattr(s, "load_captured_sources", loader)
    monkeypatch.setattr(
        s,
        "evaluate_support",
        lambda rows, diagnostics: (
            {"passed": True, "support_manifest_hash": "synthetic"},
            _all_support_clocks(),
        ),
    )
    monkeypatch.setattr(
        s,
        "publish_bundle",
        lambda output, report, clocks: {
            **report,
            "bundle_manifest_hash": "synthetic-bundle",
        },
    )
    report = s.run_authoritative(tmp_path, cfg)
    assert observed["status"] == "claimed_before_first_bvol_or_dvol_value_decode"
    assert observed["protected_reads_at_claim"]["bvol_rows_decoded"] == 0
    assert observed["protected_reads_at_claim"]["dvol_rows_decoded"] == 0
    assert observed["attempt_claim"] == cfg.attempt_claim
    assert report["bundle_manifest_hash"] == "synthetic-bundle"


def test_authoritative_preclaim_failure_creates_no_claim_and_never_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def loader(
        captured: s.CapturedSources,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        nonlocal called
        called = True
        return (), {}

    monkeypatch.setattr(
        s,
        "load_registration",
        lambda repo, cfg: (_ for _ in ()).throw(
            s.SourceSupportTerminalError("synthetic preclaim failure")
        ),
    )
    monkeypatch.setattr(s, "load_captured_sources", loader)
    with pytest.raises(s.SourceSupportTerminalError, match="preclaim failure"):
        s.run_authoritative(tmp_path)
    assert called is False
    assert not (tmp_path / s.ATTEMPT_CLAIM).exists()
    assert not (tmp_path / s.FAILURE_RECEIPT).exists()


def test_authoritative_loader_crash_writes_terminal_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = s.Config()
    _patch_authoritative_preclaim(monkeypatch)

    def crash(
        captured: s.CapturedSources,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        raise RuntimeError("synthetic loader crash")

    monkeypatch.setattr(s, "load_captured_sources", crash)
    with pytest.raises(RuntimeError, match="synthetic loader crash"):
        s.run_authoritative(tmp_path, cfg)
    claim = json.loads(
        (tmp_path / cfg.attempt_claim).read_text(encoding="utf-8")
    )
    failure = json.loads(
        (tmp_path / cfg.failure_receipt).read_text(encoding="utf-8")
    )
    assert failure["claim_hash"] == claim["claim_hash"]
    assert failure["error_type"] == "RuntimeError"
    assert failure["retry_resume_fallback_or_repair_allowed"] is False
    assert failure["later_stages_allowed"] is False


def test_authoritative_publication_crash_writes_terminal_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = s.Config()
    _patch_authoritative_preclaim(monkeypatch)
    monkeypatch.setattr(
        s,
        "load_captured_sources",
        lambda captured: ((), {"synthetic": True}),
    )
    monkeypatch.setattr(
        s,
        "evaluate_support",
        lambda rows, diagnostics: (
            {"passed": True, "support_manifest_hash": "synthetic"},
            _all_support_clocks(),
        ),
    )
    monkeypatch.setattr(
        s,
        "publish_bundle",
        lambda output, report, clocks: (_ for _ in ()).throw(
            OSError("synthetic publication crash")
        ),
    )
    with pytest.raises(OSError, match="publication crash"):
        s.run_authoritative(tmp_path, cfg)
    failure = json.loads(
        (tmp_path / cfg.failure_receipt).read_text(encoding="utf-8")
    )
    assert failure["error_type"] == "OSError"
    assert failure["retry_resume_fallback_or_repair_allowed"] is False


def test_authoritative_post_commit_error_does_not_write_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = s.Config()
    _patch_authoritative_preclaim(monkeypatch)
    monkeypatch.setattr(
        s,
        "load_captured_sources",
        lambda captured: ((), {"synthetic": True}),
    )
    monkeypatch.setattr(
        s,
        "evaluate_support",
        lambda rows, diagnostics: (
            {"passed": True, "support_manifest_hash": "synthetic"},
            _all_support_clocks(),
        ),
    )
    monkeypatch.setattr(
        s,
        "publish_bundle",
        lambda output, report, clocks: (_ for _ in ()).throw(
            s.SourceSupportPublicationCommittedError(
                "synthetic committed publication"
            )
        ),
    )
    with pytest.raises(
        s.SourceSupportPublicationCommittedError,
        match="committed publication",
    ):
        s.run_authoritative(tmp_path, cfg)
    assert (tmp_path / cfg.attempt_claim).is_file()
    assert not (tmp_path / cfg.failure_receipt).exists()


@pytest.mark.parametrize(
    "existing",
    ["attempt_claim", "output_directory", "failure_receipt"],
)
def test_authoritative_preexisting_terminal_artifact_forbids_retry_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: str,
) -> None:
    cfg = s.Config()
    path = tmp_path / getattr(cfg, existing)
    if existing == "output_directory":
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic\n", encoding="utf-8")
    called = False

    def loader(
        captured: s.CapturedSources,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        nonlocal called
        called = True
        return (), {}

    monkeypatch.setattr(s, "load_captured_sources", loader)
    with pytest.raises(s.SourceSupportTerminalError, match="no retry"):
        s.run_authoritative(tmp_path, cfg)
    assert called is False


def test_evaluator_ast_has_no_forbidden_data_or_network_dependencies() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "training"
        / "evaluate_cross_venue_volatility_shape_handoff_source_support.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    attributes: set[str] = set()
    training_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
            if node.module == "training":
                training_imports.update(alias.name for alias in node.names)
            elif node.module.startswith("training."):
                pytest.fail(f"unexpected direct training import: {node.module}")
        elif isinstance(node, ast.Attribute):
            attributes.add(node.attr)
    assert training_imports == {
        "cross_venue_volatility_shape_handoff",
        "preregister_cross_venue_volatility_shape_handoff",
    }
    assert imports.isdisjoint(
        {
            "aiohttp",
            "httpx",
            "numpy",
            "pandas",
            "requests",
            "urllib",
            "websockets",
        }
    )
    assert attributes.isdisjoint(
        {
            "PRIOR_VOLATILITY_COMPARATORS",
            "load_gross9_authority",
            "load_market",
            "load_funding",
            "load_outcomes",
        }
    )
    assert "read_csv(" not in source
    assert "read_parquet(" not in source

    prereg_attributes = {
        node.attr
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "prereg"
        )
    }
    assert prereg_attributes == {
        "DEFAULT_OUTPUT",
        "SOURCE_ARTIFACTS",
        "_run_git",
        "atomic_write_once",
        "canonical_hash",
        "repository_identity",
        "sha256_file",
        "validate_registration",
    }
    protected_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not any(value.startswith("data/") for value in protected_literals)
    assert not any(
        token in value
        for value in protected_literals
        for token in (
            "BTCUSDT",
            "OPDR-24",
            "PSR-30/6",
            "PCBR-12",
            "CMSR-36",
            "PRIOR_VOLATILITY_COMPARATORS",
        )
    )

    read_call_sites: dict[str, set[str]] = {
        "open": set(),
        "read_bytes": set(),
        "read_text": set(),
    }
    builtin_open_call_sites: set[str] = set()
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def call_scope(node: ast.AST) -> str:
        current = node
        while current in parents:
            current = parents[current]
            if isinstance(current, ast.FunctionDef):
                return current.name
        return "<module>"

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            builtin_open_call_sites.add(call_scope(node))
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in read_call_sites
        ):
            read_call_sites[node.func.attr].add(call_scope(node))
    assert read_call_sites == {
        "open": {"_fsync_directory", "_write_file"},
        "read_bytes": {
            "load_bvol",
            "load_dvol",
            "validate_source_bindings",
        },
        "read_text": {"load_registration"},
    }
    assert builtin_open_call_sites == set()


@pytest.mark.parametrize("arguments", [["--forbidden"], ["unexpected"]])
def test_cli_rejects_explicit_arguments_before_authoritative_run(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    called = False

    def run(repo: Path) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(s, "run_authoritative", run)
    with pytest.raises(SystemExit, match="takes no arguments"):
        s.main(arguments)
    assert called is False


def test_cli_none_reads_and_rejects_process_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def run(repo: Path) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(s, "run_authoritative", run)
    monkeypatch.setattr(sys, "argv", ["evaluator", "--forbidden"])
    with pytest.raises(SystemExit, match="takes no arguments"):
        s.main(None)
    assert called is False


@pytest.mark.parametrize(
    "passed,expected_code,expected_status",
    [(True, 0, "support_passed"), (False, 2, "support_failed")],
)
def test_cli_returns_support_status_from_synthetic_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    passed: bool,
    expected_code: int,
    expected_status: str,
) -> None:
    monkeypatch.setattr(
        s,
        "run_authoritative",
        lambda repo: {
            "passed": passed,
            "bundle_manifest_hash": "synthetic",
        },
    )
    assert s.main([]) == expected_code
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == expected_status
    assert payload["passed"] is passed
    assert payload["outcomes_opened"] is False
