from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import hashlib
from pathlib import Path

import pytest

from training import cchr_comparator_clock_common as clocks


UTC = timezone.utc


def _candidate(
    candidate_id: str,
    entry: datetime,
    *,
    hold_bars: int = 2,
    side: int = 1,
    origin: datetime | None = None,
) -> clocks.ClockCandidate:
    signal = entry - timedelta(minutes=5)
    return clocks.ClockCandidate(
        candidate_id=candidate_id,
        causal_origins=(origin or signal,),
        signal_time=signal,
        decision_time=entry,
        entry_time=entry,
        exit_time=entry + timedelta(minutes=5 * hold_bars),
        side=side,
    )


def test_timestamp_and_candidate_validation_fail_closed() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        clocks.normalize_utc(datetime(2023, 1, 1), label="synthetic")
    with pytest.raises(ValueError, match="five-minute"):
        clocks.normalize_utc(datetime(2023, 1, 1, 0, 1, tzinfo=UTC), label="synthetic")
    invalid = _candidate("member", datetime(2023, 1, 1, 0, 5, tzinfo=UTC), side=0)
    with pytest.raises(ValueError, match="side"):
        clocks.validate_candidate(invalid)


def test_containment_includes_causal_origins_and_strict_exit_boundary() -> None:
    entry = datetime(2023, 1, 1, 0, 5, tzinfo=UTC)
    contained = _candidate("member", entry)
    assert clocks.candidate_split(contained) == "selection"

    crossed_origin = _candidate(
        "member",
        entry,
        origin=datetime(2022, 12, 31, 23, 55, tzinfo=UTC),
    )
    assert clocks.candidate_split(crossed_origin) is None

    exit_on_end = _candidate(
        "member",
        datetime(2023, 12, 31, 23, 50, tzinfo=UTC),
        hold_bars=2,
    )
    assert clocks.candidate_split(exit_on_end) is None


def test_schedule_is_per_member_and_accepts_touching_half_open_intervals() -> None:
    start = datetime(2023, 2, 1, tzinfo=UTC)
    candidates = [
        _candidate("a", start, hold_bars=2),
        _candidate("a", start + timedelta(minutes=5), hold_bars=2),
        _candidate("a", start + timedelta(minutes=10), hold_bars=2, side=-1),
        _candidate("b", start + timedelta(minutes=5), hold_bars=2),
    ]
    frame = clocks.schedule_candidates(candidates)
    assert list(frame["candidate_id"]) == ["a", "a", "b"]
    assert list(frame.loc[frame["candidate_id"].eq("a"), "entry_time"]) == [
        "2023-02-01T00:00:00Z",
        "2023-02-01T00:10:00Z",
    ]
    assert set(frame.loc[frame["candidate_id"].eq("b"), "entry_time"]) == {
        "2023-02-01T00:05:00Z"
    }


def test_duplicate_raw_member_entry_fails_before_scheduling() -> None:
    entry = datetime(2023, 2, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="duplicate raw"):
        clocks.schedule_candidates([_candidate("a", entry), _candidate("a", entry)])


def test_exact_clock_schema_and_expected_members_are_enforced() -> None:
    frame = clocks.schedule_candidates(
        [_candidate("a", datetime(2023, 2, 1, tzinfo=UTC))]
    )
    normalized = clocks.validate_clock_frame(frame, expected_candidate_ids=("a",))
    assert tuple(normalized.columns) == clocks.CLOCK_COLUMNS
    with pytest.raises(ValueError, match="candidate map"):
        clocks.validate_clock_frame(frame, expected_candidate_ids=("a", "b"))
    with pytest.raises(ValueError, match="six-column"):
        clocks.validate_clock_frame(frame.assign(return_pct=1.0))


def test_deterministic_gzip_clock_has_stable_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(clocks, "REPOSITORY_ROOT", tmp_path)
    frame = clocks.schedule_candidates(
        [_candidate("a", datetime(2023, 2, 1, tzinfo=UTC))]
    )
    first = clocks.write_deterministic_gzip_clock(
        frame, "first.csv.gz", expected_candidate_ids=("a",)
    )
    second = clocks.write_deterministic_gzip_clock(
        frame, "second.csv.gz", expected_candidate_ids=("a",)
    )
    assert first == second
    assert (tmp_path / "first.csv.gz").read_bytes() == (
        tmp_path / "second.csv.gz"
    ).read_bytes()
    payload = gzip.decompress((tmp_path / "first.csv.gz").read_bytes())
    assert hashlib.sha256((tmp_path / "first.csv.gz").read_bytes()).hexdigest() == first
    assert payload.decode().splitlines() == [
        "candidate_id,split,decision_time,entry_time,exit_time,side",
        "a,selection,2023-02-01T00:00:00Z,2023-02-01T00:00:00Z,2023-02-01T00:10:00Z,1",
    ]


def test_hash_bound_reader_materializes_only_allowlisted_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(clocks, "REPOSITORY_ROOT", tmp_path)
    source = tmp_path / "source.csv"
    source.write_text(
        "date,causal_value,future_return\n2023-01-01T00:00:00Z,1.5,99.0\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = clocks.read_hash_bound_columns(
        "source.csv",
        expected_sha256=digest,
        columns=("date", "causal_value"),
        parse_dates=("date",),
    )
    assert list(frame.columns) == ["date", "causal_value"]
    assert "future_return" not in frame
    with pytest.raises(ValueError, match="hash differs"):
        clocks.read_hash_bound_columns(
            "source.csv",
            expected_sha256="0" * 64,
            columns=("date", "causal_value"),
        )


def test_candidate_map_hash_is_order_invariant_and_rejects_empty_keys() -> None:
    first = {"b": {"hold": 2}, "a": {"hold": 1}}
    second = {"a": {"hold": 1}, "b": {"hold": 2}}
    assert clocks.candidate_map_hash(first) == clocks.candidate_map_hash(second)
    with pytest.raises(ValueError, match="byte-exact"):
        clocks.candidate_map_hash({"": {}})
