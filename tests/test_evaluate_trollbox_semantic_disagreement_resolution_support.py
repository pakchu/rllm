from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from training.evaluate_trollbox_semantic_disagreement_resolution_support import (
    BAR,
    DEADLINE,
    HOLD,
    Candidate,
    SemanticEvent,
    build_primary_candidates,
    build_report,
    canonical_hash,
    load_json,
    schedule_candidates,
    write_report,
)


BASE = datetime(2020, 7, 1, tzinfo=timezone.utc)


def _event(
    minutes: int,
    label: str,
    *,
    bullish: int,
    bearish: int,
    unclear: int = 0,
) -> SemanticEvent:
    end = BASE + timedelta(minutes=minutes)
    entry = end + BAR
    return SemanticEvent(
        observation_start=end - BAR,
        observation_end=end,
        entry_earliest=entry,
        legacy_exit_time=entry + timedelta(hours=2),
        crowd_label=label,
        bullish_participants=bullish,
        bearish_participants=bearish,
        unclear_participants=unclear,
        selected_participants=bullish + bearish + unclear,
        selected_messages=bullish + bearish + unclear,
        meta_instruction_guarded_messages=0,
    )


def _candidate(minutes: int, *, side: int = 1) -> Candidate:
    onset = BASE + timedelta(minutes=minutes)
    resolution = onset + timedelta(hours=1)
    entry = resolution + BAR
    return Candidate(
        onset_end=onset,
        resolution_end=resolution,
        entry=entry,
        exit=entry + HOLD,
        side=side,
        onset_bullish=2,
        onset_bearish=2,
    )


def test_primary_uses_first_clear_event_at_inclusive_deadline() -> None:
    events = [
        _event(0, "UNCLEAR", bullish=2, bearish=2),
        _event(60, "UNCLEAR", bullish=3, bearish=2),
        _event(360, "BULLISH", bullish=4, bearish=1),
        _event(365, "BEARISH", bullish=1, bearish=4),
    ]

    candidates, audit = build_primary_candidates(events)

    assert len(candidates) == 1
    assert candidates[0].onset_end == BASE
    assert candidates[0].resolution_end == BASE + DEADLINE
    assert candidates[0].side == 1
    assert audit.resolved_episodes == 1
    assert audit.expired_episodes == 0


def test_expiry_happens_before_current_event_can_rearm() -> None:
    events = [
        _event(0, "UNCLEAR", bullish=2, bearish=2),
        _event(420, "UNCLEAR", bullish=2, bearish=2),
        _event(480, "BEARISH", bullish=1, bearish=3),
    ]

    candidates, audit = build_primary_candidates(events)

    assert len(candidates) == 1
    assert candidates[0].onset_end == BASE + timedelta(minutes=420)
    assert candidates[0].side == -1
    assert audit.resolved_episodes == 1
    assert audit.expired_episodes == 1


def test_scheduler_drops_overlap_without_queueing() -> None:
    first = _candidate(0)
    overlapping = _candidate(120, side=-1)
    later = _candidate(480, side=-1)

    accepted, audit = schedule_candidates([later, overlapping, first])

    assert [row.entry for row in accepted] == [first.entry, later.entry]
    assert audit.raw_candidates == 3
    assert audit.split_boundary_drops == 0
    assert audit.overlap_suppressions == 1
    assert audit.accepted_candidates == 2


def test_scheduler_drops_cross_split_hold() -> None:
    entry = datetime(2021, 12, 31, 22, tzinfo=timezone.utc)
    candidate = Candidate(
        onset_end=entry - timedelta(hours=2),
        resolution_end=entry - BAR,
        entry=entry,
        exit=entry + HOLD,
        side=1,
        onset_bullish=2,
        onset_bearish=2,
    )

    accepted, audit = schedule_candidates([candidate])

    assert accepted == []
    assert audit.split_boundary_drops == 1


def test_real_frozen_source_passes_without_market_access() -> None:
    report = build_report()

    assert report["support_gate"]["passed"] is True
    assert report["primary"]["build_audit"] == {
        "raw_candidates": 255,
        "resolved_episodes": 255,
        "expired_episodes": 153,
        "unresolved_end_of_source": 0,
    }
    assert report["primary"]["schedule_audit"] == {
        "raw_candidates": 255,
        "split_contained_candidates": 255,
        "split_boundary_drops": 0,
        "overlap_suppressions": 17,
        "accepted_candidates": 238,
    }
    assert report["primary"]["splits"]["train"]["accepted_events"] == 163
    assert report["primary"]["splits"]["selection"]["accepted_events"] == 75
    assert report["primary"]["splits"]["train"]["side_counts"] == {
        "LONG": 70,
        "SHORT": 93,
    }
    assert report["primary"]["splits"]["selection"]["side_counts"] == {
        "LONG": 35,
        "SHORT": 40,
    }
    assert report["outcome_boundary"] == {
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "outcome_rows_loaded": 0,
        "return_or_pnl_fields_read": 0,
        "raw_private_text_opened": False,
        "raw_private_text_committed": False,
        "post_2022_semantic_rows_loaded": 0,
        "network_calls": 0,
        "outcomes_opened": False,
    }
    core = {k: v for k, v in report.items() if k not in {"created_at", "result_hash"}}
    assert report["result_hash"] == canonical_hash(core)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_json(path)


def test_report_publication_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "support.json"
    report = {"passed": True}

    write_report(output, report)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        write_report(output, report)
