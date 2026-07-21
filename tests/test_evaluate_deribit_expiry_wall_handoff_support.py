from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import training.evaluate_deribit_expiry_wall_handoff_support as dewh
from training.evaluate_deribit_expiry_wall_handoff_support import (
    BAR,
    HOLD_BARS,
    Candidate,
    _dehr_release_side_control,
    _load_comparators,
    normalized_entry,
    publish,
    schedule_candidates,
    strict_prior_calendar_midrank,
    support_gate,
)


def test_strict_prior_calendar_midrank_excludes_current_and_averages_ties() -> None:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(days=index) for index in range(4)]

    ranks = strict_prior_calendar_midrank(
        timestamps,
        [1.0, 2.0, 2.0, 3.0],
        minimum=2,
    )

    assert np.isnan(ranks[0])
    assert np.isnan(ranks[1])
    assert ranks[2] == pytest.approx(0.75)
    assert ranks[3] == pytest.approx(1.0)


def test_normalized_entry_never_backdates_source_observation() -> None:
    exact = datetime(2022, 1, 1, 9, 5, tzinfo=timezone.utc)
    offset = datetime(2022, 1, 1, 9, 5, 0, 125_000, tzinfo=timezone.utc)

    assert normalized_entry(exact) == datetime(2022, 1, 1, 9, 10, tzinfo=timezone.utc)
    assert normalized_entry(offset) == datetime(2022, 1, 1, 9, 15, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="must be UTC"):
        normalized_entry(datetime(2022, 1, 1, 9, 5))


def _candidate(entry: datetime, *, side: int = 1) -> Candidate:
    decision = entry - timedelta(minutes=5)
    return Candidate(
        causal_origin=decision - timedelta(hours=1),
        delivery_time=decision - timedelta(hours=1),
        decision_time=decision,
        availability_time=decision,
        entry_time=entry,
        exit_time=entry + HOLD_BARS * BAR,
        side=side,
    )


def test_schedule_contains_splits_and_suppresses_overlap() -> None:
    first = _candidate(datetime(2020, 7, 1, 9, 10, tzinfo=timezone.utc))
    overlap = _candidate(datetime(2020, 7, 1, 10, 10, tzinfo=timezone.utc))
    later = _candidate(datetime(2020, 7, 2, 9, 10, tzinfo=timezone.utc), side=-1)
    before_split = _candidate(datetime(2020, 6, 30, 9, 10, tzinfo=timezone.utc))

    accepted, audit = schedule_candidates([later, overlap, first, before_split])

    assert [row.entry_time for row in accepted] == [first.entry_time, later.entry_time]
    assert [row.split for row in accepted] == ["train", "train"]
    assert audit.raw_candidates == 4
    assert audit.split_contained_candidates == 3
    assert audit.split_boundary_drops == 1
    assert audit.overlap_suppressions == 1
    assert audit.accepted_candidates == 2


def _summary(
    events: int,
    *,
    longs: int,
    shorts: int,
    years: dict[str, int],
    halves: dict[str, int],
    quarters: dict[str, int],
    active_months: int,
    month_share: float,
    weekday_share: float,
) -> dict[str, object]:
    return {
        "accepted_events": events,
        "side_counts": {"LONG": longs, "SHORT": shorts},
        "year_counts": years,
        "half_counts": halves,
        "quarter_counts": quarters,
        "active_months": active_months,
        "maximum_calendar_month_share": month_share,
        "maximum_utc_entry_weekday_share": weekday_share,
    }


def test_support_gate_is_conjunctive() -> None:
    train = _summary(
        80,
        longs=40,
        shorts=40,
        years={"2020": 10, "2021": 35, "2022": 35},
        halves={
            "2020-H2": 10,
            "2021-H1": 17,
            "2021-H2": 18,
            "2022-H1": 17,
            "2022-H2": 18,
        },
        quarters={},
        active_months=26,
        month_share=0.10,
        weekday_share=0.25,
    )
    selection = _summary(
        24,
        longs=12,
        shorts=12,
        years={"2023": 24},
        halves={"2023-H1": 12, "2023-H2": 12},
        quarters={
            "2023-Q1": 6,
            "2023-Q2": 6,
            "2023-Q3": 6,
            "2023-Q4": 6,
        },
        active_months=10,
        month_share=0.20,
        weekday_share=0.30,
    )
    primary = {"splits": {"train": train, "selection": selection}}

    assert support_gate(primary)["passed"] is True
    cast_selection = primary["splits"]["selection"]
    assert isinstance(cast_selection, dict)
    cast_selection["accepted_events"] = 19
    result = support_gate(primary)
    assert result["passed"] is False
    assert result["checks"]["selection_total_between_20_and_100"] is False


def test_publish_is_create_only(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    clock_path = tmp_path / "clock.csv.gz"
    report = {"pure_clock": None, "value": 1}

    publish(report_path, clock_path, report, None)
    assert report_path.exists()
    assert not clock_path.exists()
    with pytest.raises(FileExistsError):
        publish(report_path, clock_path, report, None)


def test_dehr_release_side_control_preserves_exact_train_clocks() -> None:
    train = _candidate(datetime(2022, 1, 1, 9, 10, tzinfo=timezone.utc))
    train = replace(train, split="train")
    selection = _candidate(datetime(2023, 1, 1, 9, 10, tzinfo=timezone.utc))
    selection = replace(selection, split="selection")

    controlled, audit = _dehr_release_side_control(
        [train, selection], {train.causal_origin: -1}
    )

    assert len(controlled) == 1
    assert controlled[0].entry_time == train.entry_time
    assert controlled[0].side == -1
    assert audit == {
        "eligible_dewh_train_clocks": 1,
        "matched_dehr_release_sides": 1,
        "missing_train_clocks": 0,
        "selection_control_unavailable_by_frozen_design": True,
    }


def test_comparator_loader_revalidates_frozen_clock_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    drifted = tmp_path / "afcs.csv"
    drifted.write_text("drifted\n", encoding="utf-8")
    monkeypatch.setattr(dewh, "AFCS_CLOCK", drifted)

    with pytest.raises(ValueError, match="frozen comparator binding changed"):
        _load_comparators()
