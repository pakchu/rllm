from __future__ import annotations

import gzip
import hashlib
import io
from datetime import date

import numpy as np
import pandas as pd
import pytest

from training import build_cboe_option_pressure_rank_rotation_support as s
from training import preregister_cboe_option_pressure_rank_rotation as p


def state_frame(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    records = []
    for raw_date, term, tail, option in rows:
        record = {
            "observation_date": pd.Timestamp(raw_date, tz="UTC"),
            "term_pressure": term,
            "tail_pressure": tail,
            "option_pressure": option,
            "term_tail_order": s._term_tail_order(term, tail),
        }
        for sponsor in s.SURFACES:
            record[f"{sponsor}_position"] = s.ordinal_position(
                term, tail, option, sponsor
            )
        records.append(record)
    frame = pd.DataFrame(records)
    for column in (f"{surface}_position" for surface in s.SURFACES):
        frame[column] = pd.array(frame[column], dtype="Int64")
    return frame


def clock_row(
    entry: str,
    exit_time: str,
    side: str = "LONG",
) -> dict[str, object]:
    return {
        "entry_time": pd.Timestamp(entry),
        "exit_time": pd.Timestamp(exit_time),
        "side": side,
    }


def test_preregistration_and_implementation_bindings_are_frozen() -> None:
    payload = s.validate_preregistration()
    assert payload["policy"]["policy_id"] == "OPRR-288"
    assert s.sha256_file(s.PREREGISTRATION) == s.PREREGISTRATION_SHA256
    assert s.sha256_file(s.IMPLEMENTATION_CONTRACT) == (
        s.IMPLEMENTATION_CONTRACT_SHA256
    )
    assert payload["source_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
    assert payload["outcomes_opened"] is False


def test_strict_prior_rank_never_uses_current_value() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    ranks = s.strict_prior_midranks(values, lookback=3, minimum=2)
    assert np.isnan(ranks[0])
    assert np.isnan(ranks[1])
    assert ranks[2] == 1.0
    assert ranks[3] == 1.0
    tied = s.strict_prior_midranks([1.0, 1.0, 1.0], lookback=2, minimum=2)
    assert tied[2] == 0.5


def test_source_validation_rejects_duplicate_nonpositive_and_future_rows() -> None:
    allowlist = ("observation_date", "x")
    valid = pd.DataFrame(
        {"observation_date": ["2023-01-03", "2023-01-04"], "x": ["1", "2"]}
    )
    result = s.validate_source_frame(
        valid, allowlist=allowlist, source_name="synthetic"
    )
    assert result["x"].tolist() == [1.0, 2.0]

    duplicate = pd.DataFrame(
        {"observation_date": ["2023-01-03", "2023-01-03"], "x": ["1", "2"]}
    )
    with pytest.raises(RuntimeError, match="duplicated"):
        s.validate_source_frame(
            duplicate, allowlist=allowlist, source_name="synthetic"
        )
    nonpositive = pd.DataFrame(
        {"observation_date": ["2023-01-03"], "x": ["0"]}
    )
    with pytest.raises(RuntimeError, match="primitive invalid"):
        s.validate_source_frame(
            nonpositive, allowlist=allowlist, source_name="synthetic"
        )
    future = pd.DataFrame(
        {"observation_date": ["2024-01-02"], "x": ["1"]}
    )
    with pytest.raises(RuntimeError, match="2024-or-later"):
        s.validate_source_frame(
            future, allowlist=allowlist, source_name="synthetic"
        )


def test_ordinal_transition_uses_immediately_previous_state_and_never_skips_tie() -> None:
    assert s.ordinal_position(0.90, 0.50, 0.50, "term") == 2
    assert s.ordinal_position(0.50, 0.10, 0.50, "tail") == 0
    assert s.ordinal_position(0.50, 0.50, 0.90, "option") is None
    states = state_frame(
        [
            ("2022-01-03", 0.20, 0.60, 0.10),
            ("2022-01-04", 0.30, 0.70, 0.50),
            ("2022-01-05", 0.40, 0.80, 0.90),
            ("2022-01-06", 0.50, 0.50, 0.40),
            ("2022-01-07", 0.60, 0.80, 0.90),
        ]
    )
    transitions = s.build_transitions(states)
    assert transitions.loc[0, "option_rotation"] == 1
    assert transitions.loc[1, "option_rotation"] == 1
    assert pd.isna(transitions.loc[2, "option_rotation"])
    assert pd.isna(transitions.loc[3, "option_rotation"])
    assert transitions.loc[3, "source_date"] == pd.Timestamp(
        "2022-01-07T00:00:00Z"
    )


def test_prospective_session_calendar_ignores_future_source_membership() -> None:
    assert s._next_session_date(pd.Timestamp("2023-04-06", tz="UTC")) == date(
        2023, 4, 10
    )
    assert s._next_session_date(pd.Timestamp("2023-06-16", tz="UTC")) == date(
        2023, 6, 20
    )
    assert s._next_session_date(pd.Timestamp("2023-12-22", tz="UTC")) == date(
        2023, 12, 26
    )
    assert s._next_session_date(pd.Timestamp("2023-12-29", tz="UTC")) == date(
        2024, 1, 2
    )
    assert len(s.CLOSURES) == 47
    assert s._ny_time(date(2023, 7, 5), 9, 35) == pd.Timestamp(
        "2023-07-05T13:35:00Z"
    )
    assert s._ny_time(date(2023, 1, 3), 9, 35) == pd.Timestamp(
        "2023-01-03T14:35:00Z"
    )


def test_primary_requires_rotation_and_all_three_directional_confirmations() -> None:
    states = state_frame(
        [
            ("2022-01-03", 0.20, 0.60, 0.10),
            ("2022-01-04", 0.30, 0.70, 0.50),
            ("2022-01-05", 0.40, 0.80, 0.90),
            ("2022-01-06", 0.30, 0.70, 0.50),
            ("2022-01-07", 0.20, 0.60, 0.10),
            # Option rotates upward but term falls, so primary rejects it.
            ("2022-01-10", 0.19, 0.61, 0.40),
        ]
    )
    transitions = s.build_transitions(states)
    primary = s.raw_candidates(transitions, "primary")
    assert primary["side"].tolist() == ["SHORT", "SHORT", "LONG", "LONG"]
    assert primary["rotation_direction"].tolist() == ["UP", "UP", "DOWN", "DOWN"]
    assert primary["option_own_change_agreement"].eq("AGREE").all()
    assert primary["term_confirmation"].eq("AGREE").all()
    assert primary["tail_confirmation"].eq("AGREE").all()
    option_only = s.raw_candidates(transitions, "option_own_confirmed")
    assert len(option_only) == 5
    assert len(primary) == 4


def test_controls_random_side_and_clock_schema_are_deterministic() -> None:
    states = state_frame(
        [
            ("2022-01-03", 0.20, 0.60, 0.10),
            ("2022-01-04", 0.30, 0.70, 0.50),
            ("2022-01-05", 0.40, 0.80, 0.90),
            ("2022-01-06", 0.30, 0.70, 0.50),
            ("2022-01-07", 0.20, 0.60, 0.10),
        ]
    )
    controls, raw = s.build_controls(s.build_transitions(states))
    assert list(controls) == list(p.CONTROL_ORDER)
    assert list(raw) == list(p.CONTROL_ORDER)
    primary = controls["primary"]
    random_control = controls["deterministic_random_side"]
    assert random_control["entry_time"].tolist() == primary["entry_time"].tolist()
    expected = []
    for entry in primary["entry_time"]:
        message = b"OPRR-288|" + s._format_time(entry).encode("ascii")
        expected.append(
            "LONG" if hashlib.sha256(message).digest()[0] < 128 else "SHORT"
        )
    assert random_control["side"].tolist() == expected
    first = s.deterministic_clock_bytes(controls)
    second = s.deterministic_clock_bytes(controls)
    assert first == second
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as handle:
        header = handle.readline().decode("utf-8").strip().split(",")
    assert header == list(s.CLOCK_COLUMNS)
    assert not any(
        token in column.lower()
        for column in header
        for token in s.FORBIDDEN_CLOCK_TOKENS
    )


def test_global_reservation_accepts_equality_and_suppresses_overlap() -> None:
    base = {
        column: "X"
        for column in s.CLOCK_COLUMNS
        if column not in (
            "signal_id", "source_date", "signal_available_time",
            "entry_time", "exit_time", "side",
        )
    }
    rows = []
    for index, (entry, exit_time) in enumerate(
        [
            ("2022-01-03T10:00:00Z", "2022-01-04T10:00:00Z"),
            ("2022-01-03T20:00:00Z", "2022-01-04T20:00:00Z"),
            ("2022-01-04T10:00:00Z", "2022-01-05T10:00:00Z"),
        ]
    ):
        rows.append(
            {
                "control": "primary",
                "signal_id": f"{index:064d}",
                "source_date": pd.Timestamp("2022-01-01", tz="UTC"),
                "signal_available_time": pd.Timestamp(entry) - s.BAR,
                "entry_time": pd.Timestamp(entry),
                "exit_time": pd.Timestamp(exit_time),
                "side": "LONG",
                **base,
            }
        )
    reserved = s.reserve_nonoverlap(pd.DataFrame(rows, columns=s.CLOCK_COLUMNS))
    assert reserved["entry_time"].tolist() == [
        pd.Timestamp("2022-01-03T10:00:00Z"),
        pd.Timestamp("2022-01-04T10:00:00Z"),
    ]


def test_raw_retention_is_computed_before_reservation() -> None:
    def raw(entries: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "source_date": pd.to_datetime(entries, utc=True),
                "entry_time": pd.to_datetime(entries, utc=True) + pd.Timedelta(hours=14),
            }
        )
        frame["exit_time"] = frame["entry_time"] + s.DAY
        frame["signal_id"] = [f"{i:064d}" for i in range(len(frame))]
        return frame

    primary = raw(["2021-01-04", "2021-01-06"])
    control = raw(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"])
    assert s._raw_retention(primary, control, "train") == 0.5


def test_clock_stats_use_local_calendar_days_not_elapsed_dst_hours() -> None:
    rows = pd.DataFrame(
        [
            clock_row("2021-03-12T14:35:00Z", "2021-03-13T14:35:00Z"),
            clock_row("2021-03-15T13:35:00Z", "2021-03-16T13:35:00Z", "SHORT"),
        ]
    )
    rows["signal_id"] = ["a", "b"]
    stats = s.clock_stats(rows)
    assert stats["maximum_gap_days"] == 3.0
    assert stats["long"] == 1
    assert stats["short"] == 1


def test_tolerant_matching_uses_order_preserving_local_date_dp() -> None:
    left = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2023-01-03T14:35:00Z", "2023-01-05T14:35:00Z"], utc=True
            )
        }
    )
    right = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2023-01-04T14:35:00Z", "2023-01-06T14:35:00Z"], utc=True
            )
        }
    )
    assert s.maximum_tolerant_matches(left["entry_time"], right["entry_time"]) == 2
    assert s.tolerant_entry_jaccard(left, right) == 1.0
    assert s.exact_entry_jaccard(left, right) == 0.0


def test_signed_occupancy_is_half_open_clipped_and_rejects_overlap() -> None:
    start = pd.Timestamp("2023-01-01T00:00:00Z")
    end = pd.Timestamp("2023-01-01T00:20:00Z")
    rows = pd.DataFrame(
        [
            clock_row("2022-12-31T23:55:00Z", "2023-01-01T00:05:00Z"),
            clock_row("2023-01-01T00:05:00Z", "2023-01-01T00:15:00Z", "SHORT"),
        ]
    )
    occupancy = s._signed_occupancy(rows, start, end)
    assert occupancy.tolist() == [1, -1, -1, 0]
    overlapping = pd.DataFrame(
        [
            clock_row("2023-01-01T00:00:00Z", "2023-01-01T00:10:00Z"),
            clock_row("2023-01-01T00:05:00Z", "2023-01-01T00:15:00Z"),
        ]
    )
    with pytest.raises(RuntimeError, match="overlaps"):
        s._signed_occupancy(overlapping, start, end)


def test_failed_source_support_never_opens_comparator_or_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = state_frame(
        [
            ("2022-01-03", 0.20, 0.60, 0.10),
            ("2022-01-04", 0.30, 0.70, 0.50),
        ]
    )
    monkeypatch.setattr(
        s,
        "evaluate_novelty",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("comparator must remain sealed")
        ),
    )
    report, clock_bytes = s.build_support_from_states(states)
    assert clock_bytes
    assert report["source_support_passed"] is False
    assert report["comparator_rows_decoded"] == 0
    assert report["outcomes_opened"] is False
    assert report["funding_loaded"] is False
    assert report["outcome_boundary"]["BTC_market_rows_decoded"] == 0
    assert report["outcome_boundary"]["PnL_CAGR_MDD_values_decoded"] == 0
    assert report["decision"] == (
        "retire_OPRR_288_unchanged_before_comparators_and_outcomes"
    )


def test_comparator_entry_points_require_guarded_authorization() -> None:
    with pytest.raises(RuntimeError, match="requires committed source"):
        s._authorize_comparator_open(
            source_support_passed=False,
            composition_passed=True,
            artifact_eligible=True,
        )
    with pytest.raises(RuntimeError, match="authorization missing"):
        s.evaluate_novelty(
            pd.DataFrame(),
            p.build_manifest(),
            authorization=object(),
        )


def test_write_once_rejects_drift(tmp_path) -> None:
    output = tmp_path / "artifact.bin"
    assert s._write_once(output, b"alpha") == "created"
    assert s._write_once(output, b"alpha") == "verified_existing"
    with pytest.raises(RuntimeError, match="noncanonical"):
        s._write_once(output, b"beta")
