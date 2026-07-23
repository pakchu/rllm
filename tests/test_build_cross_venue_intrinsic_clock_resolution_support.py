from __future__ import annotations

import gzip
import hashlib
import io
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_cross_venue_intrinsic_clock_resolution_support as s


def _feature_row(
    source_day: str = "2022-01-01",
    *,
    early: int = 72,
    late: int = 144,
    leader: str = "spot",
    side_sign: int = 1,
) -> dict[str, object]:
    return {
        "source_day": pd.Timestamp(source_day, tz="UTC"),
        "spot_anchor_index": early if leader == "spot" else late,
        "um_anchor_index": late if leader == "spot" else early,
        "early_index": early,
        "late_index": late,
        "leader": leader,
        "side_sign": side_sign,
        "gap_bars": late - early,
        "gap_reference_count": 180,
        "gap_threshold": 12.0,
        "gap_pass": True,
        "initial_conflict": True,
        "late_alignment": True,
        "leader_persistence": True,
        "laggard_resolution": True,
        "primary": True,
        "gap_only": True,
        "initial_conflict_only": True,
        "late_alignment_only": True,
        "no_leader_persistence": True,
        "no_gap_tail": True,
        "fixed_valid": True,
        "fixed_early_index": early - 12,
        "fixed_late_index": late - 12,
        "fixed_leader": leader,
        "fixed_side_sign": side_sign,
        "stale_valid": True,
    }


def _features(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=s.FEATURE_COLUMNS)


def _candidate(
    entry: str,
    *,
    control: str = "primary",
    side: int = 1,
    leader: str = "spot",
    hold_bars: int = 72,
) -> dict[str, object]:
    entry_time = pd.Timestamp(entry)
    decision = entry_time
    resolution = decision - 2 * s.BAR
    source_day = resolution.floor("D")
    early = resolution - 12 * s.BAR
    row = s._candidate_row(
        control,
        source_day,
        int((early - source_day) / s.BAR),
        int((resolution - source_day) / s.BAR),
        side,
        leader,
    )
    row["exit_time"] = entry_time + hold_bars * s.BAR
    return row


def _synthetic_source(days: int = 125) -> pd.DataFrame:
    dates = pd.date_range(
        "2020-01-01",
        periods=days * s.ROWS_PER_DAY,
        freq=s.BAR,
        tz="UTC",
    )
    spot_day = np.concatenate(
        [
            np.full(96, 2.0),
            np.full(s.ROWS_PER_DAY - 96, 0.5),
        ]
    )
    um_day = np.ones(s.ROWS_PER_DAY)
    spot_signed_day = 0.5 * spot_day
    um_signed_day = np.concatenate(
        [
            np.full(72, -0.5),
            np.full(s.ROWS_PER_DAY - 72, 1.0),
        ]
    )
    return pd.DataFrame(
        {
            "date": dates,
            "spot_quote_notional": np.tile(spot_day, days),
            "um_quote_notional": np.tile(um_day, days),
            "spot_signed_quote_notional": np.tile(spot_signed_day, days),
            "um_signed_quote_notional": np.tile(um_signed_day, days),
            "_row_valid": True,
        }
    )


def test_loader_passes_exact_allowlist_to_read_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    source = tmp_path / "source.csv.gz"

    def fake_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame(columns=s.prereg.SOURCE_ALLOWLIST)

    monkeypatch.setattr(s.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(s, "validate_source_frame", lambda frame: frame)
    monkeypatch.setattr(s.prereg, "SOURCE", str(source))
    result = s.load_source(str(source))

    assert list(captured["usecols"]) == list(s.prereg.SOURCE_ALLOWLIST)
    assert list(result.columns) == list(s.prereg.SOURCE_ALLOWLIST)


def test_source_validation_marks_invalid_rows_without_loading_future_fields() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq=s.BAR, tz="UTC")
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature_available_time_utc": dates + s.BAR,
            "trade_earliest_time_utc": dates + s.BAR,
            "spot_quote_notional": [1.0, 1.0, 1.0],
            "um_quote_notional": [1.0, 1.0, 1.0],
            "spot_signed_quote_notional": [0.5, 2.0, 0.5],
            "um_signed_quote_notional": [-0.5, -0.5, -0.5],
            "source_complete": ["true", "true", "false"],
        },
        columns=s.prereg.SOURCE_ALLOWLIST,
    )
    result = s.validate_source_frame(frame, exact_grid=False)
    assert result["_row_valid"].tolist() == [True, False, False]
    assert not any(
        token in column
        for column in result.columns
        for token in ("close", "return", "basis", "pnl")
    )


def test_exact_prior_windows_gap_reference_and_prefix_only_causality() -> None:
    frame = _synthetic_source()
    final_day = len(frame) // s.ROWS_PER_DAY - 1
    # Invalid after the entry/decision prefix must not erase the final event.
    frame.loc[
        final_day * s.ROWS_PER_DAY + 250,
        "_row_valid",
    ] = False
    features, funnel = s.build_daily_features(frame)
    final = features.loc[
        features["source_day"].eq(pd.Timestamp("2020-05-04", tz="UTC"))
    ].iloc[0]

    assert final["gap_reference_count"] == 103
    assert final["gap_threshold"] == pytest.approx(72.0)
    assert bool(final["gap_pass"])
    assert bool(final["initial_conflict"])
    assert bool(final["late_alignment"])
    assert bool(final["primary"])
    assert bool(final["fixed_valid"])
    assert bool(final["stale_valid"])
    assert funnel["raw_primary"] > 0


def test_current_gap_is_excluded_from_strictly_prior_q60() -> None:
    frame = _synthetic_source()
    day_index = len(frame) // s.ROWS_PER_DAY - 1
    start = day_index * s.ROWS_PER_DAY
    stop = start + s.ROWS_PER_DAY
    final_um_quote = np.full(s.ROWS_PER_DAY, 0.8)
    final_um_signed = np.concatenate(
        [np.full(72, -0.4), np.full(s.ROWS_PER_DAY - 72, 0.8)]
    )
    frame.loc[start : stop - 1, "um_quote_notional"] = final_um_quote
    frame.loc[start : stop - 1, "um_signed_quote_notional"] = final_um_signed

    features, _ = s.build_daily_features(frame)
    final = features.iloc[-1]
    assert final["gap_bars"] == 109
    assert final["gap_threshold"] == pytest.approx(72.0)
    assert bool(final["gap_pass"])


def test_missing_buffer_row_cancels_pair_but_later_defect_does_not() -> None:
    baseline = _synthetic_source()
    features, _ = s.build_daily_features(baseline)
    target_day = pd.Timestamp("2020-05-04", tz="UTC")
    assert target_day in set(features["source_day"])

    broken = baseline.copy()
    day_index = len(broken) // s.ROWS_PER_DAY - 1
    # UM late anchor is index 143, so index 144 is the required buffer row.
    broken.loc[
        day_index * s.ROWS_PER_DAY + 144,
        "_row_valid",
    ] = False
    rejected, _ = s.build_daily_features(broken)
    assert target_day not in set(rejected["source_day"])


def test_stale_laggard_requires_immediately_prior_complete_day() -> None:
    frame = _synthetic_source()
    day_index = len(frame) // s.ROWS_PER_DAY - 1
    prior_day_start = (day_index - 1) * s.ROWS_PER_DAY
    # A late prior-day defect does not affect that day's live event, but it
    # prevents the day from serving as the complete stale-flow source.
    frame.loc[prior_day_start + 250, "_row_valid"] = False
    features, _ = s.build_daily_features(frame)
    final = features.iloc[-1]
    assert bool(final["primary"])
    assert not bool(final["stale_valid"])


def test_lower_median_is_not_linear_median() -> None:
    assert s._lower_median([10, 20, 30, 40]) == 20
    assert s._lower_median([10, 20, 30]) == 20


def test_candidate_latency_hold_and_identity_are_frozen() -> None:
    row = s.raw_candidates(_features([_feature_row()]), "primary").iloc[0]
    assert row["signal_available_time"] == row["resolution_time"] + s.BAR
    assert row["decision_time"] == row["resolution_time"] + 2 * s.BAR
    assert row["entry_time"] == row["decision_time"]
    assert row["exit_time"] == row["entry_time"] + 72 * s.BAR
    assert row["side"] == "LONG"
    assert row["leader"] == "spot"
    assert row["signal_id"] == s.signal_id(
        "primary",
        row["source_day"],
        row["causal_origin_time"],
        row["resolution_time"],
        row["signal_available_time"],
        row["decision_time"],
        row["entry_time"],
        row["exit_time"],
        row["side"],
        row["leader"],
    )
    expected_identity = {
        "causal_origin_time": s._format_time(row["causal_origin_time"]),
        "control": "primary",
        "decision_time": s._format_time(row["decision_time"]),
        "entry_time": s._format_time(row["entry_time"]),
        "exit_time": s._format_time(row["exit_time"]),
        "leader": "spot",
        "policy": asdict(s.prereg.Policy()),
        "policy_id": "CVICR-72",
        "resolution_time": s._format_time(row["resolution_time"]),
        "side": "LONG",
        "signal_available_time": s._format_time(
            row["signal_available_time"]
        ),
        "source_day": s._format_day(row["source_day"]),
        "source_sha256": s.prereg.SOURCE_SHA256,
    }
    assert row["signal_id"] == s.canonical_hash(expected_identity)


def test_control_masks_and_same_clock_side_controls() -> None:
    row = _feature_row()
    row["primary"] = False
    row["leader_persistence"] = False
    row["late_alignment"] = False
    row["late_alignment_only"] = False
    row["no_gap_tail"] = False
    controls, raw_counts = s.build_controls(_features([row]))

    assert raw_counts["primary"] == 0
    assert raw_counts["gap_only"] == 1
    assert raw_counts["initial_conflict_only"] == 1
    assert raw_counts["no_leader_persistence"] == 1
    assert controls["exact_direction_flip"].empty
    assert controls["deterministic_random_side"].empty


def test_random_side_uses_frozen_entry_hash_rule() -> None:
    controls, _ = s.build_controls(_features([_feature_row()]))
    primary = controls["primary"].iloc[0]
    random = controls["deterministic_random_side"].iloc[0]
    digest = hashlib.sha256(
        f"CVICR-72|{s._format_time(primary['entry_time'])}".encode("ascii")
    ).digest()
    assert random["side"] == ("LONG" if digest[0] < 128 else "SHORT")
    assert random["entry_time"] == primary["entry_time"]
    assert random["exit_time"] == primary["exit_time"]


def test_global_reservation_precedes_split_containment() -> None:
    crossing = _candidate(
        "2022-12-31T22:00:00Z",
        hold_bars=36,
    )
    suppressed = _candidate(
        "2023-01-01T00:30:00Z",
        side=-1,
        leader="um",
    )
    reserved = s.reserve_nonoverlap(
        pd.DataFrame([crossing, suppressed], columns=s.CLOCK_COLUMNS)
    )
    assert len(reserved) == 1
    assert s._contained(reserved, s.TRAIN_START, s.TRAIN_END).empty
    assert s._contained(
        reserved, s.SELECTION_START, s.SELECTION_END
    ).empty


def test_entry_equal_previous_exit_is_accepted() -> None:
    first = _candidate("2022-01-01T06:00:00Z")
    second = _candidate("2022-01-01T12:00:00Z", side=-1, leader="um")
    reserved = s.reserve_nonoverlap(
        pd.DataFrame([first, second], columns=s.CLOCK_COLUMNS)
    )
    assert len(reserved) == 2


def test_delayed_controls_recompute_entry_exit_and_reservation() -> None:
    features = _features(
        [
            _feature_row("2022-01-01", late=144),
            _feature_row("2022-01-01", early=145, late=216),
        ]
    )
    controls, _ = s.build_controls(features)
    delayed = controls["one_bar_execution_delay"].iloc[0]
    assert delayed["entry_time"] == delayed["decision_time"] + s.BAR
    assert delayed["exit_time"] == delayed["entry_time"] + 72 * s.BAR


def test_empty_required_controls_fail_source_support() -> None:
    empty = pd.DataFrame(columns=s.CLOCK_COLUMNS)
    controls = {name: empty.copy() for name in s.prereg.CONTROL_ORDER}
    _, checks, selectivity = s.support_checks(controls)
    assert checks["train_events_min"] is False
    assert checks["mechanism_count_ratio_selectivity"] is False
    assert checks["fixed_expected_time_entry_selectivity"] is False
    assert selectivity["gap_only"]["train"]["primary_over_control"] is None


def test_first_failure_is_explicit_and_stage_ordered() -> None:
    base = {
        "train_events_min": True,
        "mechanism_count_ratio_selectivity": True,
        "fixed_expected_time_entry_selectivity": True,
        "stale_laggard_entry_selectivity": True,
    }
    checks = dict(base)
    checks["train_events_min"] = False
    checks["mechanism_count_ratio_selectivity"] = False
    assert s.first_failure(
        checks, {}, artifact_eligible=True
    ) == ("source_support", "train_events_min")

    checks = dict(base)
    checks["fixed_expected_time_entry_selectivity"] = False
    assert s.first_failure(
        checks, {}, artifact_eligible=True
    ) == (
        "mechanism_selectivity",
        "fixed_expected_time_entry_selectivity",
    )
    assert s.first_failure(
        base, {"comparator:exact": False}, artifact_eligible=True
    ) == ("novelty", "comparator:exact")
    assert s.first_failure(
        base, {"comparator:exact": True}, artifact_eligible=True
    ) == ("none", None)


def test_clock_stats_sorts_before_side_and_leader_runs() -> None:
    rows = pd.DataFrame(
        [
            _candidate("2022-01-03T06:00:00Z", side=-1, leader="um"),
            _candidate("2022-01-01T06:00:00Z", side=1, leader="spot"),
            _candidate("2022-01-02T06:00:00Z", side=1, leader="spot"),
        ],
        columns=s.CLOCK_COLUMNS,
    )
    stats = s.clock_stats(rows)
    assert stats["maximum_same_side_run"] == 2
    assert stats["maximum_same_leader_run"] == 2


def test_tolerant_matching_is_one_to_one() -> None:
    left = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2022-01-01T00:00:00Z", "2022-01-01T00:05:00Z"],
                utc=True,
            )
        }
    )
    right = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2022-01-01T00:04:00Z"],
                utc=True,
            )
        }
    )
    assert s.maximum_tolerant_matches(
        left["entry_time"], right["entry_time"], s.BAR
    ) == 1
    assert s.tolerant_entry_jaccard(left, right, s.BAR) == pytest.approx(0.5)


def test_occupancy_correlation_fails_closed_on_zero_variance() -> None:
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = pd.Timestamp("2022-01-02T00:00:00Z")
    empty = pd.DataFrame(columns=["entry_time", "exit_time", "side_sign"])
    active = pd.DataFrame(
        {
            "entry_time": [start + s.BAR],
            "exit_time": [start + 3 * s.BAR],
            "side_sign": [1],
        }
    )
    correlation, position = s.occupancy_metrics(empty, active, start, end)
    assert correlation is None
    assert position == 0.0


def _write_comparator(
    path: Path,
    rows: list[tuple[str, str, str, str, str]],
) -> list[str]:
    header = ["group", "entry", "exit", "side", "unused"]
    path.write_text(
        ",".join(header)
        + "\n"
        + "".join(",".join(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return header


def _comparator_contract(
    path: Path,
    header: list[str],
    *,
    selected_groups: list[str],
    identifier: str = "SYNTH",
    coverage: tuple[str, str] = (
        "2020-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ),
) -> dict[str, object]:
    return {
        "id": identifier,
        "path": str(path),
        "sha256": s.sha256_file(path),
        "header": header,
        "header_sha256": "test-only",
        "entry_column": "entry",
        "exit_column": "exit",
        "side_column": "side",
        "side_encoding": {"LONG": 1, "SHORT": -1},
        "group_column": "group",
        "selected_groups": selected_groups,
        "declared_coverage": list(coverage),
        "six_hour_tolerant_gate": identifier.startswith("IV"),
    }


def test_comparator_decoder_binds_hash_header_usecols_and_groups(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "clock.csv"
    header = _write_comparator(
        path,
        [
            (
                "V01",
                "2022-01-01T00:00:00Z",
                "2022-01-01T01:00:00Z",
                "LONG",
                "ignored",
            ),
            (
                "V02",
                "2022-01-02T00:00:00Z",
                "2022-01-02T01:00:00Z",
                "SHORT",
                "ignored",
            ),
            (
                "V01",
                "2023-01-02T00:00:00Z",
                "2023-01-02T01:00:00Z",
                "LONG",
                "outside-declared-CVTT-coverage",
            ),
        ],
    )
    contract = _comparator_contract(
        path,
        header,
        selected_groups=["V01", "V02"],
        identifier="CVTT-SYNTH",
        coverage=(
            "2020-01-01T00:00:00Z",
            "2023-01-01T00:00:00Z",
        ),
    )
    real_read_csv = pd.read_csv
    captured: list[list[str]] = []

    def recording_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        captured.append(list(kwargs["usecols"]))
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(s.pd, "read_csv", recording_read_csv)
    groups, decoded = s._read_comparator_groups(
        {"novelty_contract": {"comparators": [contract]}}
    )

    assert decoded == 3
    assert set(groups) == {"CVTT-SYNTH:V01", "CVTT-SYNTH:V02"}
    assert len(groups["CVTT-SYNTH:V01"]["rows"]) == 1
    assert groups["CVTT-SYNTH:V01"]["end"] == pd.Timestamp(
        "2023-01-01T00:00:00Z"
    )
    assert set(captured[0]) == {"group", "entry", "exit", "side"}
    assert "unused" not in captured[0]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("hash", "hash drift"),
        ("header", "header drift"),
        ("empty", "group empty"),
        ("side", "side invalid"),
        ("duplicate", "entries duplicated"),
    ],
)
def test_comparator_decoder_fails_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    path = tmp_path / f"{mutation}.csv"
    rows = [
        (
            "A",
            "2022-01-01T00:00:00Z",
            "2022-01-01T01:00:00Z",
            "LONG",
            "x",
        )
    ]
    if mutation == "side":
        rows[0] = (*rows[0][:3], "BAD", rows[0][4])
    if mutation == "duplicate":
        rows.append(rows[0])
    header = _write_comparator(path, rows)
    contract = _comparator_contract(
        path,
        header,
        selected_groups=["MISSING" if mutation == "empty" else "A"],
    )
    if mutation == "hash":
        contract["sha256"] = "0" * 64
    if mutation == "header":
        contract["header"] = ["wrong"]
    with pytest.raises(RuntimeError, match=message):
        s._read_comparator_groups(
            {"novelty_contract": {"comparators": [contract]}}
        )


def test_occupancy_rejects_overlapping_selected_comparator_group() -> None:
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = pd.Timestamp("2022-01-02T00:00:00Z")
    overlapping = pd.DataFrame(
        {
            "entry_time": [start + s.BAR, start + 2 * s.BAR],
            "exit_time": [start + 4 * s.BAR, start + 5 * s.BAR],
            "side_sign": [1, -1],
        }
    )
    with pytest.raises(RuntimeError, match="overlaps itself"):
        s._signed_occupancy(overlapping, start, end)


def test_novelty_applies_six_hour_gate_only_to_intrinsic_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = pd.Timestamp("2022-01-03T00:00:00Z")
    primary = pd.DataFrame(
        [_candidate("2022-01-01T06:00:00Z")],
        columns=s.CLOCK_COLUMNS,
    )
    comparator = pd.DataFrame(
        {
            "entry_time": [pd.Timestamp("2022-01-01T06:00:00Z")],
            "exit_time": [pd.Timestamp("2022-01-01T12:00:00Z")],
            "side_sign": [1],
        }
    )
    groups = {
        "dense": {
            "rows": comparator,
            "start": start,
            "end": end,
            "six_hour_gate": False,
            "artifact_id": "DENSE",
            "selected_group": "primary",
        },
        "intrinsic": {
            "rows": comparator,
            "start": start,
            "end": end,
            "six_hour_gate": True,
            "artifact_id": "IVLIR",
            "selected_group": "primary",
        },
    }
    monkeypatch.setattr(
        s,
        "_read_comparator_groups",
        lambda payload: (groups, 2),
    )
    payload = {
        "novelty_contract": {
            "comparators": [],
            "exact_entry_jaccard_max": 0.10,
            "one_bar_tolerant_jaccard_max": 0.20,
            "twelve_bar_tolerant_jaccard_max": 0.35,
            "six_hour_tolerant_jaccard_intrinsic_family_max": 0.60,
            "absolute_signed_occupancy_pearson_max": 0.40,
        }
    }
    _, checks, decoded = s.evaluate_novelty(primary, payload)
    assert decoded == 2
    assert "dense:six_hour_tolerant_jaccard" not in checks
    assert "intrinsic:six_hour_tolerant_jaccard" in checks


def test_deterministic_clock_gzip_and_no_outcome_columns() -> None:
    controls, _ = s.build_controls(_features([_feature_row()]))
    first = s.deterministic_clock_bytes(controls)
    second = s.deterministic_clock_bytes(controls)
    assert first == second
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as zipped:
        header = zipped.readline().decode().rstrip("\n").split(",")
    assert header == list(s.CLOCK_COLUMNS)
    assert not any(
        token in column
        for column in header
        for token in s.FORBIDDEN_CLOCK_TOKENS
    )


def test_write_once_accepts_identical_and_rejects_drift(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    assert s._write_once(output, b"first") == "created"
    assert s._write_once(output, b"first") == "verified_existing"
    with pytest.raises(RuntimeError, match="noncanonical existing"):
        s._write_once(output, b"second")
