from __future__ import annotations

import pandas as pd
import pytest

from training import build_high_volatility_state_ordered_filter_support as support
from training import preregister_high_volatility_state_ordered_filter as prereg


def _action(action: str) -> pd.DataFrame:
    decisions = pd.to_datetime(
        ["2023-07-01T00:00:00Z", "2023-07-01T08:00:00Z", "2023-07-01T16:00:00Z"],
        utc=True,
    )
    return pd.DataFrame(
        {
            "candidate": action,
            "control": "primary",
            "split": "train",
            "decision_time": decisions,
            "feature_available_time": decisions,
            "entry_time": decisions + pd.Timedelta("5m"),
            "exit_time": decisions + pd.Timedelta("8h5m"),
            "side": [1, -1, 1],
        },
        columns=support.ACTION_COLUMNS,
    )


def test_exact_state_filter_uses_only_source_valid_and_frozen_rank() -> None:
    action = prereg.ACTION_ORDER[0]
    clock = _action(action)
    concentration = pd.DataFrame(
        {
            "decision_time": clock["decision_time"],
            "source_valid": [True, True, False],
            "concentration_rank": [0.80, 0.79, 0.99],
        },
        columns=support.FILTER_COLUMNS["HVTCCR-8"],
    )
    result = support.filter_action_clock(action, "HVTCCR-8", clock, concentration)
    assert result["decision_time"].tolist() == [clock.loc[0, "decision_time"]]
    assert result.loc[0, "side"] == clock.loc[0, "side"]
    assert result.loc[0, "entry_time"] == clock.loc[0, "entry_time"]
    assert result.loc[0, "exit_time"] == clock.loc[0, "exit_time"]

    complexity = pd.DataFrame(
        {
            "decision_time": clock["decision_time"],
            "source_valid": [True, True, False],
            "complexity_rank": [0.25, 0.26, 0.01],
        },
        columns=support.FILTER_COLUMNS["HVLZC-8"],
    )
    result = support.filter_action_clock(action, "HVLZC-8", clock, complexity)
    assert result["decision_time"].tolist() == [clock.loc[0, "decision_time"]]


def test_filter_does_not_require_onset_variation_side_or_directional_agreement() -> None:
    action = prereg.ACTION_ORDER[0]
    clock = _action(action)
    state = pd.DataFrame(
        {
            "decision_time": clock["decision_time"],
            "source_valid": True,
            "concentration_rank": [0.90, 0.95, 0.85],
        },
        columns=support.FILTER_COLUMNS["HVTCCR-8"],
    )
    result = support.filter_action_clock(action, "HVTCCR-8", clock, state)
    assert result["side"].tolist() == [1, -1, 1]
    assert result["entry_time"].tolist() == clock["entry_time"].tolist()
    assert result["exit_time"].tolist() == clock["exit_time"].tolist()
    assert result["feature_available_time"].tolist() == clock["feature_available_time"].tolist()


def test_exact_decision_mapping_and_action_availability_fail_closed() -> None:
    action = prereg.ACTION_ORDER[0]
    clock = _action(action)
    state = pd.DataFrame(
        {
            "decision_time": clock["decision_time"] + pd.Timedelta("1ns"),
            "source_valid": True,
            "concentration_rank": 0.9,
        },
        columns=support.FILTER_COLUMNS["HVTCCR-8"],
    )
    with pytest.raises(RuntimeError, match="missing exact filter decision state"):
        support.filter_action_clock(action, "HVTCCR-8", clock, state)

    late = clock.copy()
    late.loc[0, "feature_available_time"] = late.loc[0, "entry_time"] + pd.Timedelta("1ns")
    exact = state.copy()
    exact["decision_time"] = clock["decision_time"]
    with pytest.raises(RuntimeError, match="feature unavailable at entry"):
        support.filter_action_clock(action, "HVTCCR-8", late, exact)


def test_filtering_preserves_reserved_nonoverlap_without_new_reservation() -> None:
    action = prereg.ACTION_ORDER[0]
    clock = _action(action)
    state = pd.DataFrame(
        {
            "decision_time": clock["decision_time"],
            "source_valid": True,
            "complexity_rank": [0.2, 0.5, 0.2],
        },
        columns=support.FILTER_COLUMNS["HVLZC-8"],
    )
    result = support.filter_action_clock(action, "HVLZC-8", clock, state)
    assert result["entry_time"].tolist() == [clock.loc[0, "entry_time"], clock.loc[2, "entry_time"]]
    assert result["exit_time"].tolist() == [clock.loc[0, "exit_time"], clock.loc[2, "exit_time"]]

    overlap = clock.copy()
    overlap.loc[1, "decision_time"] = overlap.loc[0, "decision_time"] + pd.Timedelta("4h")
    overlap.loc[1, "feature_available_time"] = overlap.loc[1, "decision_time"]
    overlap.loc[1, "entry_time"] = overlap.loc[1, "decision_time"] + pd.Timedelta("5m")
    overlap.loc[1, "exit_time"] = overlap.loc[1, "entry_time"] + pd.Timedelta("8h")
    with pytest.raises(RuntimeError, match="reservation overlap"):
        support._validate_reserved_action_clock(overlap, action)


def test_support_stats_and_frozen_all_stage_gates() -> None:
    entries = pd.to_datetime(
        ["2023-07-01T00:05:00Z", "2023-07-02T00:05:00Z", "2023-08-01T00:05:00Z"],
        utc=True,
    )
    frame = pd.DataFrame({"split": "train", "entry_time": entries, "side": [1, -1, 1]})
    stats = support.support_stats(frame, "train")
    assert stats == {
        "events": 3,
        "longs": 2,
        "shorts": 1,
        "minority_side_share": pytest.approx(1 / 3),
        "max_month_share": pytest.approx(2 / 3),
    }
    all_stats = {stage: support.support_stats(frame, stage) for stage in prereg.build()["stages"]}
    checks = support._support_checks(all_stats)
    assert set(checks) == {
        f"{stage}_{gate}"
        for stage in prereg.build()["stages"]
        for gate in ("minimum_events", "side_balance", "month_concentration")
    }


def test_all_frozen_hashes_and_component_pass_states() -> None:
    assert support.PREREG_SHA == "53e78737a357eb183c8247bbca847a8319334400be5390e4bf01861a368e3484"
    verified = support.verify_frozen_inputs()
    assert tuple(verified) == prereg.ACTION_ORDER + prereg.FILTER_ORDER
    assert all(row["source_support_passed"] is True for row in verified.values())
    assert all(row["gross9_passed"] is True for row in verified.values())
    assert all(
        artifact["verified"] is True
        for row in verified.values()
        for artifact in row.values()
        if isinstance(artifact, dict)
    )


def test_run_is_deterministic_and_keeps_sealed_inputs_closed(tmp_path) -> None:
    clock_dir = tmp_path / "clocks"
    result_path = tmp_path / "support.json"
    first = support.run(clock_dir, result_path)
    first_bytes = result_path.read_bytes()
    first_hashes = {
        candidate: row["clock"]["sha256"] for candidate, row in first["candidates"].items()
    }
    second = support.run(clock_dir, result_path)
    assert result_path.read_bytes() == first_bytes
    assert {
        candidate: row["clock"]["sha256"] for candidate, row in second["candidates"].items()
    } == first_hashes
    assert tuple(first["candidates"]) == prereg.CANDIDATE_FAMILY
    assert first["action_primary_clock_fields_opened"] == list(support.ACTION_COLUMNS)
    assert first["eligibility_source_state_fields_opened"] == {
        key: list(value) for key, value in support.FILTER_COLUMNS.items()
    }
    assert first["additional_reservation_applied"] is False
    assert first["filter_postentry_returns_or_pnl_opened"] is False
    assert first["entry_exit_prices_opened"] is False
    assert first["returns_opened"] is False
    assert first["funding_opened"] is False
    assert first["pnl_opened"] is False
    assert first["gross9_comparator_rows_opened"] is False
    assert first["advance_to_economic_outcomes"] is False
    assert first["eligible_candidates_for_combination_gross9"] == [
        candidate for candidate, row in first["candidates"].items() if row["support_passed"]
    ]
