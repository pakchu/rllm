from __future__ import annotations

import pandas as pd
import pytest

from training import (
    build_high_volatility_cross_structure_priority_router_support as support,
)
from training import (
    preregister_high_volatility_cross_structure_priority_router as prereg,
)


def _action(action: str, decisions: list[str], sides: list[int]) -> pd.DataFrame:
    decision = pd.to_datetime(decisions, utc=True)
    return pd.DataFrame(
        {
            "candidate": action,
            "control": "primary",
            "split": "train",
            "decision_time": decision,
            "feature_available_time": decision,
            "entry_time": decision + pd.Timedelta("5m"),
            "exit_time": decision + pd.Timedelta("8h5m"),
            "side": sides,
        },
        columns=support.ACTION_COLUMNS,
    )


def _actions() -> dict[str, pd.DataFrame]:
    return {
        "CARSC-8": _action("CARSC-8", ["2023-07-01T00:00:00Z"], [1]),
        "HVCMMI-8": _action(
            "HVCMMI-8",
            ["2023-07-01T00:00:00Z", "2023-07-01T08:00:00Z"],
            [-1, 1],
        ),
        "HVIABR-8": _action("HVIABR-8", ["2023-07-01T08:00:00Z"], [-1]),
    }


def _state(eligibility: str) -> pd.DataFrame:
    decisions = pd.to_datetime(
        [
            "2023-07-01T00:00:00Z",
            "2023-07-01T08:00:00Z",
            "2023-07-01T16:00:00Z",
        ],
        utc=True,
    )
    rank = support.FILTER_COLUMNS[eligibility][-1]
    values = [0.9, 0.9, 0.9] if eligibility == "HVTCCR-8" else [0.2, 0.2, 0.2]
    return pd.DataFrame(
        {"decision_time": decisions, "source_valid": True, rank: values},
        columns=support.FILTER_COLUMNS[eligibility],
    )


def test_router_chooses_first_active_and_cash_without_materializing_position() -> None:
    clock, routing = support.route_priority_clock(
        prereg.PRIORITY_ORDERS[0], "HVTCCR-8", _actions(), _state("HVTCCR-8")
    )
    assert clock["action_id"].tolist() == ["CARSC-8", "HVCMMI-8"]
    assert clock["side"].tolist() == [1, 1]
    assert routing == {
        "eligible_state_decisions": 3,
        "routed_action_decisions": 2,
        "cash_decisions": 1,
        "selected_action_counts": {"CARSC-8": 1, "HVCMMI-8": 1, "HVIABR-8": 0},
    }


def test_reversed_priority_preserves_selected_action_clock() -> None:
    actions = _actions()
    clock, _ = support.route_priority_clock(
        prereg.PRIORITY_ORDERS[1], "HVLZC-8", actions, _state("HVLZC-8")
    )
    assert clock["action_id"].tolist() == ["HVCMMI-8", "HVIABR-8"]
    chosen = actions["HVIABR-8"].iloc[0]
    row = clock.iloc[1]
    for field in (
        "control",
        "split",
        "decision_time",
        "side",
        "feature_available_time",
        "entry_time",
        "exit_time",
    ):
        assert row[field] == chosen[field]


def test_exact_state_grid_and_routed_nonoverlap_fail_closed() -> None:
    broken = _state("HVTCCR-8")
    broken.loc[1, "decision_time"] += pd.Timedelta("1ns")
    with pytest.raises(RuntimeError, match="off exact 8h grid|not contiguous exact 8h"):
        support.route_priority_clock(
            prereg.PRIORITY_ORDERS[0], "HVTCCR-8", _actions(), broken
        )

    actions = _actions()
    actions["CARSC-8"].loc[0, "exit_time"] += pd.Timedelta("1ns")
    with pytest.raises(RuntimeError, match="action hold drift"):
        support.route_priority_clock(
            prereg.PRIORITY_ORDERS[0], "HVTCCR-8", actions, _state("HVTCCR-8")
        )


def test_state_filter_is_exact_frozen_source_valid_and_rank_only() -> None:
    state = _state("HVTCCR-8")
    state["source_valid"] = [True, True, False]
    state["concentration_rank"] = [0.80, 0.79, 0.99]
    clock, routing = support.route_priority_clock(
        prereg.PRIORITY_ORDERS[0], "HVTCCR-8", _actions(), state
    )
    assert clock["decision_time"].tolist() == [state.loc[0, "decision_time"]]
    assert routing["eligible_state_decisions"] == 1


def test_support_stats_and_all_stage_gates_match_preregistration() -> None:
    entries = pd.to_datetime(
        ["2023-07-01T00:05:00Z", "2023-07-02T00:05:00Z", "2023-08-01T00:05:00Z"],
        utc=True,
    )
    frame = pd.DataFrame({"split": "train", "entry_time": entries, "side": [1, -1, 1]})
    assert support.support_stats(frame, "train") == {
        "events": 3,
        "longs": 2,
        "shorts": 1,
        "minority_side_share": pytest.approx(1 / 3),
        "max_month_share": pytest.approx(2 / 3),
    }
    stats = {
        stage: support.support_stats(frame, stage) for stage in prereg.build()["stages"]
    }
    assert set(support._support_checks(stats)) == {
        f"{stage}_{gate}"
        for stage in prereg.build()["stages"]
        for gate in ("minimum_events", "side_balance", "month_concentration")
    }


def test_frozen_inputs_and_double_materialization_are_deterministic(tmp_path) -> None:
    assert (
        support.PREREG_SHA
        == "d31652fd7bf80b9a2da73a15f4c4235d08b2f40ae583a0f90fd3b32dfd775bc0"
    )
    verified = support.verify_frozen_inputs()
    assert tuple(verified) == prereg.ACTION_ORDER + prereg.ELIGIBILITY_ORDER

    clock_dir = tmp_path / "clocks"
    result_path = tmp_path / "support.json"
    first = support.run(clock_dir, result_path)
    first_bytes = result_path.read_bytes()
    first_clocks = {
        candidate: row["clock"]["sha256"]
        for candidate, row in first["candidates"].items()
    }
    second = support.run(clock_dir, result_path)
    assert result_path.read_bytes() == first_bytes
    assert {
        candidate: row["clock"]["sha256"]
        for candidate, row in second["candidates"].items()
    } == first_clocks
    assert tuple(first["candidates"]) == prereg.CANDIDATE_FAMILY
    assert first["router_postentry_returns_or_pnl_opened"] is False
    assert first["entry_exit_prices_opened"] is False
    assert first["returns_opened"] is False
    assert first["funding_opened"] is False
    assert first["pnl_opened"] is False
    assert first["gross9_comparator_rows_opened"] is False
    assert first["advance_to_economic_outcomes"] is False
    assert first["eligible_routers_for_combination_gross9"] == [
        "CARSC-8__THEN__HVCMMI-8__THEN__HVIABR-8__ELIGIBLE_BY__HVTCCR-8",
        "HVIABR-8__THEN__HVCMMI-8__THEN__CARSC-8__ELIGIBLE_BY__HVTCCR-8",
    ]
    assert first["eligible_router_count"] == 2
    assert first["eligible_router_count"] == len(
        first["eligible_routers_for_combination_gross9"]
    )
