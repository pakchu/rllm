from __future__ import annotations

import pandas as pd
import pytest

from training import build_high_volatility_flow_priority_router_support as support
from training import preregister_high_volatility_flow_priority_router as prereg


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
            "exit_time": decision + pd.Timedelta("6h5m"),
            "side": sides,
        },
        columns=support.ACTION_COLUMNS,
    )


def _actions() -> dict[str, pd.DataFrame]:
    return {
        "HVAFC-6": _action("HVAFC-6", ["2023-07-01T00:00:00Z"], [1]),
        "HVELR-6": _action(
            "HVELR-6",
            ["2023-07-01T00:00:00Z", "2023-07-01T08:00:00Z"],
            [-1, 1],
        ),
        "RIVSCR-6": _action("RIVSCR-6", ["2023-07-01T08:00:00Z"], [-1]),
    }


def _state() -> pd.DataFrame:
    decisions = pd.to_datetime(
        [
            "2023-07-01T00:00:00Z",
            "2023-07-01T08:00:00Z",
            "2023-07-01T16:00:00Z",
        ],
        utc=True,
    )
    return pd.DataFrame(
        {
            "decision_time": decisions,
            "source_valid": True,
            "concentration_rank": [0.9, 0.9, 0.9],
        },
        columns=support.FILTER_COLUMNS["HVTCCR-8"],
    )


def test_router_chooses_first_active_and_cash_without_materializing_position() -> None:
    clock, routing = support.route_priority_clock(
        prereg.PRIORITY_ORDERS[0], _actions(), _state()
    )
    assert clock["action_id"].tolist() == ["HVAFC-6", "HVELR-6"]
    assert clock["side"].tolist() == [1, 1]
    assert routing == {
        "eligible_state_decisions": 3,
        "routed_action_decisions": 2,
        "cash_decisions": 1,
        "selected_action_counts": {"HVAFC-6": 1, "HVELR-6": 1, "RIVSCR-6": 0},
    }


def test_reverse_priority_preserves_exact_six_hour_action_clock() -> None:
    actions = _actions()
    clock, _ = support.route_priority_clock(
        prereg.PRIORITY_ORDERS[1], actions, _state()
    )
    assert clock["action_id"].tolist() == ["HVELR-6", "RIVSCR-6"]
    chosen = actions["RIVSCR-6"].iloc[0]
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
    assert (clock["exit_time"] - clock["entry_time"]).eq(pd.Timedelta("6h")).all()


def test_only_two_frozen_priority_routes_and_all_action_clocks_are_accepted() -> None:
    with pytest.raises(ValueError, match="two frozen orders"):
        support.route_priority_clock(
            ("HVELR-6", "HVAFC-6", "RIVSCR-6"), _actions(), _state()
        )
    missing = _actions()
    del missing["HVELR-6"]
    with pytest.raises(RuntimeError, match="all and only frozen action clocks"):
        support.route_priority_clock(prereg.PRIORITY_ORDERS[0], missing, _state())


def test_exact_state_grid_and_six_hour_action_hold_fail_closed() -> None:
    broken = _state()
    broken.loc[1, "decision_time"] += pd.Timedelta("1ns")
    with pytest.raises(RuntimeError, match="off exact 8h grid|not contiguous exact 8h"):
        support.route_priority_clock(prereg.PRIORITY_ORDERS[0], _actions(), broken)

    actions = _actions()
    actions["HVAFC-6"].loc[0, "exit_time"] += pd.Timedelta("1ns")
    with pytest.raises(RuntimeError, match="action hold drift"):
        support.route_priority_clock(prereg.PRIORITY_ORDERS[0], actions, _state())


def test_hvtccr_eligibility_is_exact_source_valid_and_concentration_rank() -> None:
    state = _state()
    state["source_valid"] = [True, True, False]
    state["concentration_rank"] = [0.80, 0.79, 0.99]
    clock, routing = support.route_priority_clock(
        prereg.PRIORITY_ORDERS[0], _actions(), state
    )
    assert clock["decision_time"].tolist() == [state.loc[0, "decision_time"]]
    assert clock["eligibility_id"].eq("HVTCCR-8").all()
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
    checks = support._support_checks(stats)
    assert set(checks) == {
        f"{stage}_{gate}"
        for stage in prereg.build()["stages"]
        for gate in ("minimum_events", "side_balance", "month_concentration")
    }
    assert all(
        checks[f"{stage}_minimum_events"] is False
        for stage in prereg.build()["stages"]
    )


def test_frozen_inputs_and_double_materialization_are_deterministic(tmp_path) -> None:
    assert support.PREREG_SHA == "c9a6c2799155fa89bf6fdecdfd66a97e5777a468efe5ad290e643eb8704a21c8"
    verified = support.verify_frozen_inputs()
    assert tuple(verified) == prereg.ACTION_ORDER + (prereg.ELIGIBILITY_POLICY,)

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
    assert first["eligible_routers_for_combination_gross9"] == list(
        prereg.CANDIDATE_FAMILY
    )
    assert {
        candidate: [
            row["support"][stage]["events"]
            for stage in ("train", "test", "eval", "final")
        ]
        for candidate, row in first["candidates"].items()
    } == {candidate: [24, 40, 61, 35] for candidate in prereg.CANDIDATE_FAMILY}
    assert all(
        all(row["support_checks"].values())
        for row in first["candidates"].values()
    )
    assert first["eligible_router_count"] == len(
        first["eligible_routers_for_combination_gross9"]
    )
