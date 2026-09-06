from __future__ import annotations

import pandas as pd
import pytest

from training import (
    build_high_volatility_cross_structure_action_vote_support as support,
)
from training import preregister_high_volatility_cross_structure_action_vote as prereg


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
        "CARSC-8": _action(
            "CARSC-8", ["2023-07-01T00:00:00Z", "2023-07-01T16:00:00Z"], [1, -1]
        ),
        "HVCMMI-8": _action(
            "HVCMMI-8",
            [
                "2023-07-01T00:00:00Z",
                "2023-07-01T08:00:00Z",
                "2023-07-01T16:00:00Z",
            ],
            [1, 1, -1],
        ),
        "HVIABR-8": _action(
            "HVIABR-8", ["2023-07-01T00:00:00Z", "2023-07-01T08:00:00Z"], [-1, -1]
        ),
        "HVTFR-8": _action(
            "HVTFR-8", ["2023-07-01T08:00:00Z", "2023-07-02T00:00:00Z"], [1, 1]
        ),
        "HVSVF-8": _action("HVSVF-8", ["2023-07-01T08:00:00Z"], [-1]),
    }


def _state() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_time": pd.date_range(
                "2023-07-01T00:00:00Z", periods=5, freq="8h"
            ),
            "source_valid": [True, True, True, True, True],
            "concentration_rank": [0.80, 0.90, 0.99, 0.80, 0.79],
        },
        columns=support.STATE_COLUMNS,
    )


def test_vote_uses_all_exact_decision_actions_and_skips_tie_or_no_quorum() -> None:
    clock, accounting = support.build_action_vote_clock(_actions(), _state())

    assert clock["decision_time"].tolist() == [
        pd.Timestamp("2023-07-01T00:00:00Z"),
        pd.Timestamp("2023-07-01T16:00:00Z"),
    ]
    assert clock["side"].tolist() == [1, -1]
    assert clock["active_action_count"].tolist() == [3, 2]
    assert clock["long_vote_count"].tolist() == [2, 0]
    assert clock["short_vote_count"].tolist() == [1, 2]
    assert accounting == {
        "eligible_state_decisions": 4,
        "majority_action_decisions": 2,
        "no_quorum_decisions": 1,
        "tie_decisions": 1,
        "action_vote_counts": {
            "CARSC-8": 2,
            "HVCMMI-8": 3,
            "HVIABR-8": 2,
            "HVTFR-8": 2,
            "HVSVF-8": 1,
        },
        "active_action_count_distribution": {
            "0": 0,
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 1,
            "5": 0,
        },
        "long_majority_decisions": 1,
        "short_majority_decisions": 1,
    }


def test_vote_preserves_d_plus_five_eight_hours_and_nonoverlap() -> None:
    actions = _actions()
    actions["HVCMMI-8"].loc[0, "feature_available_time"] += pd.Timedelta("4m")
    clock, _ = support.build_action_vote_clock(actions, _state())
    assert clock["entry_time"].eq(
        clock["decision_time"] + pd.Timedelta("5m")
    ).all()
    assert clock["exit_time"].eq(clock["entry_time"] + pd.Timedelta("8h")).all()
    assert clock.loc[0, "feature_available_time"] == pd.Timestamp(
        "2023-07-01T00:04:00Z"
    )
    assert clock["entry_time"].iloc[1] >= clock["exit_time"].iloc[0]

    broken = _actions()
    broken["CARSC-8"].loc[0, "exit_time"] += pd.Timedelta("1ns")
    with pytest.raises(RuntimeError, match="action hold drift"):
        support.build_action_vote_clock(broken, _state())


def test_state_gate_is_exact_source_valid_and_concentration_rank_boundary() -> None:
    state = _state()
    state["source_valid"] = [True, False, True, True, True]
    clock, accounting = support.build_action_vote_clock(_actions(), state)
    assert pd.Timestamp("2023-07-01T08:00:00Z") not in set(clock["decision_time"])
    assert accounting["eligible_state_decisions"] == 3

    broken = _state()
    broken.loc[1, "decision_time"] += pd.Timedelta("1ns")
    with pytest.raises(RuntimeError, match="off exact 8h grid|not contiguous exact 8h"):
        support.build_action_vote_clock(_actions(), broken)


def test_support_stats_compute_every_preregistered_stage_gate() -> None:
    clock, _ = support.build_action_vote_clock(_actions(), _state())
    stats = {
        split: support.support_stats(clock, split)
        for split in prereg.build()["stages"]
    }
    checks = support._support_checks(stats)
    assert set(checks) == {
        f"{stage}_{gate}"
        for stage in prereg.build()["stages"]
        for gate in ("minimum_events", "side_balance", "month_concentration")
    }
    assert stats["train"] == {
        "events": 2,
        "longs": 1,
        "shorts": 1,
        "minority_side_share": 0.5,
        "max_month_share": 1.0,
    }


def test_frozen_hashes_seals_and_double_materialization_are_deterministic(
    tmp_path,
) -> None:
    assert (
        support.PREREG_SHA
        == "340627ccd4928acb6297f0959fd001cc07066e05e00bfde98db88ae0cb0c550e"
    )
    verified = support.verify_frozen_inputs()
    assert tuple(verified) == prereg.ACTION_ORDER + (prereg.ELIGIBILITY_ID,)

    clock_path = tmp_path / "clock.csv.gz"
    result_path = tmp_path / "support.json"
    first = support.run(clock_path, result_path)
    first_result = result_path.read_bytes()
    first_clock = clock_path.read_bytes()
    second = support.run(clock_path, result_path)

    assert result_path.read_bytes() == first_result
    assert clock_path.read_bytes() == first_clock
    assert second == first
    assert first["manifest_hash"] == support.canonical_hash(
        {key: value for key, value in first.items() if key != "manifest_hash"}
    )
    assert len(first["support_checks"]) == 12
    assert first["action_vote_postentry_returns_or_pnl_opened"] is False
    assert first["entry_exit_prices_opened"] is False
    assert first["returns_opened"] is False
    assert first["funding_opened"] is False
    assert first["pnl_opened"] is False
    assert first["gross9_comparator_rows_opened"] is False
    assert first["advance_to_economic_outcomes"] is False
