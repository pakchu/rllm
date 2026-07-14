from __future__ import annotations

import pandas as pd

from training.evaluate_causal_weak_signal_sequence_interactions import (
    InteractionSpec,
    _last_signal_state,
    _recent_relation_mask,
    build_interaction_schedule,
)


def _schedule(rows: list[tuple[int, int]]) -> pd.DataFrame:
    values = []
    dates = pd.date_range("2022-01-01", periods=100, freq="5min")
    for position, side in rows:
        entry = position + 1
        exit_ = entry + 3
        values.append(
            {
                "signal_position": position,
                "entry_position": entry,
                "exit_position": exit_,
                "signal_date": str(dates[position]),
                "entry_date": str(dates[entry]),
                "exit_date": str(dates[exit_]),
                "side": side,
                "branch": "primary",
                "hold_bars": 3,
            }
        )
    return pd.DataFrame(values)


def test_recent_relation_uses_only_same_or_past_signal() -> None:
    trigger = _schedule([(10, 1), (20, 1), (30, -1)])
    antecedent = _schedule([(11, 1), (18, 1), (25, 1)])
    state = _last_signal_state(antecedent, 100)
    mask = _recent_relation_mask(
        trigger,
        state,
        minimum_age_bars=0,
        maximum_age_bars=5,
        same_side=True,
    )
    assert mask.tolist() == [False, True, False]


def test_interaction_filters_trigger_without_adding_or_retiming_trades() -> None:
    schedules = {
        "umfr": _schedule([(10, 1), (30, -1), (50, 1)]),
        "catch": _schedule([(8, 1), (29, 1), (45, 1)]),
        "cspr": _schedule([]),
        "clasp": _schedule([]),
        "luri": _schedule([]),
        "rift": _schedule([]),
    }
    spec = InteractionSpec("o1", "umfr", "catch", 12)
    selected = build_interaction_schedule(spec, schedules, frame_length=100)
    assert selected["signal_position"].tolist() == [10, 50]
    assert selected["entry_position"].tolist() == [11, 51]
    assert selected["exit_position"].tolist() == [14, 54]


def test_long_only_and_opposite_derivative_veto_are_causal() -> None:
    schedules = {
        "rift": _schedule([(20, 1), (40, 1), (60, -1)]),
        "catch": _schedule([(18, 1), (38, 1), (58, -1)]),
        "umfr": _schedule([(39, -1)]),
        "luri": _schedule([]),
        "cspr": _schedule([]),
        "clasp": _schedule([]),
    }
    spec = InteractionSpec(
        "o3",
        "rift",
        "catch",
        12,
        long_only=True,
        veto_opposite_derivative=True,
    )
    selected = build_interaction_schedule(spec, schedules, frame_length=100)
    assert selected["signal_position"].tolist() == [20]
