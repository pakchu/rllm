from __future__ import annotations

import pandas as pd
import pytest

from training import evaluate_cross_collateral_book_validated_flow_rejection_2023 as ev


def _clock() -> pd.DataFrame:
    rows = []
    for index in range(144):
        signal = pd.Timestamp("2023-01-01") + pd.Timedelta(hours=7 * index)
        entry = signal + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=6)
        rows.append(
            {
                "quarter": f"q{entry.quarter}",
                "signal_position": index * 84,
                "entry_position": index * 84 + 1,
                "exit_position": index * 84 + 73,
                "signal_date": signal,
                "entry_date": entry,
                "exit_date": exit_,
                "side": 1 if index % 2 else -1,
                "branch": "synthetic",
                "hold_bars": 72,
            }
        )
    return pd.DataFrame(rows)


def test_validate_clock_enforces_next_open_and_six_hours() -> None:
    frame = _clock()
    assert ev.validate_clock(frame) is frame
    broken = frame.copy()
    broken.loc[0, "exit_position"] += 1
    with pytest.raises(RuntimeError, match="six-hour exit"):
        ev.validate_clock(broken)


def test_transform_controls_do_not_change_hold_or_primary() -> None:
    frame = _clock()
    string_clock = frame.copy()
    for column in ("signal_date", "entry_date", "exit_date"):
        string_clock[column] = string_clock[column].astype(str)
    delayed = ev.transform_clock(string_clock, "delay_five_minutes")
    assert (delayed["exit_date"] - delayed["entry_date"]).eq(
        pd.Timedelta(hours=6)
    ).all()
    assert frame.iloc[0]["entry_date"] != delayed.iloc[0]["entry_date"]
    flipped = ev.transform_clock(frame, "direction_flip")
    assert flipped["side"].eq(-frame["side"]).all()


def test_selection_checks_require_every_robustness_gate() -> None:
    good = {
        "absolute_return_pct": 10.0,
        "cagr_pct": 10.0,
        "strict_mdd_pct": 2.0,
        "cagr_to_strict_mdd": 5.0,
        "trades": 140,
        "mean_net_bps": 10.0,
    }
    primary = {name: dict(good) for name in ev.WINDOWS}
    checks = ev.selection_checks(
        primary,
        long_only=dict(good),
        short_only=dict(good),
        stress=dict(good),
        delayed=dict(good),
        flipped={**good, "cagr_pct": -1.0},
        without_book={**good, "mean_net_bps": 5.0},
        signflip={"p_value_one_sided": 0.05},
    )
    assert all(checks.values())
    primary["q4"]["absolute_return_pct"] = -0.1
    failed = ev.selection_checks(
        primary,
        long_only=dict(good),
        short_only=dict(good),
        stress=dict(good),
        delayed=dict(good),
        flipped={**good, "cagr_pct": -1.0},
        without_book={**good, "mean_net_bps": 5.0},
        signflip={"p_value_one_sided": 0.05},
    )
    assert failed["every_quarter_absolute_return_positive"] is False


def test_outcome_paths_and_parameters_are_immutable() -> None:
    assert ev.CONFIG.leverage == 0.5
    assert ev.CONFIG.base_cost_bp_per_notional_side == 6.0
    assert ev.CONFIG.stress_cost_bp_per_notional_side == 10.0
    with pytest.raises(ValueError, match="immutable"):
        ev.run("/tmp/not-cbfr.json", ev.DEFAULT_DOCS)
