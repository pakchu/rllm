from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import build_perp_only_wick_rejection_support as support
from training import preregister_perp_only_wick_rejection as prereg


def _minute_frame(*, missing: set[int] | None = None) -> pd.DataFrame:
    missing = set() if missing is None else missing
    rows = [index for index in range(15) if index not in missing]
    price = np.asarray([100.0 + 0.01 * index for index in rows])
    return pd.DataFrame(
        {
            "date": pd.to_datetime("2023-01-01") + pd.to_timedelta(rows, unit="min"),
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price + 0.1,
        }
    )


def _frame(rows: int = 20) -> pd.DataFrame:
    frame = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=rows, freq="5min")})
    for venue in ("perp", "spot"):
        frame[f"{venue}_open"] = 100.0
        frame[f"{venue}_high"] = 100.0
        frame[f"{venue}_low"] = 100.0
        frame[f"{venue}_close"] = 100.0
        frame[f"{venue}_complete"] = True
    frame["joint_complete"] = True
    return frame


def _policy() -> prereg.Policy:
    return replace(
        prereg.Policy(),
        baseline_bars=2,
        baseline_min_periods=2,
        wick_excess_quantile=0.5,
        minimum_perp_wick_bp=1.0,
    )


def test_five_minute_aggregation_requires_all_five_minutes() -> None:
    complete = support.aggregate_five_minute(_minute_frame(), prefix="x_")
    incomplete = support.aggregate_five_minute(_minute_frame(missing={6}), prefix="x_")
    assert complete["x_complete"].tolist() == [True, True, True]
    assert incomplete["x_complete"].tolist() == [True, False, True]
    assert complete.loc[0, "x_open"] == pytest.approx(100.0)
    assert complete.loc[0, "x_close"] == pytest.approx(100.14)


def test_prior_quantile_is_strictly_lagged_over_clean_observations() -> None:
    values = pd.Series([1.0, 2.0, 100.0, 4.0, 5.0])
    clean = pd.Series([True, True, False, True, True])
    threshold = support.prior_clean_quantile(
        values, clean, quantile=0.5, window=2, min_periods=2
    )
    assert threshold.iloc[3] == 1.5
    assert threshold.iloc[4] == 3.0


def test_upper_perp_only_wick_routes_short_after_completed_bar() -> None:
    frame = _frame()
    # Two baseline rows have zero excess. Row 2 is a large perp-only upper wick
    # with a non-positive perp body.
    frame.loc[2, "perp_high"] = 101.0
    frame.loc[2, "perp_low"] = 99.9
    frame.loc[2, "perp_close"] = 99.9
    signals, diagnostics = support.classify_signals(frame, _policy())
    assert diagnostics["primary_upper"].iloc[2]
    assert signals["primary"].loc[2, "side"] == -1
    assert signals["primary"].loc[2, "entry_delay_bars"] == 3


def test_lower_perp_only_wick_routes_long() -> None:
    frame = _frame()
    frame.loc[2, "perp_low"] = 99.0
    frame.loc[2, "perp_high"] = 100.1
    frame.loc[2, "perp_close"] = 100.1
    signals, diagnostics = support.classify_signals(frame, _policy())
    assert diagnostics["primary_lower"].iloc[2]
    assert signals["primary"].loc[2, "side"] == 1


def test_feature_prefix_is_invariant_to_future_mutation() -> None:
    frame = _frame(30)
    frame.loc[2, "perp_high"] = 101.0
    frame.loc[2, "perp_low"] = 99.9
    frame.loc[2, "perp_close"] = 99.9
    baseline, _ = support.classify_signals(frame, _policy())
    changed = frame.copy()
    changed.loc[20:, ["perp_high", "spot_high"]] = 1_000.0
    replay, _ = support.classify_signals(changed, _policy())
    pd.testing.assert_frame_equal(baseline["primary"].loc[:19], replay["primary"].loc[:19])


def test_schedule_uses_causal_latency_but_not_future_hold_spot_availability() -> None:
    frame = _frame(25)
    signal = pd.DataFrame(
        {
            "side": [0, 0, -1, *([0] * 22)],
            "branch": ["none", "none", "primary:upper_rejection", *(["none"] * 22)],
            "entry_delay_bars": [0, 0, 3, *([0] * 22)],
            "hold_bars": [0, 0, 12, *([0] * 22)],
        }
    )
    frame.loc[8, "joint_complete"] = False  # Future held path: must not delete trade.
    schedule = support.nonoverlapping_schedule(signal, frame)
    assert schedule.loc[0, "signal_position"] == 2
    assert schedule.loc[0, "entry_position"] == 5
    assert schedule.loc[0, "exit_position"] == 17
    frame.loc[3, "joint_complete"] = False  # Completed latency bucket: skip.
    assert support.nonoverlapping_schedule(signal, frame).empty


def test_schedule_handles_positions_beyond_int16() -> None:
    rows = 40_000
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="5min"),
            "joint_complete": np.ones(rows, dtype=bool),
        }
    )
    signal = pd.DataFrame(
        {
            "side": np.zeros(rows, dtype=np.int8),
            "branch": np.full(rows, "none", dtype=object),
            "entry_delay_bars": np.zeros(rows, dtype=np.int16),
            "hold_bars": np.zeros(rows, dtype=np.int16),
        }
    )
    signal.loc[35_000, ["side", "branch", "entry_delay_bars", "hold_bars"]] = [
        -1,
        "primary:upper_rejection",
        3,
        12,
    ]
    schedule = support.nonoverlapping_schedule(signal, frame)
    assert schedule.loc[0, "entry_position"] == 35_003
    assert schedule.loc[0, "exit_position"] == 35_015


def test_support_gate_accepts_balanced_distributed_clock() -> None:
    dates: list[str] = []
    sides: list[int] = []
    branches: list[str] = []
    for year in range(2020, 2024):
        count = 180 if year < 2023 else 60
        year_dates = pd.date_range(f"{year}-01-01", f"{year}-12-20", periods=count)
        dates.extend(year_dates.astype(str))
        for index in range(count):
            sides.append(1 if index % 2 else -1)
            branches.append(
                "primary:lower_rejection" if index % 2 else "primary:upper_rejection"
            )
    metrics = support._support(
        pd.DataFrame({"entry_date": dates, "side": sides, "branch": branches})
    )
    assert metrics["train_2020_2022"] == 540
    assert metrics["selection_2023"] == 60
    assert metrics["passes_support"] is True
