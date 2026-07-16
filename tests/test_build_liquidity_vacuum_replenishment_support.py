from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import build_liquidity_vacuum_replenishment_support as support
from training import preregister_liquidity_vacuum_replenishment as prereg


def _policy(**kwargs: object) -> prereg.Policy:
    return replace(
        prereg.Policy(),
        baseline_bars=2,
        baseline_min_periods=2,
        setup_quantile=0.5,
        minimum_agg_trade_count=1,
        hold_bars=2,
        **kwargs,
    )


def _frame(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "agg_trade_count": np.full(rows, 100),
            "signed_quote_notional": np.full(rows, 1.0),
            "signed_price_response": np.zeros(rows),
            "interarrival_burstiness": np.full(rows, 1.0),
            "event_notional_hhi": np.full(rows, 1.0),
            "micro_log_return": np.zeros(rows),
            "quarantined": np.zeros(rows, dtype=bool),
        }
    )


def test_prior_clean_quantile_uses_only_strictly_prior_clean_observations() -> None:
    values = pd.Series([1.0, 2.0, 100.0, 4.0, 5.0])
    clean = pd.Series([True, True, False, True, True])
    threshold = support.prior_clean_quantile(
        values, clean, quantile=0.5, window=2, min_periods=2
    )
    assert np.isnan(threshold.iloc[0])
    assert np.isnan(threshold.iloc[1])
    assert np.isnan(threshold.iloc[2])
    assert threshold.iloc[3] == 1.5
    assert threshold.iloc[4] == 3.0


def test_primary_requires_separate_reversal_bar_and_routes_opposite_setup() -> None:
    frame = _frame()
    frame.loc[2, ["interarrival_burstiness", "event_notional_hhi"]] = 2.0
    frame.loc[2, "signed_price_response"] = 0.01
    frame.loc[2, "signed_quote_notional"] = 10.0
    frame.loc[3, "signed_quote_notional"] = -10.0
    frame.loc[3, "micro_log_return"] = -0.01
    signals, diagnostics = support.classify_signals(frame, _policy())
    assert diagnostics["setup"].iloc[2]
    assert diagnostics["confirmation"].iloc[3]
    assert signals["primary"].loc[3, "side"] == -1
    assert signals["primary"].loc[2, "side"] == 0


def test_feature_prefix_is_invariant_to_future_mutation() -> None:
    frame = _frame(16)
    frame.loc[2, ["interarrival_burstiness", "event_notional_hhi"]] = 2.0
    frame.loc[2, "signed_price_response"] = 0.01
    frame.loc[3, "signed_quote_notional"] = -10.0
    frame.loc[3, "micro_log_return"] = -0.01
    baseline, _ = support.classify_signals(frame, _policy())
    changed = frame.copy()
    changed.loc[10:, "signed_quote_notional"] = -1e12
    changed.loc[10:, "interarrival_burstiness"] = 1e12
    replay, _ = support.classify_signals(changed, _policy())
    pd.testing.assert_frame_equal(baseline["primary"].loc[:9], replay["primary"].loc[:9])


def test_schedule_uses_next_open_fixed_hold_and_full_clean_path() -> None:
    frame = _frame(10)
    signal = pd.DataFrame(
        {
            "date": frame["date"],
            "side": [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            "branch": ["none", "none", "none", "lvrt_r0", *(["none"] * 6)],
            "hold_bars": [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
            "lookback_bars": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        }
    )
    schedule = support.nonoverlapping_schedule(signal, frame)
    assert schedule.loc[0, "setup_position"] == 2
    assert schedule.loc[0, "signal_position"] == 3
    assert schedule.loc[0, "entry_position"] == 4
    assert schedule.loc[0, "exit_position"] == 6
    frame.loc[5, "quarantined"] = True
    assert support.nonoverlapping_schedule(signal, frame).empty


def test_schedule_rejects_split_crossing_trade() -> None:
    frame = _frame(10)
    signal = pd.DataFrame(
        {
            "date": frame["date"],
            "side": [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            "branch": ["none", "none", "none", "lvrt_r0", *(["none"] * 6)],
            "hold_bars": [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
            "lookback_bars": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        }
    )
    schedule = support.nonoverlapping_schedule(
        signal,
        frame,
        start="2023-01-01",
        end="2023-01-01 00:30",
    )
    assert schedule.empty


def test_schedule_handles_positions_beyond_int16() -> None:
    rows = 40_000
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="5min"),
            "quarantined": np.zeros(rows, dtype=bool),
        }
    )
    signal = pd.DataFrame(
        {
            "side": np.zeros(rows, dtype=np.int8),
            "branch": np.full(rows, "none", dtype=object),
            "hold_bars": np.zeros(rows, dtype=np.int16),
            "lookback_bars": np.zeros(rows, dtype=np.int16),
        }
    )
    signal.loc[35_000, ["side", "branch", "hold_bars", "lookback_bars"]] = [
        -1,
        "lvrt_r0",
        12,
        1,
    ]
    schedule = support.nonoverlapping_schedule(signal, frame)
    assert schedule.loc[0, "signal_position"] == 35_000
    assert schedule.loc[0, "exit_position"] == 35_013


def test_quarantine_extends_forward_without_backfill() -> None:
    available = pd.Series([True, True, False, True, True, True])
    gap = pd.Series([False, False, False, False, False, True])
    result = support.quarantine_mask(available, gap, post_gap_bars=2)
    assert result.tolist() == [False, False, True, True, True, True]


def test_support_gate_checks_counts_balance_and_month_concentration() -> None:
    dates: list[str] = []
    sides: list[int] = []
    for year in range(2020, 2024):
        for month in range(1, 13):
            for day in (1, 8, 15, 22, 28):
                dates.append(f"{year}-{month:02d}-{day:02d}")
                sides.append(1 if len(sides) % 2 == 0 else -1)
    schedule = pd.DataFrame({"entry_date": dates, "side": sides})
    metrics = support._support(schedule)
    assert metrics["nonoverlap_total"] == 240
    assert metrics["passes_support"] is False
    extra = schedule.iloc[:20].copy()
    extra["entry_date"] = pd.date_range("2020-02-02", periods=20, freq="D").astype(str)
    metrics = support._support(pd.concat([schedule, extra], ignore_index=True))
    assert metrics["nonoverlap_total"] == 260
    assert metrics["passes_support"] is True


def test_frozen_support_rejects_policy_mutation() -> None:
    with pytest.raises(ValueError, match="policy is frozen"):
        support.run_support(replace(prereg.Policy(), hold_bars=13))
