from __future__ import annotations

import numpy as np
import pandas as pd

from training.audit_confirmed_pullback_squeeze_live_parity import (
    _activation_hash,
    decision_mask,
    live_decision_features,
    schedule_window,
    selection_passes,
)
from training.search_inventory_purge_reclaim_alpha import Config, ExecutionEngine


def test_live_decision_clock_is_hour_open_and_legacy_clock_is_55_minutes() -> None:
    dates = pd.Series(pd.date_range("2020-01-01 15:00:00", periods=400, freq="5min"))

    live = decision_mask(dates, "live_hour_signal_bar")
    legacy = decision_mask(dates, "legacy_positional", window_size=144)

    assert live.any()
    assert legacy.any()
    assert set(dates[live].dt.minute) == {0}
    assert set(dates[legacy].dt.minute) == {55}
    assert not np.array_equal(live, legacy)


def test_live_features_delay_market_state_but_keep_boundary_auxiliary_values() -> None:
    features = pd.DataFrame(
        {
            "trend_96": [1.0, 2.0, 3.0],
            "quote_vol_z_1d": [10.0, 20.0, 30.0],
            "funding_rate": [0.1, 0.2, 0.3],
            "funding_available": [1.0, 1.0, 1.0],
            "premium_index_change": [-0.1, -0.2, -0.3],
            "premium_available": [1.0, 1.0, 1.0],
        }
    )

    live = live_decision_features(features)

    assert np.isnan(live.loc[0, "trend_96"])
    assert live.loc[1, "trend_96"] == 1.0
    assert live.loc[1, "quote_vol_z_1d"] == 10.0
    assert live.loc[1, "funding_rate"] == 0.2
    assert live.loc[1, "premium_index_change"] == -0.2


def test_activation_hash_binds_length_and_dates() -> None:
    active = np.array([True, False, True])
    dates = pd.Series(pd.date_range("2023-01-01", periods=3, freq="5min"))

    original = _activation_hash(active, dates)

    assert _activation_hash(active, dates + pd.Timedelta(minutes=5)) != original
    extended_dates = pd.concat(
        [dates, pd.Series([dates.iloc[-1] + pd.Timedelta(minutes=5)])],
        ignore_index=True,
    )
    assert _activation_hash(np.r_[active, False], extended_dates) != original


def _selection_stats(ratio: float = 3.1) -> dict[str, dict[str, float | int]]:
    def row(*, trades: int = 100, absolute: float = 1.0) -> dict[str, float | int]:
        return {
            "trades": trades,
            "absolute_return_pct": absolute,
            "cagr_to_strict_mdd": ratio,
            "strict_mdd_pct": 10.0,
        }

    return {
        "train": row(trades=100),
        "train_2020h2": row(),
        "train_2021": row(),
        "train_2022": row(),
        "select_2023": row(trades=20),
        "select_2023_h1": row(trades=10),
        "select_2023_h2": row(trades=10),
        "pre_2024": row(trades=120),
    }


def test_selection_contract_cannot_be_rescued_by_only_one_window() -> None:
    assert selection_passes(_selection_stats())

    weak_train = _selection_stats(ratio=2.99)
    assert not selection_passes(weak_train)

    unstable = _selection_stats()
    unstable["select_2023_h2"]["absolute_return_pct"] = -0.01
    assert not selection_passes(unstable)

    sparse = _selection_stats()
    sparse["select_2023_h1"]["trades"] = 4
    assert not selection_passes(sparse)

    excessive_drawdown = _selection_stats()
    excessive_drawdown["pre_2024"]["strict_mdd_pct"] = 15.01
    assert not selection_passes(excessive_drawdown)


def test_schedule_enters_next_open_uses_stop_first_and_purges_boundary_exit() -> None:
    rows = 30
    dates = pd.date_range("2023-01-01", periods=rows, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 100.0),
            "low": np.full(rows, 100.0),
        }
    )
    # Signal 2 enters 3. Both stop and take touch; conservative ordering stops.
    market.loc[3, "high"] = 106.0
    market.loc[3, "low"] = 96.0
    funding = pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[ns]"),
            "funding_rate": pd.Series([], dtype=float),
        }
    )
    cfg = Config(
        input_csv="market.csv",
        metrics_csv="",
        funding_csv="funding.csv",
        output="out.json",
        manifest_output="",
    )
    engine = ExecutionEngine(market, funding, cfg)
    active = np.zeros(rows, dtype=bool)
    active[[2, 4, 27]] = True

    trades = schedule_window(
        engine,
        active,
        start=dates[0],
        end=dates[-1],
        hold_bars=5,
        take_bps=500,
        stop_bps=300,
    )

    assert len(trades) == 2
    assert trades[0].signal_position == 2
    assert trades[0].entry_position == 3
    assert trades[0].exit_position == 3
    assert trades[0].gross_return == -0.03
    # The boundary-crossing signal at 27 is purged.
    assert all(trade.signal_position != 27 for trade in trades)
