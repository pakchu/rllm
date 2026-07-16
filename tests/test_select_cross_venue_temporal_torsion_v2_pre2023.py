from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import select_cross_venue_temporal_torsion_v2_pre2023 as selector
from training.preregister_cross_venue_temporal_torsion_alpha_v2 import Policy


def _engine(
    opens: list[float], highs: list[float], lows: list[float], funding=None
) -> selector.ExecutionEngine:
    dates = pd.date_range("2020-01-01", periods=len(opens), freq="5min")
    market = pd.DataFrame(
        {"date": dates, "open": opens, "high": highs, "low": lows, "close": opens}
    )
    if funding is None:
        funding = pd.DataFrame(
            {"date": pd.to_datetime([]), "funding_rate": pd.Series(dtype=float)}
        )
    return selector.ExecutionEngine(market, funding, selector.Config())


def test_execution_enters_at_t_plus_ten_minutes_and_includes_exit_open() -> None:
    engine = _engine(
        [100, 101, 102, 103, 104, 105],
        [100, 102, 103, 104, 105, 106],
        [100, 100, 98, 102, 103, 104],
    )
    trade = engine.trade_at(signal=0, side=1, hold_bars=2)
    assert trade is not None
    assert trade.entry_position == 2
    assert trade.exit_position == 4
    assert trade.entry_date == pd.Timestamp("2020-01-01 00:10")
    assert trade.price_factor == 1.0 + 0.5 * (104 / 102 - 1.0)
    assert trade.adverse_price_factor == 1.0 + 0.5 * (98 / 102 - 1.0)


def test_funding_is_applied_only_after_entry_through_exit() -> None:
    dates = pd.to_datetime(
        ["2020-01-01 00:10", "2020-01-01 00:15", "2020-01-01 00:20"]
    )
    funding = pd.DataFrame({"date": dates, "funding_rate": [0.1, 0.02, 0.03]})
    engine = _engine([100] * 7, [100] * 7, [100] * 7, funding)
    trade = engine.trade_at(signal=0, side=1, hold_bars=2)
    assert trade is not None
    assert np.isclose(trade.funding_factor, (1 - 0.5 * 0.02) * (1 - 0.5 * 0.03))


def test_strict_mdd_uses_favorable_before_adverse_and_both_costs() -> None:
    trade = selector.Trade(
        signal_position=0,
        entry_position=2,
        exit_position=4,
        entry_date=pd.Timestamp("2020-01-01 00:10"),
        exit_date=pd.Timestamp("2020-01-01 00:20"),
        side=1,
        price_factor=1.0,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        funding_credit_factor=1.0,
        favorable_price_factor=1.10,
        adverse_price_factor=0.90,
    )
    stats = selector.strict_equity_stats(
        [trade],
        start="2020-01-01",
        end="2021-01-01",
        leverage=0.5,
        cost_notional_per_side=0.0006,
    )
    cost = 1 - 0.5 * 0.0006
    expected = 1 - (cost * 0.90 * cost) / (cost * 1.10)
    assert np.isclose(stats["strict_mdd_pct"], expected * 100)
    assert np.isclose(stats["absolute_return_pct"], (cost * cost - 1) * 100)


def test_cagr_uses_full_calendar_when_trades_are_sparse() -> None:
    stats = selector.strict_equity_stats(
        [],
        start="2020-01-01",
        end="2023-01-01",
        leverage=0.5,
        cost_notional_per_side=0.0006,
    )
    assert stats["absolute_return_pct"] == 0.0
    assert stats["cagr_pct"] == 0.0
    assert stats["trades"] == 0


def test_dynamic_long_short_clock_is_preserved() -> None:
    engine = _engine([100, 100, 100, 101, 99, 100, 100], [101] * 7, [98] * 7)
    trades = selector.trades_from_arrays(
        engine, [0, 2], [1, -1], hold_bars=1
    )
    assert [trade.side for trade in trades] == [1, -1]
    assert [trade.entry_position for trade in trades] == [2, 4]


def test_vectorized_log_factor_matches_execution_engine() -> None:
    engine = _engine(
        [100, 100, 101, 102, 100, 99, 100, 101],
        [102] * 8,
        [98] * 8,
    )
    cfg = selector.Config()
    signals = np.asarray([0, 3])
    sides = np.asarray([1, -1])
    vector = selector.vectorized_log_factors(engine, signals, sides, 1, cfg)
    trades = selector.trades_from_arrays(engine, signals, sides, hold_bars=1)
    cost_log = 2 * np.log(1 - cfg.leverage * cfg.base_cost_notional_per_side)
    expected = np.asarray(
        [np.log(t.price_factor * t.funding_factor) + cost_log for t in trades]
    )
    assert np.allclose(vector, expected)


def test_selection_gates_require_all_years_and_familywise_significance() -> None:
    good = {
        "absolute_return_pct": 5.0,
        "cagr_pct": 5.0,
        "strict_mdd_pct": 2.0,
        "cagr_to_strict_mdd": 2.5,
        "trades": 200,
    }
    stats = {name: dict(good) for name in selector.WINDOWS}
    stats["combined_2020_2022"]["trades"] = 700
    gates = selector.selection_gates(
        stats,
        {"absolute_return_pct": 1.0},
        {"bonferroni_p_value": 0.05},
    )
    assert all(gates.values())
    stats["fit_2021"]["absolute_return_pct"] = -0.1
    assert not selector.selection_gates(
        stats,
        {"absolute_return_pct": 1.0},
        {"bonferroni_p_value": 0.05},
    )["every_calendar_year_absolute_return_positive"]


def test_generated_control_clock_uses_source_route_and_nonoverlap() -> None:
    features = pd.DataFrame(
        {
            "source_quarantined": [0] * 10,
            "spot_source_side": [1] * 10,
            "um_source_side": [1] * 10,
            "spot_direction_confirmed": [1] * 10,
            "um_direction_confirmed": [1] * 10,
            "spot_flow_to_return_delay": [0.2] * 10,
            "um_flow_to_return_delay": [-0.2] * 10,
        }
    )
    indices, sides = selector.generated_control_clock(
        features, Policy("V", "spot_preload_um_echo", 3), "same_venue_preload_only"
    )
    assert indices.tolist() == [0]
    assert sides.tolist() == [1]


def test_overlapping_positions_fail_strict_stats() -> None:
    base = dict(
        signal_position=0,
        entry_date=pd.Timestamp("2020-01-01"),
        exit_date=pd.Timestamp("2020-01-01 00:10"),
        side=1,
        price_factor=1.0,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        funding_credit_factor=1.0,
        favorable_price_factor=1.0,
        adverse_price_factor=1.0,
    )
    first = selector.Trade(entry_position=2, exit_position=5, **base)
    second = selector.Trade(entry_position=4, exit_position=6, **base)
    with pytest.raises(RuntimeError, match="overlapping"):
        selector.strict_equity_stats(
            [first, second],
            start="2020-01-01",
            end="2021-01-01",
            leverage=0.5,
            cost_notional_per_side=0.0006,
        )
