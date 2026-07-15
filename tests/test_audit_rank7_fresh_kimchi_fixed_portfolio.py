from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.audit_rank7_fresh_kimchi_fixed_portfolio import (
    EXPECTED_PORTFOLIO_SPEC_HASH,
    PORTFOLIO_SPEC,
    portfolio_spec_hash,
    subaccount_bar_path,
    synchronized_portfolio_stats,
)
from training.search_inventory_purge_reclaim_alpha import Config, ExecutionEngine, Trade


def execution_config(*, cost: float = 0.0) -> Config:
    return Config(
        input_csv="",
        metrics_csv="",
        funding_csv="",
        output="",
        manifest_output="",
        leverage=0.5,
        fee_rate=cost,
        slippage_rate=0.0,
    )


def market(*, high: float = 101.0, low: float = 99.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="5min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 6,
            "high": [high] * 6,
            "low": [low] * 6,
            "close": [100.0] * 6,
        }
    )


def no_funding() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime([]), "funding_rate": pd.Series(dtype=float)})


def trade(*, side: int, exit_position: int, gross_return: float, price_factor: float) -> Trade:
    return Trade(
        signal_position=0,
        entry_position=1,
        exit_position=exit_position,
        side=side,
        gross_return=gross_return,
        price_factor=price_factor,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=1.01,
        adverse_price_factor=0.99,
        entry_date="2024-01-01 00:05:00",
    )


def test_subaccount_cap_path_preserves_exact_costed_final_equity() -> None:
    cfg = execution_config(cost=0.001)
    cap_trade = trade(side=1, exit_position=3, gross_return=0.0, price_factor=1.0)
    path = subaccount_bar_path(
        market(),
        no_funding(),
        [cap_trade],
        cfg,
        start="2024-01-01",
        end="2024-01-01 00:30:00",
        hold_bars=lambda _trade: 2,
    )
    expected = (1.0 - 0.5 * 0.001) ** 2
    assert path.final_equity == pytest.approx(expected)
    assert path.close_value[-1] == pytest.approx(expected)
    assert path.active[3] == np.bool_(False)


def test_stop_exit_caps_adverse_market_extreme_at_realized_stop() -> None:
    cfg = execution_config()
    stopped = trade(side=1, exit_position=1, gross_return=-0.025, price_factor=0.9875)
    path = subaccount_bar_path(
        market(high=105.0, low=80.0),
        no_funding(),
        [stopped],
        cfg,
        start="2024-01-01",
        end="2024-01-01 00:30:00",
        hold_bars=lambda _trade: 10,
    )
    assert path.market_low_value[1] == pytest.approx(0.9875)
    assert path.market_high_value[1] == pytest.approx(1.025)


def test_same_bar_stop_before_take_matches_execution_engine() -> None:
    cfg = execution_config()
    bars = market(high=105.0, low=95.0)
    engine = ExecutionEngine(bars, no_funding(), cfg)
    stopped = engine.trade_at(0, 1, 2, 400, 250)
    assert stopped is not None
    assert stopped.exit_position == 1
    assert stopped.gross_return == pytest.approx(-0.025)
    path = subaccount_bar_path(
        bars,
        no_funding(),
        [stopped],
        cfg,
        start="2024-01-01",
        end="2024-01-01 00:30:00",
        hold_bars=lambda _trade: 2,
    )
    assert path.market_low_value[1] == pytest.approx(stopped.price_factor)


def test_same_price_portfolio_offsets_simultaneous_long_and_short() -> None:
    cfg = execution_config()
    long_trade = trade(side=1, exit_position=3, gross_return=0.0, price_factor=1.0)
    short_trade = trade(side=-1, exit_position=3, gross_return=0.0, price_factor=1.0)
    kwargs = {
        "market": market(high=110.0, low=90.0),
        "funding": no_funding(),
        "cfg": cfg,
        "start": "2024-01-01",
        "end": "2024-01-01 00:30:00",
        "hold_bars": lambda _trade: 2,
    }
    long_path = subaccount_bar_path(trades=[long_trade], **kwargs)
    short_path = subaccount_bar_path(trades=[short_trade], **kwargs)
    stats = synchronized_portfolio_stats(
        {"long": long_path, "short": short_path},
        {"long": 0.5, "short": 0.5},
        start=kwargs["start"],
        end=kwargs["end"],
        trade_counts={"long": 1, "short": 1},
    )
    assert stats["absolute_return_pct"] == pytest.approx(0.0)
    assert stats["synchronized_strict_mdd_pct"] == pytest.approx(0.0)


def test_fixed_subaccount_final_equity_is_weighted_not_rebalanced() -> None:
    cfg = execution_config()
    winner = trade(side=1, exit_position=3, gross_return=0.1, price_factor=1.05)
    flat = trade(side=-1, exit_position=3, gross_return=0.0, price_factor=1.0)
    kwargs = {
        "market": market(),
        "funding": no_funding(),
        "cfg": cfg,
        "start": "2024-01-01",
        "end": "2024-01-01 00:30:00",
        "hold_bars": lambda _trade: 2,
    }
    winner_path = subaccount_bar_path(trades=[winner], **kwargs)
    flat_path = subaccount_bar_path(trades=[flat], **kwargs)
    stats = synchronized_portfolio_stats(
        {"winner": winner_path, "flat": flat_path},
        {"winner": 0.75, "flat": 0.25},
        start=kwargs["start"],
        end=kwargs["end"],
        trade_counts={"winner": 1, "flat": 1},
    )
    assert stats["final_equity"] == pytest.approx(0.75 * 1.05 + 0.25)


def test_portfolio_rejects_weights_that_do_not_sum_to_one() -> None:
    cfg = execution_config()
    path = subaccount_bar_path(
        market(),
        no_funding(),
        [],
        cfg,
        start="2024-01-01",
        end="2024-01-01 00:30:00",
        hold_bars=lambda _trade: 2,
    )
    with pytest.raises(ValueError, match="sum to one"):
        synchronized_portfolio_stats(
            {"cash": path},
            {"cash": 0.9},
            start="2024-01-01",
            end="2024-01-01 00:30:00",
            trade_counts={"cash": 0},
        )


def test_funding_at_entry_midpoint_and_exit_matches_execution_convention() -> None:
    cfg = execution_config()
    bars = market()
    rates = np.asarray([0.001, 0.002, 0.003])
    funding = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01 00:05:00", "2024-01-01 00:10:00", "2024-01-01 00:15:00"]
            ),
            "funding_rate": rates,
        }
    )
    expected_funding = float(np.prod(1.0 - 0.5 * rates))
    funded = Trade(
        **{
            **trade(side=1, exit_position=3, gross_return=0.0, price_factor=1.0).__dict__,
            "funding_factor": expected_funding,
            "funding_debit_factor": expected_funding,
        }
    )
    path = subaccount_bar_path(
        bars,
        funding,
        [funded],
        cfg,
        start="2024-01-01",
        end="2024-01-01 00:30:00",
        hold_bars=lambda _trade: 2,
    )
    assert path.final_equity == pytest.approx(expected_funding)


def test_split_crossing_trade_is_rejected_not_clipped() -> None:
    cfg = execution_config()
    crossing = trade(side=1, exit_position=5, gross_return=0.0, price_factor=1.0)
    with pytest.raises(ValueError, match="contained inside"):
        subaccount_bar_path(
            market(),
            no_funding(),
            [crossing],
            cfg,
            start="2024-01-01",
            end="2024-01-01 00:20:00",
            hold_bars=lambda _trade: 4,
        )


def test_portfolio_spec_is_single_cell_and_hash_pinned() -> None:
    assert PORTFOLIO_SPEC["weights"] == {
        "frozen_annual_rank7": 0.75,
        "fresh_kimchi_fx": 0.25,
    }
    assert PORTFOLIO_SPEC["weight_grid_cells"] == 1
    assert portfolio_spec_hash() == EXPECTED_PORTFOLIO_SPEC_HASH


def test_strict_mdd_carries_prior_realized_peak_into_later_trade() -> None:
    cfg = execution_config()
    first = trade(side=1, exit_position=3, gross_return=0.1, price_factor=1.05)
    second = Trade(
        signal_position=4,
        entry_position=5,
        exit_position=5,
        side=1,
        gross_return=-0.10,
        price_factor=0.95,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=1.0,
        adverse_price_factor=0.95,
        entry_date="2024-01-01 00:25:00",
    )
    path = subaccount_bar_path(
        market(),
        no_funding(),
        [first, second],
        cfg,
        start="2024-01-01",
        end="2024-01-01 00:30:00",
        hold_bars=lambda row: 2 if row.signal_position == 0 else 10,
    )
    stats = synchronized_portfolio_stats(
        {"sleeve": path},
        {"sleeve": 1.0},
        start="2024-01-01",
        end="2024-01-01 00:30:00",
        trade_counts={"sleeve": 2},
    )
    assert stats["synchronized_strict_mdd_pct"] >= 5.0 - 1e-10
