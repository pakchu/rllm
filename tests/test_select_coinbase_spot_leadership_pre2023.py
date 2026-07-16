from __future__ import annotations

import numpy as np
import pandas as pd

from training import select_coinbase_spot_leadership_pre2023 as selection


def _market(rows: int = 8) -> pd.DataFrame:
    open_price = np.asarray([100, 100, 102, 101, 103, 104, 103, 105], dtype=float)[:rows]
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="5min"),
            "open": open_price,
            "high": open_price + 2,
            "low": open_price - 2,
            "close": open_price,
        }
    )


def test_trade_is_next_open_and_funding_is_strictly_after_entry() -> None:
    market = _market()
    funding = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01 00:05:00", "2020-01-01 00:05:00.002"],
                format="mixed",
            ),
            "funding_rate": [0.5, 0.001],
        }
    )
    engine = selection.ExecutionEngine(market, funding, selection.Config())
    trade = engine.trade_at(signal=0, side=1, hold_bars=1)
    assert trade is not None
    assert trade.entry_position == 1 and trade.exit_position == 2
    assert np.isclose(trade.funding_factor, 1.0 - 0.5 * 0.001)
    assert not np.isclose(trade.funding_factor, (1.0 - 0.5 * 0.5) * (1.0 - 0.5 * 0.001))


def test_strict_mdd_uses_global_hwm_and_hypothetical_liquidation_cost() -> None:
    market = _market()
    funding = pd.DataFrame({"date": pd.to_datetime([]), "funding_rate": []})
    engine = selection.ExecutionEngine(market, funding, selection.Config())
    trades = [engine.trade_at(0, 1, 1), engine.trade_at(3, -1, 1)]
    stats = selection.strict_equity_stats(
        [trade for trade in trades if trade is not None],
        start="2020-01-01",
        end="2021-01-01",
        leverage=0.5,
        cost_notional_per_side=0.0006,
    )
    assert stats["strict_mdd_pct"] > 0.06
    assert stats["trades"] == 2
    assert stats["calendar_end_exclusive"] == "2021-01-01"


def test_exit_open_is_part_of_pre_liquidation_high_water() -> None:
    market = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="5min"),
            "open": [100.0, 100.0, 120.0, 120.0],
            "high": [101.0, 101.0, 121.0, 121.0],
            "low": [99.0, 99.0, 119.0, 119.0],
            "close": [100.0, 100.0, 120.0, 120.0],
        }
    )
    engine = selection.ExecutionEngine(
        market,
        pd.DataFrame({"date": pd.to_datetime([]), "funding_rate": []}),
        selection.Config(),
    )
    trade = engine.trade_at(0, 1, 1)
    assert trade is not None
    assert np.isclose(trade.favorable_price_factor, 1.1)


def test_calendar_cagr_counts_idle_time() -> None:
    stats = selection.strict_equity_stats(
        [],
        start="2020-01-01",
        end="2023-01-01",
        leverage=0.5,
        cost_notional_per_side=0.0006,
    )
    assert stats["absolute_return_pct"] == 0.0
    assert stats["cagr_pct"] == 0.0
    assert stats["trades"] == 0


def test_selection_gates_require_all_years_five_halves_and_adjusted_p() -> None:
    good = {
        "absolute_return_pct": 1.0,
        "cagr_pct": 1.0,
        "strict_mdd_pct": 0.4,
        "cagr_to_strict_mdd": 2.5,
        "trades": 50,
    }
    stats = {name: dict(good) for name in selection.WINDOWS}
    stats["combined_2020_2022"]["trades"] = 150
    gates = selection.selection_gates(
        stats,
        {"absolute_return_pct": 1.0},
        {"bonferroni_p_value": 0.08},
    )
    assert all(gates.values())
    stats["2022_h2"]["absolute_return_pct"] = -1.0
    stats["2022_h1"]["absolute_return_pct"] = -1.0
    assert selection.selection_gates(
        stats, {"absolute_return_pct": 1.0}, {"bonferroni_p_value": 0.08}
    )["positive_half_years_at_least_5_of_6"] is False


def test_role_swap_is_not_a_side_flip_alias() -> None:
    features = pd.DataFrame(
        {
            "ZR": [2.5],
            "ZP": [0.0],
            "ZV": [0.0],
            "ZCB": [1.2],
            "ZBN": [0.0],
            "source_quarantined": [0],
        }
    )
    policy = selection.Policy("X", "relative_return_lead", 1, 1)
    assert selection.transformed_policy_mask(features, policy, "primary")[0]
    assert not selection.transformed_policy_mask(features, policy, "venue_role_swap")[0]


def test_vectorized_random_return_matches_trade_engine() -> None:
    market = _market()
    funding = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01 00:05:00.002"]),
            "funding_rate": [0.001],
        }
    )
    cfg = selection.Config()
    engine = selection.ExecutionEngine(market, funding, cfg)
    policy = selection.Policy("X", "relative_return_lead", 1, 1)
    vectorized = selection.precompute_signal_log_factors(engine, policy, cfg)
    trade = engine.trade_at(0, 1, 1)
    assert trade is not None
    cost_log = 2.0 * np.log(1.0 - cfg.leverage * cfg.base_cost_notional_per_side)
    expected = np.log(trade.price_factor * trade.funding_factor) + cost_log
    assert np.isclose(vectorized[0], expected)
