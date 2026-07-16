from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.search_delta_neutral_funding_carry_alpha import (
    Config,
    CostModel,
    Policy,
    Sources,
    gate_actions,
    map_funding_to_market,
    reconstruct_spot,
    simulate_window,
)


def market_frame(
    start: str = "2023-01-01 01:00:00",
    periods: int = 4,
    *,
    spot_high: float = 100.0,
    spot_low: float = 100.0,
    perp_high: float = 100.0,
    perp_low: float = 100.0,
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="5min")
    return pd.DataFrame(
        {
            "date": dates,
            "perp_open": 100.0,
            "perp_high": perp_high,
            "perp_low": perp_low,
            "perp_close": 100.0,
            "spot_open": 100.0,
            "spot_high": spot_high,
            "spot_low": spot_low,
            "spot_close": 100.0,
            "spot_proxy": False,
            "spot_observations": 5,
        }
    )


def source_bundle(market: pd.DataFrame, funding: pd.DataFrame | None = None) -> Sources:
    if funding is None:
        funding = pd.DataFrame(
            columns=[
                "date",
                "funding_rate",
                "reported_mark_price",
                "exec_index",
                "settlement_mark",
                "fallback_index",
            ]
        )
    return Sources(market=market, funding=funding, source_hashes={}, diagnostics={})


def test_funding_mapping_is_strictly_after_event_and_mark_is_completed_before_event() -> None:
    market = market_frame("2023-01-01 07:58:00", periods=8)
    market["date"] = pd.date_range("2023-01-01 07:58:00", periods=8, freq="1min")
    market["perp_close"] = [99.0, 100.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0]
    funding = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-01 08:00:00")],
            "funding_rate": [0.001],
            "reported_mark_price": [np.nan],
        }
    )
    mapped, diagnostics = map_funding_to_market(market, funding)
    assert mapped.loc[0, "exec_index"] == 7  # 08:05, not same-timestamp 08:00
    assert mapped.loc[0, "fallback_index"] == 1  # 07:59 bar ends at 08:00
    assert mapped.loc[0, "settlement_mark"] == 100.0
    assert diagnostics["funding_missing_reported_mark"] == 1


def test_event_triggered_entry_does_not_capture_that_event_but_receives_next() -> None:
    market = market_frame("2023-01-01 00:00:00", periods=7)
    funding = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 00:10:00"]),
            "funding_rate": [0.001, 0.001],
            "reported_mark_price": [np.nan, np.nan],
            "exec_index": [1, 3],
            "settlement_mark": [100.0, 100.0],
            "fallback_index": [-1, 1],
        }
    )
    result = simulate_window(
        source_bundle(market, funding),
        {1: True},
        start="2023-01-01 00:00:00",
        end="2023-01-01 00:35:00",
        cfg=Config(),
        costs=CostModel(0.0, 0.0),
    )["stats"]
    assert result["funding_events_received"] == 1
    assert result["funding_cash_pct_initial"] == pytest.approx(0.05)
    assert result["absolute_return_pct"] == pytest.approx(0.05)


def test_two_leg_entry_and_exit_costs_are_charged() -> None:
    result = simulate_window(
        source_bundle(market_frame(periods=2)),
        {},
        start="2023-01-01 01:00:00",
        end="2023-01-01 01:10:00",
        cfg=Config(),
        costs=CostModel(0.001, 0.002),
        force_initial_active=True,
    )["stats"]
    assert result["transaction_cost_pct_initial"] == pytest.approx(0.3)
    assert result["absolute_return_pct"] == pytest.approx(-0.3)
    assert result["gross_turnover_x_initial"] == pytest.approx(2.0)


def test_strict_mdd_combines_favorable_then_adverse_two_leg_extremes() -> None:
    market = market_frame(
        periods=2,
        spot_high=105.0,
        spot_low=95.0,
        perp_high=110.0,
        perp_low=90.0,
    )
    result = simulate_window(
        source_bundle(market),
        {},
        start="2023-01-01 01:00:00",
        end="2023-01-01 01:10:00",
        cfg=Config(),
        costs=CostModel(0.0, 0.0),
        force_initial_active=True,
    )["stats"]
    assert result["absolute_return_pct"] == pytest.approx(0.0)
    assert result["close_mdd_pct"] == pytest.approx(0.0)
    assert result["strict_mdd_pct"] == pytest.approx((1.0 - 0.975 / 1.025) * 100.0)


def test_missing_spot_repair_uses_only_prior_complete_basis_and_widens_extrema() -> None:
    dates = pd.date_range("2023-01-01", periods=3, freq="5min")
    futures = pd.DataFrame(
        {"date": dates, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
    )
    spot = pd.DataFrame(
        {
            "date": [dates[0], dates[2]],
            "open": [102.0, 104.0],
            "high": [103.0, 105.0],
            "low": [101.0, 103.0],
            "close": [102.0, 104.0],
        }
    )
    repaired, diagnostics = reconstruct_spot(futures, spot, cushion=0.0025)
    # Prior complete basis is +2%; no future spot observation can repair the missing minute.
    assert repaired.loc[1, "spot_open"] == pytest.approx(102.0)
    assert repaired.loc[1, "spot_close"] == pytest.approx(102.0)
    assert repaired.loc[1, "spot_high"] >= 101.0 * (1.02 + 0.0025)
    assert repaired.loc[1, "spot_low"] <= 99.0 * (1.02 - 0.0025)
    assert diagnostics["missing_spot_bars"] == 1
    assert diagnostics["proxy_spot_bars"] == 1


def test_gate_uses_current_settled_event_then_executes_at_precomputed_next_open() -> None:
    funding = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=4, freq="8h"),
            "funding_rate": [0.0001] * 4,
            "exec_index": [1, 2, 3, 4],
        }
    )
    actions, trace = gate_actions(
        funding,
        Policy(lookback_events=3, entry_threshold=0.0001, exit_threshold=0.0, min_hold_events=3),
    )
    assert actions == {3: True}
    assert trace[0]["funding_position"] == 2
    assert trace[0]["execution_index"] == 3
