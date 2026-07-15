from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.audit_fresh_kimchi_orthogonal_alpha import (
    CANDIDATE_SPEC,
    daily_marked_returns,
    fresh_candidate_masks,
    pnl_correlation,
    trade_timing_overlap,
)
from training.search_inventory_purge_reclaim_alpha import Config, Trade


def _trade(entry: int, exit_: int, side: int = 1, price_factor: float = 1.01) -> Trade:
    return Trade(
        signal_position=max(0, entry - 1),
        entry_position=entry,
        exit_position=exit_,
        side=side,
        gross_return=(price_factor - 1.0) / 0.5,
        price_factor=price_factor,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=max(1.0, price_factor),
        adverse_price_factor=min(1.0, price_factor),
        entry_date=str(pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * entry)),
    )


def test_fresh_candidate_masks_fail_closed_on_unavailable_sources() -> None:
    n = 4
    market = pd.DataFrame(
        {
            "funding_available": [1, 0, 1, 1],
            "usdkrw_available": [1, 1, 0, 1],
            "kimchi_available": [1, 1, 1, 0],
        }
    )
    features = pd.DataFrame(
        {
            "funding_rate": [-1.0] * n,
            "bd_flow_accel": [1.0] * n,
            "kl_local_impulse_144": [-9.0] * n,
            "usdkrw_momentum": [1.0] * n,
            "htf_1d_return_1": [-1.0] * n,
        }
    )
    long_active, short_active, diagnostics = fresh_candidate_masks(market, features)
    assert np.flatnonzero(long_active).tolist() == [0]
    assert np.flatnonzero(short_active).tolist() == [0, 1, 3]
    assert diagnostics["blocked_stale_long_rows"] == 3
    assert diagnostics["blocked_stale_short_rows"] == 1
    assert diagnostics["fresh_long_availability_violations"] == 0
    assert diagnostics["fresh_short_availability_violations"] == 0
    assert CANDIDATE_SPEC["availability"]["long"] == [
        "funding_available",
        "usdkrw_available",
        "kimchi_available",
    ]


def test_trade_timing_overlap_uses_near_entries_and_position_time() -> None:
    candidate = [_trade(10, 20), _trade(40, 50)]
    baseline = [_trade(12, 22), _trade(70, 80)]
    result = trade_timing_overlap(candidate, baseline, total_bars=100, near_bars=3)
    assert result["exact_entry_jaccard"] == 0.0
    assert result["candidate_entries_near_6h_fraction"] == 0.5
    assert result["baseline_entries_near_6h_fraction"] == 0.5
    assert result["candidate_position_overlap_fraction"] == pytest.approx(8 / 20)
    assert result["baseline_position_overlap_fraction"] == pytest.approx(8 / 20)
    assert result["position_jaccard"] == pytest.approx(8 / 32)


def test_daily_marked_returns_preserve_exact_trade_factor() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="12h")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 101.0, 102.0, 101.0, 103.0],
        }
    )
    funding = pd.DataFrame({"date": pd.to_datetime([]), "funding_rate": pd.Series(dtype=float)})
    leverage = 0.5
    price_factor = 1.0 + leverage * (101.0 / 101.0 - 1.0)
    trade = _trade(1, 3, price_factor=price_factor)
    cfg = Config(
        input_csv="",
        metrics_csv="",
        funding_csv="",
        output="",
        manifest_output="",
        leverage=leverage,
        fee_rate=0.0005,
        slippage_rate=0.0001,
    )
    daily = daily_marked_returns(
        market,
        funding,
        [trade],
        cfg,
        start="2024-01-01",
        end="2024-01-04",
    )
    expected = (1.0 - leverage * 0.0006) ** 2 * price_factor
    assert float((daily + 1.0).prod()) == pytest.approx(expected)


def test_pnl_correlation_zero_fills_and_reports_both_coefficients() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    left = pd.Series([0.01, 0.0, -0.01, 0.0], index=index)
    right = pd.Series([0.0, 0.01, 0.0, -0.01], index=index)
    result = pnl_correlation(left, right)
    assert result["pearson"] == pytest.approx(0.0)
    assert result["spearman"] == pytest.approx(0.0)
