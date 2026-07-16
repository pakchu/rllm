from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import build_post_funding_cross_sectional_crowding_release_support as pfcr


def synthetic_sources(events: int = 110) -> tuple[pd.DataFrame, pd.DataFrame]:
    settlements = pd.date_range("2023-03-01", periods=events, freq="8h")
    funding = pd.DataFrame(index=settlements, columns=pfcr.SYMBOLS, dtype=float)
    for index, symbol in enumerate(pfcr.SYMBOLS):
        funding[symbol] = 0.0001 * np.sin(np.arange(events) / 4.0 + index)
    funding.iloc[-1, :] = [0.0008, -0.0006, 0.0002, 0.0001, 0.0, -0.0001]
    beta_index = pd.date_range(settlements.min(), settlements.max(), freq="1h")
    beta = pd.DataFrame(1.0, index=beta_index, columns=pfcr.SYMBOLS)
    return funding, beta


def test_current_spread_is_compared_to_strictly_prior_history() -> None:
    funding, beta = synthetic_sources()
    baseline = pfcr.build_clock(funding, beta)
    changed = funding.copy()
    changed.iloc[-1, 0] = 0.01
    changed_clock = pfcr.build_clock(changed, beta)
    assert not changed_clock.empty
    last = changed_clock.iloc[-1]
    prior_spreads = (funding.max(axis=1) - funding.min(axis=1)).iloc[:-1]
    expected = float(prior_spreads.tail(pfcr.SPREAD_LOOKBACK).quantile(pfcr.SPREAD_QUANTILE))
    assert last["prior_spread_q90"] == pytest.approx(expected)
    assert last["current_funding_spread"] > last["prior_spread_q90"]
    assert len(changed_clock) >= len(baseline)


def test_clock_latency_and_beta_neutrality() -> None:
    funding, beta = synthetic_sources()
    clock = pfcr.build_clock(funding, beta)
    pfcr.assert_clock_contract(clock)
    assert (clock["feature_available_time"] < clock["entry_time"]).all()
    exposure = clock["long_weight"] * clock["long_beta"]
    exposure -= clock["short_weight_abs"] * clock["short_beta"]
    assert np.allclose(exposure, 0.0)


def test_support_gate_rejects_concentrated_pairs() -> None:
    rows = []
    for index in range(70):
        rows.append(
            {
                "settlement_time": pd.Timestamp("2023-01-01")
                + pd.Timedelta(days=index * 10),
                "long_symbol": "ETHUSDT",
                "short_symbol": "SOLUSDT",
            }
        )
    stats = pfcr.support_stats(pd.DataFrame(rows))
    assert stats["passes_support"] is False
    assert stats["gates"]["maximum_ordered_pair_share_at_most_0_25"] is False


def test_empty_clock_fails_support_without_crashing() -> None:
    stats = pfcr.support_stats(
        pd.DataFrame(columns=["settlement_time", "long_symbol", "short_symbol"])
    )
    assert stats["events"] == 0
    assert stats["passes_support"] is False
