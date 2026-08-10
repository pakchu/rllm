import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_high_volatility_corwin_schultz_spread_expansion_reversal_support as support
from training import preregister_high_volatility_corwin_schultz_spread_expansion_reversal as prereg


def test_corwin_schultz_formula_is_exact():
    high1, low1, high2, low2 = 110.0, 100.0, 121.0, 99.0
    beta, gamma, alpha, spread = support.corwin_schultz_spread(high1, low1, high2, low2)
    expected_beta = math.log(high1 / low1) ** 2 + math.log(high2 / low2) ** 2
    expected_gamma = math.log(max(high1, high2) / min(low1, low2)) ** 2
    denominator = 3 - 2 * math.sqrt(2)
    expected_alpha = (math.sqrt(2 * expected_beta) - math.sqrt(expected_beta)) / denominator - math.sqrt(expected_gamma / denominator)
    expected_spread = 2 * (math.exp(max(expected_alpha, 0)) - 1) / (1 + math.exp(max(expected_alpha, 0)))
    assert (beta, gamma, alpha, spread) == pytest.approx((expected_beta, expected_gamma, expected_alpha, expected_spread))


def test_rank_is_strict_prior_and_uses_midrank():
    ranks = support.strict_prior_midrank(pd.Series([1.0, 1.0, 2.0]), lookback=2, minimum=2)
    assert np.isnan(ranks.iloc[:2]).all()
    assert ranks.iloc[2] == 1.0
    tied = support.strict_prior_midrank(pd.Series([1.0, 2.0, 1.0]), lookback=2, minimum=2)
    assert tied.iloc[2] == 0.25


def test_side_and_clock_are_frozen_reversal_at_2105_for_twelve_hours():
    decision = pd.Timestamp("2024-07-01T21:00:00Z")
    frame = pd.DataFrame({
        "decision_time": [decision, decision + pd.Timedelta(days=1)], "source_valid": [True, True],
        "prior_high": [2.0, 2.0], "prior_low": [1.0, 1.0], "current_high": [2.0, 2.0], "current_low": [1.0, 1.0],
        "beta": [1.0, 1.0], "gamma": [1.0, 1.0], "alpha": [1.0, 1.0], "implied_spread": [0.2, 0.2],
        "spread_expansion": [0.1, 0.1], "spread_expansion_rank": [0.8, 0.8],
        "realized_variation": [0.2, 0.2], "variation_rank": [0.7, 0.7], "completed_return": [0.01, -0.02],
    })
    clock = support.build_clock(frame)
    assert clock.side.tolist() == [-1, 1]
    assert pd.Timestamp(clock.iloc[0].entry_time) == decision + pd.Timedelta(minutes=5)
    assert pd.Timestamp(clock.iloc[0].exit_time) == decision + pd.Timedelta(hours=12, minutes=5)
    assert (clock.entry_time.iloc[1] >= clock.exit_time.iloc[0])


def test_builder_is_ohlc_only_and_outcome_blind():
    text = support.BUILDER.read_text()
    assert support.QUERY == "SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
    assert support.END == pd.Timestamp("2026-08-01T00:00:00Z")
    for forbidden in ("funding_rates_binance", "bars_binance_spot", "quote_asset_volume"):
        assert forbidden not in text
    assert '"postentry_return_pnl_execution_price_opened": False' in text
    assert '"advance_to_economic_outcomes": False' in text
    assert '"promotion_authorized": False' in text
    assert support.CONTROLS == tuple(prereg.build()["diagnostic_controls"]["names"])
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    assert support.sha(prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
