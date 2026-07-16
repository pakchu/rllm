import json

import numpy as np
import pandas as pd

from execution.portfolio_shadow_policies import (
    build_fresh_kimchi_feature_frame,
    build_markov_feature_frame,
    observable_markov_transition_keys,
)
from preprocessing.market_features import build_market_feature_frame
from training.search_bidirectional_state_alpha import extra as research_bidirectional_features
from training.search_gaussian_hmm_regime_alpha import hourly_features
from training.search_kimchi_leadlag_bidirectional_alpha import features as research_kimchi_features


def _market(rows: int = 900) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 30_000.0 * np.exp(0.00005 * idx + 0.004 * np.sin(idx / 37.0))
    quote = 1_000_000.0 + 20_000.0 * np.cos(idx / 11.0)
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="5min"),
            "open": close * (1.0 - 0.0002),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": quote / close,
            "number_of_trades": 100.0,
            "taker_buy_base": (quote / close) * (0.50 + 0.08 * np.sin(idx / 17.0)),
            "quote_asset_volume": quote,
            "taker_buy_quote": quote * (0.50 + 0.08 * np.sin(idx / 17.0)),
            "kimchi_premium": 0.02 + 0.002 * np.sin(idx / 29.0),
            "usdkrw": 1_350.0 + 3.0 * np.cos(idx / 41.0),
        }
    )


def test_fresh_kimchi_custom_features_match_frozen_research_equations():
    market = _market()
    base = pd.DataFrame(index=market.index)
    expected = research_kimchi_features(
        market,
        research_bidirectional_features(market, base.copy()),
    )
    actual = build_fresh_kimchi_feature_frame(market, base.copy())
    for column in ("bd_flow_accel", "kl_local_impulse_144", "kl_accel_48_144"):
        np.testing.assert_allclose(
            actual[column].to_numpy(float),
            expected[column].to_numpy(float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


def test_shadow_policy_adapters_are_prefix_causal():
    market = _market(rows=2_400)
    prefix_rows = 1_800
    base = build_market_feature_frame(market, window_size=288)
    prefix_base = build_market_feature_frame(market.iloc[:prefix_rows], window_size=288)

    fresh_full = build_fresh_kimchi_feature_frame(market, base)
    fresh_prefix = build_fresh_kimchi_feature_frame(
        market.iloc[:prefix_rows].reset_index(drop=True),
        prefix_base,
    )
    markov_full = build_markov_feature_frame(market, base)
    markov_prefix = build_markov_feature_frame(
        market.iloc[:prefix_rows].reset_index(drop=True),
        prefix_base,
    )
    for column in ("bd_flow_accel", "kl_local_impulse_144"):
        np.testing.assert_allclose(
            fresh_full[column].iloc[:prefix_rows],
            fresh_prefix[column],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    for column in ("trend_96", "range_pos", "volume_zscore", "htf_4h_return_4"):
        np.testing.assert_allclose(
            markov_full[column].iloc[:prefix_rows],
            markov_prefix[column],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

    spec = json.loads(
        open("research/pools/alphas/markov_persistent_funding_premium_long_20260712.json").read()
    )["state_model"]
    np.testing.assert_array_equal(
        observable_markov_transition_keys(market, spec)[:prefix_rows],
        observable_markov_transition_keys(market.iloc[:prefix_rows].reset_index(drop=True), spec),
    )


def test_markov_transition_keys_match_frozen_research_mapping():
    market = _market(rows=2_400)
    spec = json.loads(
        open("research/pools/alphas/markov_persistent_funding_premium_long_20260712.json").read()
    )["state_model"]
    actual = observable_markov_transition_keys(market, spec)

    _, hourly = hourly_features(market)
    trend = np.where(
        hourly["trend24"] <= float(spec["trend_low"]),
        0,
        np.where(hourly["trend24"] >= float(spec["trend_high"]), 2, 1),
    )
    volatility = (hourly["vol24"] >= float(spec["vol_median"])).astype(int)
    flow = (hourly["flow24"] >= float(spec["flow_median"])).astype(int)
    state = trend * 4 + volatility * 2 + flow
    previous = pd.Series(state, index=hourly.index).shift(1).fillna(-1).astype(int)
    transitions = previous * 12 + state
    expected = pd.merge_asof(
        pd.DataFrame({"date": market["date"], "position": np.arange(len(market))}),
        pd.DataFrame({"date": hourly.index.to_numpy(), "transition": transitions.to_numpy()}),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("2h"),
    ).sort_values("position")["transition"].fillna(-1).to_numpy(int)
    np.testing.assert_array_equal(actual, expected)


def test_markov_base_features_match_frozen_144_window_contract():
    market = _market(rows=900)
    base_features = build_market_feature_frame(
        market,
        window_size=288,
        zscore_window=96,
        volume_window=96,
    )
    base_features["funding_available"] = 1.0
    expected = build_market_feature_frame(
        market,
        window_size=144,
        zscore_window=48,
        volume_window=48,
    )
    actual = build_markov_feature_frame(market, base_features)

    for column in ("trend_96", "range_pos", "volume_zscore", "htf_1d_return_4"):
        np.testing.assert_allclose(actual[column], expected[column], rtol=0.0, atol=0.0)
    assert actual["funding_available"].iloc[-1] == 1.0
