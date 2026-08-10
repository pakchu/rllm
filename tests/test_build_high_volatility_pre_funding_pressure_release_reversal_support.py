import numpy as np
import pandas as pd

from training import build_high_volatility_pre_funding_pressure_release_reversal_support as support


def _ohlc(start, periods, values, *, signed=False):
    ts = pd.date_range(start, periods=periods, freq="1min")
    close = np.broadcast_to(np.asarray(values, dtype=float), (periods,)).copy()
    open_ = close.copy()
    if signed:
        high = np.maximum(open_, close) + 0.01
        low = np.minimum(open_, close) - 0.01
    else:
        high = np.maximum(open_, close) * 1.001
        low = np.minimum(open_, close) * 0.999
    return pd.DataFrame({
        "ts": ts, "open": open_, "high": high, "low": low, "close": close,
        "duplicate_count": 1,
    })


def test_strict_prior_midrank_excludes_current_and_invalid_values():
    values = pd.Series([1.0, 1.5, np.nan, 2.0, 2.0, 3.0])
    ranks = support.strict_prior_midrank(values, lookback=3, minimum=2)
    assert np.isnan(ranks.iloc[0]) and np.isnan(ranks.iloc[1])
    assert np.isnan(ranks.iloc[2])
    assert ranks.iloc[3] == 1.0
    assert ranks.iloc[4] == 5 / 6
    # The finite prior window is [1.5, 2, 2]; the current 3 is not included.
    assert ranks.iloc[5] == 1.0


def test_signed_premium_coherence_and_exact_feature_math():
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=24)
    btc = _ohlc(start, 1440, 100.0)
    btc.loc[1380:, "close"] = 102.0
    btc.loc[1380:, "high"] = 102.0
    premium = _ohlc(start, 1440, -0.2, signed=True)
    prepared_btc = support.prepare_ohlc(btc, positive=True)
    prepared_premium = support.prepare_ohlc(premium, positive=False)
    assert bool(prepared_premium.source_valid.all())
    feature = support.boundary_features(prepared_btc, prepared_premium, decision)
    assert feature["source_valid"] is True
    assert np.isclose(feature["btc_return"], np.log(102.0 / 100.0))
    assert np.isclose(feature["premium_pressure"], -0.2)
    assert feature["pressure_alignment"] is False
    assert np.isclose(feature["btc_realized_variation"], np.sqrt(60 * np.log(1.02) ** 2))


def test_premium_coherence_rejects_bad_signed_bar_without_positive_gate():
    frame = _ohlc(pd.Timestamp("2024-01-01T00:00:00Z"), 2, -0.2, signed=True)
    frame.loc[1, "high"] = -0.3
    prepared = support.prepare_ohlc(frame, positive=False)
    assert prepared.source_valid.tolist() == [True, False]


def test_mean_premium_pressure_is_strict_nonzero():
    features = _feature_frame().iloc[[0]].copy()
    features.loc[:, "premium_pressure"] = 0.0
    features.loc[:, "pressure_alignment"] = False
    active, _, _ = support.active_and_side(features, "no_premium_alignment")
    assert active.tolist() == [False]


def _feature_frame():
    decisions = pd.to_datetime([
        "2024-07-01T00:00:00Z", "2024-07-01T08:00:00Z",
        "2024-07-01T16:00:00Z", "2024-07-02T00:00:00Z",
    ])
    return pd.DataFrame({
        "decision_time": decisions,
        "feature_available_time": decisions,
        "source_valid": [True] * 4,
        "btc_return": [0.03, -0.04, 0.05, -0.06],
        "abs_btc_return": [0.03, 0.04, 0.05, 0.06],
        "premium_pressure": [0.1, -0.1, 0.1, -0.1],
        "pressure_alignment": [True] * 4,
        "btc_realized_variation": [0.2] * 4,
        "return_tail_rank": [0.75] * 4,
        "variation_rank": [0.65] * 4,
    })


def test_primary_signal_clock_side_and_exact_boundary():
    clock = support.build_clock(_feature_frame())
    assert len(clock) == 4
    assert clock.side.tolist() == [-1, 1, -1, 1]
    assert (clock.entry_time == clock.decision_time + pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time == clock.entry_time + pd.Timedelta(hours=6)).all()
    assert set(clock.decision_time.dt.hour) == {0, 8, 16}


def test_stale_pressure_uses_exact_prior_boundary_and_controls_are_frozen():
    features = _feature_frame()
    features.loc[1, "premium_pressure"] = 0.1
    primary, _, _ = support.active_and_side(features)
    stale, _, stale_pressure = support.active_and_side(features, "one_boundary_stale_pressure")
    assert primary.tolist() == [True, False, True, True]
    assert stale.tolist() == [False, False, True, False]
    assert stale_pressure.iloc[1] == features.premium_pressure.iloc[0]
    assert support.CONTROLS == (
        "no_return_tail", "no_volatility_gate", "no_premium_alignment",
        "one_boundary_stale_pressure", "direction_flip",
    )


def test_queries_and_builder_are_outcome_blind():
    queries = support.BTC_QUERY + support.PREMIUM_QUERY
    assert "bars_binance\n" in support.BTC_QUERY
    assert "bars_binance_premium" in support.PREMIUM_QUERY
    assert "funding_rate" not in queries.lower()
    assert "pnl" not in queries.lower()
    assert support.END == pd.Timestamp("2026-08-01T00:00:00Z")
    source = open(support.__file__).read()
    assert "funding_rates_binance" not in source
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"advance_to_economic_outcomes": False' in source
