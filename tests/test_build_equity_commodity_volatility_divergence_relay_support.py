import numpy as np
import pandas as pd

from training import build_equity_commodity_volatility_divergence_relay_support as support
from training import preregister_equity_commodity_volatility_divergence_relay as prereg


def synthetic_cboe() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-06-27", "2024-06-28", "2024-07-01", "2024-07-02"])
    return support.combine_cboe_sources(
        pd.DataFrame({"observation_date": dates, "VIX_close": [10.0, 11.0, 12.0, 11.0]}),
        pd.DataFrame({"observation_date": dates, "VXN_close": [20.0, 22.0, 24.0, 23.0]}),
        pd.DataFrame({"observation_date": dates, "GVZ_close": [30.0, 33.0, 36.0, 35.0]}),
        pd.DataFrame({"observation_date": dates, "OVX_close": [40.0, 36.0, 32.0, 31.0]}),
    )


def synthetic_bars(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="1min", inclusive="left")
    return pd.DataFrame({"open": 100.0, "close": 101.0}, index=index)


def test_common_intersection_and_next_common_session_feature_clock():
    cboe = synthetic_cboe()
    decision = pd.Timestamp("2024-07-01 09:30", tz=support.NY).tz_convert("UTC")
    bars = synthetic_bars(decision - pd.Timedelta(hours=24), decision)
    features = support.build_features(cboe, bars, rank_lookback=1, rank_minimum=1)
    assert len(features) == 2
    first = features.iloc[0]
    assert first.divergence_source_date == pd.Timestamp("2024-06-28")
    assert first.next_common_source_date == pd.Timestamp("2024-07-01")
    assert first.decision_time == decision
    assert first.positive_count == 3
    assert first.negative_count == 1
    expected = np.sqrt(1440 * np.log(101.0 / 100.0) ** 2)
    assert np.isclose(first.btc_realized_variation, expected)


def test_divergence_sides_zeros_and_exact_controls():
    frame = pd.DataFrame(
        {
            "vix_change": [0.1, -0.1, 0.1, 0.0],
            "vxn_change": [0.2, -0.2, 0.2, 0.1],
            "gvz_change": [-0.3, 0.3, 0.3, -0.1],
            "ovx_change": [-0.4, 0.4, -0.4, 0.0],
            "btc_variation_rank": [0.7, 0.7, 0.7, 0.7],
        }
    )
    active, side, _ = support.conditions(frame)
    assert active.tolist() == [True, True, False, False]
    assert side.tolist() == [-1, 1, 0, 0]
    assert support.CONTROLS == (
        "no_volatility_gate", "equity_pair_only", "vix_gvz_only",
        "one_session_stale_divergence", "direction_flip",
    )
    pair_active, pair_side, _ = support.conditions(frame, "equity_pair_only")
    assert pair_active.tolist() == [True, True, True, False]
    assert pair_side.tolist() == [-1, 1, -1, 0]
    vix_gvz_active, vix_gvz_side, _ = support.conditions(frame, "vix_gvz_only")
    assert vix_gvz_active.tolist() == [True, True, False, False]
    assert vix_gvz_side.tolist() == [-1, 1, 0, 0]
    flipped_active, flipped_side, _ = support.conditions(frame, "direction_flip")
    assert flipped_active.equals(active)
    assert flipped_side.tolist() == [1, -1, 0, 0]
    stale_active, stale_side, _ = support.conditions(frame, "one_session_stale_divergence")
    assert stale_active.tolist() == [False, True, True, False]
    assert stale_side.tolist() == [0, -1, 1, 0]


def test_strict_prior_rank_excludes_current_caps_history_and_volatility_gate():
    values = pd.Series([1.0, 2.0, 3.0, 2.0])
    rank = support.strict_prior_midrank(values, lookback=2, minimum=2)
    assert np.isnan(rank.iloc[:2]).all()
    assert rank.iloc[2] == 1.0
    assert rank.iloc[3] == 0.25
    frame = pd.DataFrame(
        {
            "vix_change": [0.1], "vxn_change": [0.1],
            "gvz_change": [-0.1], "ovx_change": [-0.1],
            "btc_variation_rank": [0.64],
        }
    )
    assert not support.conditions(frame)[0].iloc[0]
    assert support.conditions(frame, "no_volatility_gate")[0].iloc[0]


def test_clock_uses_0935_et_holds_12h_and_reserves_half_open():
    decision = pd.Timestamp("2024-07-01 09:30", tz=support.NY).tz_convert("UTC")
    features = pd.DataFrame(
        {
            "divergence_source_date": pd.to_datetime(["2024-06-28", "2024-07-01"]),
            "next_common_source_date": pd.to_datetime(["2024-07-01", "2024-07-02"]),
            "decision_time": [decision, decision + pd.Timedelta(hours=1)],
            "vix_change": [0.1, -0.1], "vxn_change": [0.1, -0.1],
            "gvz_change": [-0.1, 0.1], "ovx_change": [-0.1, 0.1],
            "positive_count": [2, 2], "negative_count": [2, 2],
            "btc_realized_variation": [0.2, 0.2], "btc_variation_rank": [0.9, 0.9],
        }
    )
    clock = support.build_clock(features)
    assert len(clock) == 1
    assert clock.iloc[0].entry_time == decision + pd.Timedelta(minutes=5)
    assert clock.iloc[0].exit_time == decision + pd.Timedelta(hours=12, minutes=5)
    assert clock.iloc[0].side == -1


def test_builder_is_structurally_bound_and_outcomes_remain_sealed():
    source = support.BUILDER.read_text()
    assert support.PREREG_SHA == "ad718d3fcd5d2885813a98dd2131a4ab7feca2c95ab89658a26100db50319637"
    assert support.prereg.SOURCES == prereg.SOURCES
    assert "prereg.SOURCES" in source
    assert "FROM bars_binance" in support.BTC_QUERY
    assert "funding_rates_binance" not in source
    assert "evaluator" not in source.lower()
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
    assert '"promotion_authorized": False' in source
    assert support.ECONOMIC_OUTCOMES_AUTHORIZED is False
