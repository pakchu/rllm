import numpy as np
import pandas as pd

from training import build_nasdaq_volatility_leadership_rotation_relay_support as support
from training import preregister_nasdaq_volatility_leadership_rotation_relay as prereg


def synthetic_cboe() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2024-06-26", "2024-06-27", "2024-06-28", "2024-07-01", "2024-07-02"]
    )
    return support.combine_cboe_sources(
        pd.DataFrame({"observation_date": dates, "VIX_close": [10, 11, 10, 12, 11]}),
        pd.DataFrame({"observation_date": dates, "VXN_close": [20, 21, 24, 23, 25]}),
        pd.DataFrame({"observation_date": dates, "GVZ_close": [30, 29, 31, 30, 32]}),
        pd.DataFrame({"observation_date": dates, "OVX_close": [40, 42, 39, 41, 40]}),
    )


def synthetic_bars(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="1min", inclusive="left")
    return pd.DataFrame({"open": 100.0, "close": 101.0}, index=index)


def test_common_intersection_zscores_and_next_common_session_feature_clock():
    cboe = synthetic_cboe()
    first_decision = pd.Timestamp("2024-06-28 09:30", tz=support.NY).tz_convert("UTC")
    last_decision = pd.Timestamp("2024-07-02 09:30", tz=support.NY).tz_convert("UTC")
    bars = synthetic_bars(first_decision - pd.Timedelta(hours=24), last_decision)
    features = support.build_features(
        cboe, bars, rank_lookback=1, rank_minimum=1,
        zscore_lookback=2, zscore_minimum=2,
    )
    assert len(features) == 3
    row = features.iloc[2]
    assert row.residual_source_date == pd.Timestamp("2024-07-01")
    assert row.next_common_source_date == pd.Timestamp("2024-07-02")
    assert row.decision_time == pd.Timestamp("2024-07-02 09:30", tz=support.NY).tz_convert("UTC")
    for symbol in ("vix", "vxn", "gvz", "ovx"):
        history = cboe[f"{symbol}_change"].iloc[1:3]
        expected = (cboe[f"{symbol}_change"].iloc[3] - history.mean()) / history.std(ddof=1)
        assert np.isclose(row[f"z_{symbol}"], expected)
    expected_residual = row.z_vxn - np.median([row.z_vix, row.z_gvz, row.z_ovx])
    assert np.isclose(row.leadership_residual, expected_residual)
    expected_variation = np.sqrt(1440 * np.log(101.0 / 100.0) ** 2)
    assert np.isclose(row.btc_realized_variation, expected_variation)


def test_causal_zscore_excludes_current_caps_history_and_requires_sample_std():
    values = pd.Series([1.0, 2.0, 4.0, 8.0])
    zscore = support.causal_z(values, lookback=2, minimum=2)
    assert np.isnan(zscore.iloc[:2]).all()
    assert np.isclose(zscore.iloc[2], (4.0 - 1.5) / np.std([1.0, 2.0], ddof=1))
    assert np.isclose(zscore.iloc[3], (8.0 - 3.0) / np.std([2.0, 4.0], ddof=1))
    assert support.causal_z(pd.Series([1.0, 1.0, 2.0]), 2, 2).isna().all()


def test_leadership_sides_tail_and_exact_controls():
    frame = pd.DataFrame(
        {
            "btc_variation_rank": [0.7, 0.7, 0.7, 0.7],
            "leadership_residual": [0.5, -0.5, 0.2, 0.0],
            "absolute_leadership_rank": [0.8, 0.8, 0.69, 0.8],
            "vxn_minus_vix_raw": [-0.4, 0.4, -0.3, 0.2],
            "absolute_vxn_minus_vix_raw_rank": [0.8, 0.69, 0.8, 0.8],
        }
    )
    active, side, _ = support.conditions(frame)
    assert active.tolist() == [True, True, False, False]
    assert side.tolist() == [-1, 1, -1, 0]
    assert support.CONTROLS == (
        "no_btc_variation_gate", "no_leadership_tail", "vxn_minus_vix_raw",
        "one_session_stale_leadership", "direction_flip", "forced_long",
    )
    untail_active, _, _ = support.conditions(frame, "no_leadership_tail")
    assert untail_active.tolist() == [True, True, True, False]
    raw_active, raw_side, _ = support.conditions(frame, "vxn_minus_vix_raw")
    assert raw_active.tolist() == [True, False, True, True]
    assert raw_side.tolist() == [1, -1, 1, -1]
    flipped_active, flipped_side, _ = support.conditions(frame, "direction_flip")
    assert flipped_active.equals(active)
    assert flipped_side.tolist() == [1, -1, 1, 0]
    stale_active, stale_side, _ = support.conditions(frame, "one_session_stale_leadership")
    assert stale_active.tolist() == [False, True, True, False]
    assert stale_side.tolist() == [0, -1, 1, -1]
    forced_active, forced_side, _ = support.conditions(frame, "forced_long")
    assert forced_active.equals(active)
    assert forced_side.tolist() == [1, 1, 1, 1]


def test_strict_prior_ranks_and_btc_variation_gate():
    values = pd.Series([1.0, 2.0, 3.0, 2.0])
    rank = support.strict_prior_midrank(values, lookback=2, minimum=2)
    assert np.isnan(rank.iloc[:2]).all()
    assert rank.iloc[2] == 1.0
    assert rank.iloc[3] == 0.25
    frame = pd.DataFrame(
        {
            "btc_variation_rank": [0.64], "leadership_residual": [0.2],
            "absolute_leadership_rank": [0.8], "vxn_minus_vix_raw": [-0.1],
            "absolute_vxn_minus_vix_raw_rank": [0.8],
        }
    )
    assert not support.conditions(frame)[0].iloc[0]
    assert support.conditions(frame, "no_btc_variation_gate")[0].iloc[0]


def test_clock_uses_0935_et_holds_12h_and_reserves_half_open():
    decision = pd.Timestamp("2024-07-01 09:30", tz=support.NY).tz_convert("UTC")
    features = pd.DataFrame(
        {
            "residual_source_date": pd.to_datetime(["2024-06-28", "2024-07-01"]),
            "next_common_source_date": pd.to_datetime(["2024-07-01", "2024-07-02"]),
            "decision_time": [decision, decision + pd.Timedelta(hours=1)],
            "vix_change": [0.1, -0.1], "vxn_change": [0.2, -0.2],
            "gvz_change": [-0.1, 0.1], "ovx_change": [-0.2, 0.2],
            "z_vix": [0.1, -0.1], "z_vxn": [0.5, -0.5],
            "z_gvz": [-0.1, 0.1], "z_ovx": [-0.2, 0.2],
            "btc_realized_variation": [0.2, 0.2], "btc_variation_rank": [0.9, 0.9],
            "leadership_residual": [0.5, -0.5], "absolute_leadership_rank": [0.9, 0.9],
            "vxn_minus_vix_raw": [0.1, -0.1],
            "absolute_vxn_minus_vix_raw_rank": [0.9, 0.9],
        }
    )
    clock = support.build_clock(features)
    assert len(clock) == 1
    assert clock.iloc[0].candidate == "NVLRR-12"
    assert clock.iloc[0].entry_time == decision + pd.Timedelta(minutes=5)
    assert clock.iloc[0].exit_time == decision + pd.Timedelta(hours=12, minutes=5)
    assert clock.iloc[0].side == -1


def test_builder_is_structurally_bound_and_outcomes_remain_sealed():
    source = support.BUILDER.read_text()
    assert support.PREREG_SHA == "c091adf520e5f9b578922ef79f7a4c523da9b0964884bc523afbbe4a90ed4e6d"
    assert support.prereg.SOURCES == prereg.SOURCES
    assert "prereg.SOURCES" in source
    assert "FROM bars_binance" in support.BTC_QUERY
    assert "funding_rates_binance" not in source
    assert "evaluator" not in source.lower()
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
    assert '"promotion_authorized": False' in source
    assert support.ECONOMIC_OUTCOMES_AUTHORIZED is False
