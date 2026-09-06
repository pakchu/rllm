import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution import approved_portfolio_signals as s


def _macro_market(periods=9000, start="2026-01-01 00:00:00Z"):
    dates = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    t = np.arange(periods, dtype=float)
    close = 10000.0 * np.exp(0.00003 * t + 0.002 * np.sin(t / 200.0))
    open_ = np.r_[close[0], close[:-1]]
    quote = np.full(periods, 1000.0)
    taker = np.where((t // 12) % 24 < 12, 530.0, 470.0)
    dxy = 100.0 * np.exp(-0.00001 * t)
    kimchi = 0.01 + 0.0001 * np.sin(t / 50.0)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": 1.0,
            "quote_asset_volume": quote,
            "taker_buy_quote": taker,
            "dxy": dxy,
            "dxy_momentum": np.full(periods, 0.003),
            "dxy_available": 1.0,
            "usdkrw": 1300.0,
            "usdkrw_available": 1.0,
            "kimchi_premium": kimchi,
            "kimchi_available": 1.0,
        }
    )


def test_macro_targets_match_frozen_formula_components():
    market = _macro_market()
    got = s.build_macro_targets(market)

    # Independent restatement of the frozen fixed_positions formula over the
    # adapter's causal feature builders; avoids importing production code under
    # test for the final arithmetic.
    calc = market.copy()
    calc["date"] = calc["date"].dt.tz_convert(None)
    x = pd.concat(
        [
            s._base_hourly_features(calc),
            s._macro_features(
                calc[
                    [
                        "date",
                        "dxy",
                        "usdkrw",
                        "kimchi_premium",
                        "dxy_available",
                        "usdkrw_available",
                        "kimchi_available",
                    ]
                ],
                s._base_hourly_features(calc).index,
            ),
        ],
        axis=1,
    )
    size = np.clip(0.2 / (x.vol24.to_numpy() * np.sqrt(s.ANNUAL_HOURS)), 0.1, 1.0)
    flow = x.flow6.to_numpy()
    dollar = x.dxy_change6.to_numpy()
    mom = x.mom720.to_numpy()
    z = x.z24.to_numpy()
    dollar_raw = np.where((np.abs(flow) > 0.02) & (np.sign(flow) * dollar < 0), np.sign(flow), 0) * size
    dollar_ok = np.isfinite(x[["vol24", "flow6", "dxy_change6"]]).all(axis=1) & (x["dxy_valid"] > 0.5)
    dollar_signal = s._hold_signal(np.where(dollar_ok.to_numpy(), dollar_raw, 0.0), 24, x.index)
    switch_raw = np.maximum(
        np.where(np.abs(mom) > 0.75, np.where(np.sign(mom) * flow > 0, np.sign(mom), 0), np.where(np.abs(z) > 1.5, -np.sign(z), 0)),
        0,
    )
    switch_ok = np.isfinite(x[["mom720", "flow6", "z24"]]).all(axis=1)
    switch = s._hold_signal(np.where(switch_ok.to_numpy(), switch_raw, 0.0), 24, x.index)
    expected = pd.Series(np.clip(0.75 * dollar_signal + 0.25 * switch, -1, 1), index=(x.index + pd.Timedelta(minutes=5)).tz_localize("UTC"), name="macro_flow_target")

    pd.testing.assert_series_equal(got, expected)
    assert (got.index.minute == 5).all()
    assert got.abs().max() <= 1.0


def test_macro_targets_exclude_incomplete_hour_and_fail_closed_on_source_missingness():
    market = _macro_market(periods=9000)
    full = s.build_macro_targets(market)
    partial = s.build_macro_targets(market.iloc[:-3])
    assert partial.index[-1] < full.index[-1]

    missing_hour = market.copy()
    mask = (missing_hour["date"] >= pd.Timestamp("2026-01-11 23:00:00Z")) & (
        missing_hour["date"] < pd.Timestamp("2026-01-12 00:00:00Z")
    )
    missing_hour.loc[mask, "dxy_available"] = 0.0
    targets = s.build_macro_targets(missing_hour)
    # Macro features are delayed one completed hour; the 23:00-00:00 missing
    # source hour suppresses the next daily refresh at 01:00 / execution 01:05.
    assert abs(targets.loc[pd.Timestamp("2026-01-12 01:05:00Z")]) <= 0.25
    assert targets.loc[pd.Timestamp("2026-01-12 02:05:00Z")] == targets.loc[pd.Timestamp("2026-01-12 01:05:00Z")]


def _legacy_market(periods=7000, start="2020-01-01 00:00:00Z"):
    dates = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    t = np.arange(periods, dtype=float)
    close = 10000.0 * (1.0 + 0.00002 * t)
    dxy = 100.0 + 0.00002 * t
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "dxy": dxy,
            "dxy_momentum": np.full(periods, 0.003),
            "dxy_available": 1.0,
            "volume": 1.0,
            "quote_asset_volume": 1.0,
            "taker_buy_base": 0.5,
            "number_of_trades": 1.0,
        }
    )


def test_dollar_short_scores_phase_gates_and_lifecycle():
    market = _legacy_market()
    decision = pd.Timestamp("2020-01-22 23:55:00Z")
    out = s.score_dollar_short(market, decision)
    assert out["active"] is True
    assert out["position"] == -1.0
    assert out["execution_time"] == "2020-01-23T00:00:00+00:00"
    assert out["lifecycle"]["exit_time"] == "2020-01-23T12:00:00+00:00"
    assert out["lifecycle"]["take_profit"] is None
    assert out["lifecycle"]["stop_loss"] is None
    assert out["global_phase"]["matched"] is True


def test_macro_feature_parity_against_frozen_research_helper():
    from training import search_macro_flow_alpha_combinations as research

    market = _macro_market(periods=2000)
    market_naive = market.copy()
    market_naive["date"] = market_naive["date"].dt.tz_convert(None)
    idx = s._base_hourly_features(market_naive).index
    cols = ["date", "dxy", "usdkrw", "kimchi_premium", "dxy_available", "usdkrw_available", "kimchi_available"]
    got = s._macro_features(market_naive[cols], idx)
    expected = research.macro_features(market_naive[cols], idx)
    pd.testing.assert_frame_equal(got, expected)


def test_legacy_144_feature_parity_against_preprocessing_helper():
    from preprocessing.market_features import build_market_feature_frame

    market = _legacy_market()
    got = s._original_144_features(market)
    expected = build_market_feature_frame(market.assign(date=market["date"].dt.tz_convert(None)), window_size=144)
    np.testing.assert_allclose(got["dxy_momentum"], expected["dxy_momentum"], equal_nan=True)
    np.testing.assert_allclose(got["htf_1d_return_4"].fillna(0.0), expected["htf_1d_return_4"], equal_nan=True)


def test_dollar_short_fail_closed_for_off_phase_missing_source_and_warmup():
    market = _legacy_market()
    off_phase = s.score_dollar_short(market, "2020-01-22 23:50:00Z")
    assert off_phase["active"] is False
    assert "off_global_phase" in off_phase["reason"]

    missing = market.copy()
    row = missing.index[missing["date"].eq(pd.Timestamp("2020-01-22 23:55:00Z"))][0]
    missing.loc[row, "dxy_available"] = 0.0
    blocked = s.score_dollar_short(missing, "2020-01-22 23:55:00Z")
    assert blocked["active"] is False
    assert "dxy_unavailable_at_signal" in blocked["reason"]

    warmup = s.score_dollar_short(market.iloc[:200], "2020-01-01 11:55:00Z")
    assert warmup["active"] is False
    assert "insufficient_warmup" in warmup["reason"]


def test_macro_asof_does_not_admit_unclosed_bar():
    market=_macro_market()
    asof=market.date.iloc[-1]+pd.Timedelta(minutes=2)
    result=s.build_macro_targets(market,asof=asof)
    assert (result.index-pd.Timedelta(minutes=5)<=asof).all()
    altered=market.copy();altered.loc[altered.index[-1],'close']*=2
    pd.testing.assert_series_equal(result,s.build_macro_targets(altered,asof=asof))


def test_missing_availability_and_duplicate_dates_are_rejected():
    import pytest
    market=_macro_market()
    with pytest.raises(ValueError):s.build_macro_targets(market.drop(columns=['dxy_available']))
    with pytest.raises(ValueError):s.build_macro_targets(pd.concat([market,market.iloc[[-1]]]))
    with pytest.raises(ValueError):s.score_dollar_short(_legacy_market().drop(columns=['dxy_available']),'2020-01-22 23:55Z')


def test_macro_missing_volume_is_not_silently_summed():
    import pytest
    market=_macro_market();market.loc[100,'taker_buy_quote']=np.nan
    with pytest.raises(ValueError):s.build_macro_targets(market)
