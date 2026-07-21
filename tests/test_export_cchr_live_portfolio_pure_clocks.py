from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing.market_features import build_market_feature_frame
from training import export_cchr_live_portfolio_pure_clocks as live


def test_candidate_map_and_frozen_weights_are_exact() -> None:
    assert live.candidate_map() == {
        "live:cand_rex_veto_7": {
            "family": "live",
            "parameters": {"component": "cand_rex_veto_7"},
            "hold_bars": 144,
            "component_weight": 1.45,
        },
        "live:new_long_minimal_funding_premium": {
            "family": "live",
            "parameters": {"component": "new_long_minimal_funding_premium"},
            "hold_bars": 576,
            "component_weight": 1.75,
        },
        "live:oi_upbit_ratio288_low": {
            "family": "live",
            "parameters": {"component": "oi_upbit_ratio288_low"},
            "hold_bars": 30,
            "component_weight": 0.65,
        },
    }
    assert live.frozen_contract()["real_source_execution_authorized"] is False


def _small_market(periods: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=periods, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
            "quote_asset_volume": 1000.0,
            "taker_buy_base": 5.0,
            "usdkrw": 1300.0,
            "usdkrw_available": 1.0,
            "open_interest": 100.0,
            "open_interest_available": 1.0,
        }
    )


def test_auxiliary_join_never_exposes_future_funding_or_premium() -> None:
    market = _small_market()
    funding = pd.DataFrame(
        {
            "date": ["2023-01-01T00:30:00Z"],
            "funding_time": ["2023-01-01T00:30:00Z"],
            "funding_rate": [-0.001],
        }
    )
    premium = pd.DataFrame(
        {
            "date": ["2023-01-01T00:00:00Z"],
            "close": [-0.002],
            "close_time": [pd.Timestamp("2023-01-01T00:59:59.999Z").value // 1_000_000],
        }
    )
    joined = live.attach_live_auxiliary(market, funding, premium)
    before_funding = joined["date"] < pd.Timestamp("2023-01-01T00:30:00Z")
    assert joined.loc[before_funding, "funding_available"].eq(0.0).all()
    assert (
        joined.loc[
            joined["date"].eq(pd.Timestamp("2023-01-01T00:30:00Z")), "funding_rate"
        ]
        .eq(-0.001)
        .all()
    )
    assert (
        joined.loc[
            joined["date"] < pd.Timestamp("2023-01-01T01:00:00Z"), "premium_available"
        ]
        .eq(0.0)
        .all()
    )
    assert (
        joined.loc[
            joined["date"].eq(pd.Timestamp("2023-01-01T01:00:00Z")), "premium_available"
        ]
        .eq(1.0)
        .all()
    )


def test_auxiliary_join_rejects_an_incomplete_market_grid() -> None:
    market = _small_market().drop(index=3)
    with pytest.raises(ValueError, match="complete five-minute grid"):
        live.attach_live_auxiliary(market, pd.DataFrame(), pd.DataFrame())


def test_upbit_join_is_backward_only_and_carries_availability() -> None:
    market = _small_market().iloc[:4].copy()
    market["funding_rate"] = 0.0
    market["funding_available"] = 1.0
    market["premium_index_change"] = 0.0
    market["premium_available"] = 1.0
    features = pd.DataFrame(index=market.index)
    upbit = pd.DataFrame(
        {
            "date": ["2023-01-01T00:05:00Z", "2023-01-01T00:15:00Z"],
            "close": [20_000_000.0, 21_000_000.0],
            "volume": [1.0, 1.0],
        }
    )
    output = live.attach_upbit_volume_feature(market, features, upbit)
    assert output["upbit_volume_available"].tolist() == [0.0, 1.0, 1.0, 1.0]


def test_live_feature_union_matches_each_frozen_sleeve_contract() -> None:
    periods = 6_000
    index = np.arange(periods, dtype=float)
    close = 100.0 + 0.002 * index + np.sin(index / 17.0)
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=periods, freq="5min", tz="UTC"),
            "open": close - 0.05,
            "high": close + 0.4 + 0.02 * np.cos(index / 11.0),
            "low": close - 0.4 - 0.02 * np.sin(index / 13.0),
            "close": close,
            "volume": 10.0 + np.mod(index, 23.0),
            "quote_asset_volume": (10.0 + np.mod(index, 23.0)) * close,
            "taker_buy_base": 4.0 + np.mod(index, 7.0),
            "usdkrw": 1_300.0,
            "usdkrw_available": 1.0,
            "open_interest": 1_000.0 + 0.1 * index + np.sin(index / 31.0),
            "open_interest_available": 1.0,
            "funding_rate": -0.00001,
            "funding_available": 1.0,
            "premium_index_change": np.sin(index / 97.0) * 0.001,
            "premium_available": 1.0,
        }
    )
    expected_oi = build_market_feature_frame(
        panel,
        window_size=live.OI_BASE_WINDOW,
        zscore_window=live.OI_RETURN_Z_WINDOW,
        volume_window=48,
    )
    actual = live.build_live_features(panel)

    for column in (
        "range_vol",
        "return_zscore_48",
        "rsi_norm",
        "sma24_ratio",
        "volume_zscore",
        "htf_1d_return_4",
    ):
        np.testing.assert_allclose(
            actual[column], expected_oi[column], rtol=0.0, atol=1e-12
        )

    expected_funding_trend = (
        (
            panel["close"]
            / panel["close"].shift(live.FUNDING_TREND_PERIODS).replace(0.0, np.nan)
            - 1.0
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    np.testing.assert_allclose(
        actual["trend_96"], expected_funding_trend, rtol=0.0, atol=1e-12
    )


def _active_feature_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame(index=range(len(dates)))
    for feature, operation, threshold in live.COMPONENT_SPECS["oi_upbit_ratio288_low"][
        "gates"
    ]:
        frame[feature] = threshold + (1.0 if operation == ">=" else -1.0)
    for flag in (
        "open_interest_available",
        "upbit_volume_available",
        "usdkrw_available",
    ):
        frame[flag] = 1.0
    frame["funding_rate"] = -0.001
    frame["funding_available"] = 1.0
    frame["trend_96"] = 1.0
    frame["premium_index_change"] = 0.0
    frame["premium_available"] = 1.0
    frame["htf_1d_return_4"] = 1.0
    frame["oi_zscore"] = 0.0
    frame["htf_1w_return_4"] = 0.0
    for window in (144, 576, 2016, 8640):
        frame[f"rex_{window}_range_pos"] = -1.0
        frame[f"rex_{window}_max_to_cur_pct"] = 1.0
        frame[f"rex_{window}_cur_to_min_pct"] = 0.0
    frame["trend_24"] = 1.0
    frame["htf_4h_return_1"] = 1.0
    frame["htf_1d_return_4_rex"] = 1.0
    frame["htf_3d_return_4"] = 1.0
    frame["volume_zscore"] = 0.0
    frame["taker_imbalance"] = 0.0
    return frame


def test_component_signals_apply_absolute_stride_and_source_flags() -> None:
    dates = pd.date_range("2023-02-01", periods=48, freq="5min", tz="UTC")
    features = _active_feature_frame(dates)
    signals = live.component_signals(pd.Series(dates), features)
    for component, signal in signals.items():
        spec = live.COMPONENT_SPECS[component]
        expected_slots = live.interval_slots(
            pd.Series(dates),
            stride=int(spec["stride_bars"]),
            offset=int(spec["stride_offset_bars"]),
        )
        assert np.array_equal(signal != 0, expected_slots)
    features.loc[:, "upbit_volume_available"] = 0.0
    blocked = live.component_signals(pd.Series(dates), features)
    assert not blocked["oi_upbit_ratio288_low"].any()


def test_rex_strength_has_expected_direction_and_strict_threshold() -> None:
    dates = pd.date_range("2023-02-01", periods=2, freq="5min", tz="UTC")
    features = _active_feature_frame(dates)
    strength, direction = live.rex_strength_direction(features)
    assert (strength > 0.0).all()
    assert direction.tolist() == [1.0, 1.0]


def test_build_clock_has_next_bar_entry_fixed_hold_and_per_member_nonoverlap(
    monkeypatch,
) -> None:
    dates = pd.date_range("2023-02-01", periods=700, freq="5min", tz="UTC")
    panel = _small_market(700)
    panel["date"] = dates
    features = _active_feature_frame(dates)
    monkeypatch.setattr(live, "attach_live_auxiliary", lambda *args: panel)
    monkeypatch.setattr(live, "build_live_features", lambda _: features)
    monkeypatch.setattr(
        live, "attach_upbit_volume_feature", lambda _panel, base, _upbit: base
    )
    monkeypatch.setattr(
        live,
        "component_signals",
        lambda _dates, _features: {
            "oi_upbit_ratio288_low": np.where(
                np.isin(np.arange(700), [0, 1, 31, 699]), 1, 0
            ),
            "new_long_minimal_funding_premium": np.where(np.arange(700) == 2, 1, 0),
            "cand_rex_veto_7": np.where(np.arange(700) == 3, -1, 0),
        },
    )
    clock = live.build_clock(panel, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    oi = clock.loc[clock["candidate_id"].eq("live:oi_upbit_ratio288_low")]
    assert len(oi) == 2
    assert oi.iloc[0]["decision_time"] == "2023-02-01T00:05:00Z"
    assert oi.iloc[0]["entry_time"] == "2023-02-01T00:05:00Z"
    assert oi.iloc[0]["exit_time"] == "2023-02-01T02:35:00Z"
    assert set(clock["side"]) == {-1, 1}


def test_module_has_no_legacy_or_outcome_imports() -> None:
    source = Path(live.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert imports == {
        "__future__",
        "datetime",
        "pathlib",
        "typing",
        "numpy",
        "pandas",
        "training",
    }
    lowered = source.lower()
    for forbidden in (
        "future_return",
        "strict_mdd",
        "cagr",
        "trade_return",
        "result_json",
    ):
        assert forbidden not in lowered
