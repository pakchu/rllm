from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import _attach_delayed_metrics
from training.search_oi_cost_basis_liquidation_alpha import (
    OiCostBasisConfig,
    _candidate_identity,
    _stable_json_hash,
    build_cost_basis_features,
    build_signals,
    load_pre2024_market,
)


def _toy_market(n: int, start: str = "2021-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="5min")
    idx = np.arange(n, dtype=float)
    close = 100.0 + 0.01 * idx
    raw = 0.5 * np.sin(idx / 13.0) + 0.25 * np.cos(idx / 7.0)
    if n > 322:
        raw[320] = 3.0
    oi = np.full(n, 1000.0)
    if n > 322:
        oi[320:] *= 1.05
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.01,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "sum_open_interest": oi,
            "sum_toptrader_long_short_ratio": np.exp(raw),
            "count_long_short_ratio": np.exp(raw),
            "sum_taker_long_short_vol_ratio": np.exp(raw),
            "quote_asset_volume": np.full(n, 10_000.0),
            "taker_buy_quote": np.full(n, 5_100.0),
            "positioning_available": np.ones(n),
        }
    )


def _attachable_market(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[
        "sum_open_interest",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ])


def _toy_metrics(n: int, start: str = "2021-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="5min")
    idx = np.arange(n, dtype=float)
    raw = 0.5 * np.sin(idx / 11.0) + 0.1 * np.cos(idx / 5.0)
    return pd.DataFrame(
        {
            "create_time": dates,
            "symbol": "BTCUSDT",
            "sum_open_interest": 1000.0 + idx,
            "sum_toptrader_long_short_ratio": np.exp(raw),
            "count_long_short_ratio": np.exp(raw / 2.0),
            "sum_taker_long_short_vol_ratio": np.exp(-raw / 3.0),
        }
    )


def test_cost_basis_pressure_is_causal_pre_update() -> None:
    market = _toy_market(400)
    features = build_cost_basis_features(market, half_life=288)

    # The expansion bar at index 320 may decide attribution with delayed/prior
    # information, but its own newly opened OI cannot be in cost-basis state
    # until the next bar.
    assert features.loc[320, "long_weight"] == 0.0
    assert np.isnan(features.loc[320, "long_basis"])
    assert features.loc[321, "long_weight"] > 0.0
    assert np.isclose(features.loc[321, "long_basis"], np.log(market.loc[320, "close"]))


def test_source_delay_and_prefix_invariance() -> None:
    market = _toy_market(380)
    metrics = _toy_metrics(380)
    prefix_len = 340

    attach_market = _attachable_market(market)
    attached_full = _attach_delayed_metrics(attach_market, metrics, tolerance="5min", delay_bars=1)
    attached_prefix = _attach_delayed_metrics(attach_market.iloc[:prefix_len], metrics.iloc[:prefix_len], tolerance="5min", delay_bars=1)

    pd.testing.assert_series_equal(
        attached_full.loc[: prefix_len - 1, "positioning_source_time"].reset_index(drop=True),
        attached_prefix["positioning_source_time"].reset_index(drop=True),
        check_names=False,
    )
    assert attached_full.loc[25, "positioning_source_time"] == attached_full.loc[25, "date"] - pd.Timedelta("5min")

    full_features = build_cost_basis_features(attached_full, half_life=288)
    prefix_features = build_cost_basis_features(attached_prefix, half_life=288)
    pd.testing.assert_frame_equal(
        full_features.iloc[:prefix_len].reset_index(drop=True),
        prefix_features.reset_index(drop=True),
        check_exact=False,
        atol=0.0,
        rtol=0.0,
    )


def test_deterministic_signal_and_manifest_hash() -> None:
    score = np.array([np.nan, -3.0, -4.0, 0.0, 3.0, 4.0, 2.0, -5.0])
    long_signal, short_signal = build_signals(score, lower=-2.5, upper=2.5, mode="both")
    np.testing.assert_array_equal(long_signal, [False, False, False, False, True, False, False, False])
    np.testing.assert_array_equal(short_signal, [False, True, False, False, False, False, False, True])

    row = {
        "name": "candidate",
        "half_life": 288,
        "tail": 0.05,
        "lower": -1.25,
        "upper": 1.5,
        "mode": "long_only",
        "hold": 144,
    }
    identity = _candidate_identity(row) | {"name": row["name"]}
    assert _stable_json_hash(identity) == _stable_json_hash(json.loads(json.dumps(identity, sort_keys=True)))
    assert _stable_json_hash(identity) == "f0de15835e9524fc4d8aa3d5c9dece5842be439cbadd3d12b812ec7705b5e5e6"


def test_pre2024_loader_excludes_future_rows(tmp_path: Path) -> None:
    market = _toy_market(6, start="2023-12-31 23:40")
    metrics = _toy_metrics(6, start="2023-12-31 23:40")
    market_path = tmp_path / "market.csv"
    metrics_path = tmp_path / "metrics.csv"
    _attachable_market(market).to_csv(market_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    cfg = replace(OiCostBasisConfig(), input_csv=str(market_path), metrics_csv=str(metrics_path), selection_end="2024-01-01")
    loaded = load_pre2024_market(cfg)

    assert pd.to_datetime(loaded["date"]).max() < pd.Timestamp("2024-01-01")
    assert pd.to_datetime(loaded["positioning_source_time"], errors="coerce").max() < pd.Timestamp("2024-01-01")
