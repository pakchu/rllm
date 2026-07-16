import json

import numpy as np
import pandas as pd
import pytest

from execution.portfolio_shadow import build_shadow_report
from execution.wave_execution import WaveExecutionConfig
from preprocessing.market_features import build_market_feature_frame


def _frames(rows: int = 2_500):
    idx = np.arange(rows, dtype=float)
    close = 30_000.0 * np.exp(0.00001 * idx)
    quote = np.full(rows, 1_000_000.0)
    market = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="5min"),
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 10.0,
            "quote_asset_volume": quote,
            "number_of_trades": 100.0,
            "taker_buy_base": 5.0,
            "taker_buy_quote": quote * 0.5,
            "funding_rate": 0.0,
            "funding_available": 1.0,
            "premium_index": 0.0,
            "premium_index_change": 0.0,
            "premium_available": 1.0,
            "kimchi_premium": 0.02,
            "kimchi_available": 1.0,
            "usdkrw": 1_350.0,
            "usdkrw_available": 1.0,
            "dxy_available": 1.0,
            "external_any_available": 1.0,
            "binance_aux_any_available": 1.0,
            "open_interest": 1_000_000.0,
            "open_interest_available": 1.0,
        }
    )
    features = build_market_feature_frame(market, window_size=288)
    features["open_interest_available"] = 1.0
    return market, features


def test_shadow_report_never_enables_orders_and_rank7_fails_closed():
    portfolio = json.loads(
        open("configs/live/portfolio_added_alpha_shadow_candidate_2026-07-16.json").read()
    )
    market, features = _frames()
    report = build_shadow_report(
        portfolio=portfolio,
        enriched=market,
        features=features,
        execution_cfg=WaveExecutionConfig(),
        decision_asof=pd.Timestamp(market.iloc[-1]["date"], tz="UTC"),
    )
    assert report["orders_enabled"] is False
    assert report["complete_portfolio_runtime_ready"] is False
    assert report["runtime_blocked_sleeves"] == ["frozen_annual_rank7"]
    rank7 = next(row for row in report["scores"] if row["name"] == "frozen_annual_rank7")
    assert rank7["active"] is False
    assert "runtime_bridge=missing:annual_extratrees_model_export_required" in rank7["reasons"]


def test_shadow_report_rejects_non_shadow_portfolio():
    portfolio = json.loads(
        open("configs/live/portfolio_gross385_trainmdd40_2026-07-12.json").read()
    )
    market, features = _frames()
    with pytest.raises(RuntimeError, match="shadow_only=true"):
        build_shadow_report(
            portfolio=portfolio,
            enriched=market,
            features=features,
            execution_cfg=WaveExecutionConfig(),
            decision_asof=pd.Timestamp(market.iloc[-1]["date"], tz="UTC"),
        )
