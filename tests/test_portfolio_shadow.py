import json
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from execution.portfolio_shadow import PortfolioShadowConfig, build_shadow_report, score_shadow_once
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


def test_shadow_report_never_enables_orders_and_rank7_bridge_is_ready_but_data_fails_closed():
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
    assert report["live_promotion_ready"] is False
    assert report["complete_portfolio_runtime_ready"] is True
    assert report["runtime_blocked_sleeves"] == []
    assert report["signal_scoring_ready_count"] == 5
    assert set(report["signal_scoring_ready_sleeves"]) == {
        "fresh_kimchi_fx",
        "rex_taker_low_range_position",
        "cand_rex_veto_7",
        "markov_transition_long",
        "frozen_annual_rank7",
    }
    rank7 = next(row for row in report["scores"] if row["name"] == "frozen_annual_rank7")
    assert rank7["active"] is False
    assert any(reason.startswith("runtime_bridge=error:Rank7FeatureError") for reason in rank7["reasons"])
    assert "rank7_fail_closed=pass" in rank7["reasons"]


def test_shadow_report_blocks_rank7_bundle_contract_errors():
    portfolio = json.loads(
        open("configs/live/portfolio_added_alpha_shadow_candidate_2026-07-16.json").read()
    )
    market, features = _frames()
    scores = [
        {
            "name": "frozen_annual_rank7",
            "reasons": [
                "runtime_bridge=error:Rank7BundleError:manifest checksum mismatch",
                "rank7_fail_closed=pass",
            ],
        }
    ]
    with patch("execution.portfolio_shadow._score_sleeves", return_value=scores):
        report = build_shadow_report(
            portfolio=portfolio,
            enriched=market,
            features=features,
            execution_cfg=WaveExecutionConfig(),
            decision_asof=pd.Timestamp(market.iloc[-1]["date"], tz="UTC"),
        )

    assert report["complete_portfolio_runtime_ready"] is False
    assert report["runtime_blocked_sleeves"] == ["frozen_annual_rank7"]
    assert report["signal_scoring_ready_sleeves"] == []


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


def test_rex_taker_is_inactive_before_frozen_active_from():
    portfolio = json.loads(
        open("configs/live/portfolio_added_alpha_shadow_candidate_2026-07-16.json").read()
    )
    portfolio["base_sleeves"] = [
        row for row in portfolio["base_sleeves"] if row["name"] == "rex_taker_low_range_position"
    ]
    market, features = _frames()
    market["date"] = pd.date_range("2020-01-01", periods=len(market), freq="5min")
    report = build_shadow_report(
        portfolio=portfolio,
        enriched=market,
        features=features,
        execution_cfg=WaveExecutionConfig(),
        decision_asof=pd.Timestamp(market.iloc[-1]["date"], tz="UTC"),
    )
    score = report["scores"][0]
    assert score["active"] is False
    assert "active_from=2021-01-01T00:00:00:fail" in score["reasons"]


def test_shadow_runner_rejects_lookback_shorter_than_feature_contract():
    with pytest.raises(RuntimeError, match="shorter than the portfolio feature-history contract"):
        asyncio.run(
            score_shadow_once(
                PortfolioShadowConfig(
                    portfolio_config=Path(
                        "configs/live/portfolio_added_alpha_shadow_candidate_2026-07-16.json"
                    ),
                    output=None,
                    lookback_minutes=45_000,
                )
            )
        )
