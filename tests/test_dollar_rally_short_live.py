import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from execution import dollar_rally_short as signals
from execution.portfolio_live import (
    AlphaProcessManager,
    _build_open_intents,
    _execute_open_intents,
    _infer_signal_id_from_digest,
    _known_portfolio_sleeves,
    _load_sleeve_runtime_spec,
    _open_sleeve,
    _portfolio_has_barrier_exits,
    _score_sleeves,
    _validate_portfolio_mode,
)
from execution.signed_net_hedge import SIGNED_NET_HEDGE_MODE, parse_policy
from execution.wave_execution import WaveExecutionConfig
from preprocessing.market_features import build_market_feature_frame

RUNTIME_CONFIG = Path("configs/live/dollar_rally_short_runtime_2026-09-07.json")
RECOVERY_PORTFOLIO = Path(
    "configs/live/portfolio_dollar_rally_short_runtime_ready_2026-09-07.json"
)


def test_dollar_short_recovery_portfolio_opts_into_signed_net_contract():
    portfolio = json.loads(RECOVERY_PORTFOLIO.read_text())
    policy = parse_policy(portfolio)

    assert policy is not None
    assert policy.mode == SIGNED_NET_HEDGE_MODE
    assert float(policy.net_cap_after_fees) == pytest.approx(4.5)
    assert float(policy.weights["dollar_rally_short"]) == pytest.approx(0.5)


def _legacy_market(periods: int = 7_000) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01 00:00:00Z", periods=periods, freq="5min")
    steps = np.arange(periods, dtype=float)
    close = 10_000.0 * (1.0 + 0.00002 * steps)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
            "quote_asset_volume": close,
            "taker_buy_base": 0.5,
            "number_of_trades": 1.0,
            "dxy_momentum": 0.003,
            "dxy_available": 1.0,
        }
    )


def _features_at(
    market: pd.DataFrame, decision_time: str, *, window_size: int = 288
) -> tuple[pd.DataFrame, pd.Series]:
    decision = pd.Timestamp(decision_time)
    history = market.loc[market["date"] <= decision].reset_index(drop=True)
    features = build_market_feature_frame(
        history.assign(date=history["date"].dt.tz_convert(None)),
        window_size=window_size,
    )
    return history, features.iloc[-1]


def test_dollar_short_adapter_preserves_frozen_phase_gates_and_lifecycle():
    history, _ = _features_at(_legacy_market(), "2020-01-22 23:55:00Z")
    result = signals.score_dollar_short(history, "2020-01-22 23:55:00Z")

    assert result["active"] is True
    assert result["position"] == -1.0
    assert result["side"] == "SHORT"
    assert result["execution_time"] == "2020-01-23T00:00:00+00:00"
    assert result["lifecycle"]["exit_time"] == "2020-01-23T12:00:00+00:00"
    assert result["lifecycle"]["take_profit"] is None
    assert result["lifecycle"]["stop_loss"] is None
    assert result["global_phase"]["matched"] is True


def test_dollar_short_features_are_stable_between_frozen_and_live_windows():
    market = _legacy_market()
    history, frozen = _features_at(
        market,
        "2020-01-22 23:55:00Z",
        window_size=signals.DOLLAR_SHORT_WINDOW_SIZE,
    )
    _, live = _features_at(market, "2020-01-22 23:55:00Z", window_size=288)
    rebuilt = signals._original_144_features(history).iloc[-1]

    assert rebuilt["dxy_momentum"] == pytest.approx(frozen["dxy_momentum"])
    assert rebuilt["htf_1d_return_4"] == pytest.approx(frozen["htf_1d_return_4"])
    assert live["dxy_momentum"] == pytest.approx(frozen["dxy_momentum"])
    assert live["htf_1d_return_4"] == pytest.approx(frozen["htf_1d_return_4"])


def test_dollar_short_adapter_fails_closed_off_phase_or_without_dxy():
    market = _legacy_market()
    off_history, _ = _features_at(market, "2020-01-22 23:50:00Z")
    off_phase = signals.score_dollar_short(off_history, "2020-01-22 23:50:00Z")
    assert off_phase["active"] is False
    assert "off_global_phase" in off_phase["reason"]

    unavailable, _ = _features_at(market, "2020-01-22 23:55:00Z")
    unavailable.loc[unavailable.index[-1], "dxy_available"] = 0.0
    blocked = signals.score_dollar_short(unavailable, "2020-01-22 23:55:00Z")
    assert blocked["active"] is False
    assert "dxy_unavailable_at_signal" in blocked["reason"]

    gapped = unavailable.drop(index=unavailable.index[-100]).reset_index(drop=True)
    with pytest.raises(ValueError, match="Market grid gap"):
        signals.score_dollar_short(gapped, "2020-01-22 23:55:00Z")


def test_portfolio_live_builds_executable_half_notional_short_intent():
    decision_time = pd.Timestamp("2020-01-22 23:55:00Z")
    market = _legacy_market()
    market = market.loc[market["date"] <= decision_time].reset_index(drop=True)
    market["date"] = market["date"].dt.tz_convert(None)
    features = build_market_feature_frame(
        market,
        window_size=288,
    )
    sleeve = {
        "name": "dollar_rally_short",
        "source": str(RUNTIME_CONFIG),
        "side": "SHORT",
        "weight": 0.5,
    }
    score = _score_sleeves(
        portfolio={"base_sleeves": [sleeve]},
        enriched=market,
        features=features,
        exec_cfg=WaveExecutionConfig(leverage=8),
        asof=decision_time + pd.Timedelta(minutes=5),
    )[0]

    assert score["active"] is True
    assert score["side"] == "SHORT"
    assert score["hold_bars"] == 144
    assert score["stride_bars"] == 12
    assert score["stride_offset_bars"] == 11
    assert score["dynamic_exit"] is None
    assert score["barrier_exit"] is None
    assert score["policy_metadata"]["execution_time"] == "2020-01-23T00:00:00+00:00"
    assert "runtime_bridge=ready:dollar-rally-short-v1" in score["reasons"]

    digest = hashlib.sha1(score["signal_id"].encode()).hexdigest()[:8]
    recovered_signal, recovered_time = _infer_signal_id_from_digest(
        sleeve_name="dollar_rally_short",
        digest=digest,
        order_time_ms=int(pd.Timestamp("2020-01-23T00:00:10Z").timestamp() * 1_000),
        interval_minutes=5,
    )
    assert recovered_signal == score["signal_id"]
    assert recovered_time == pd.Timestamp("2020-01-22T23:55:00Z")

    intents = _build_open_intents(
        sleeve_scores=[score],
        state={"open_sleeves": {}, "processed_signals": {}},
        total_weight=0.5,
        leverage_budget=8.0,
        allocation_mode="research_gross",
        exec_cfg=WaveExecutionConfig(leverage=8),
        entry_timeout_fraction=0.25,
        max_entry_wait_sec=300,
        entry_maker_max_deviation_pct=0.003,
        maker_refresh_interval_sec=60,
        now=pd.Timestamp("2020-01-23T00:00:00Z"),
    )
    assert len(intents) == 1
    assert intents[0]["sleeve"]["side"] == "SHORT"
    assert intents[0]["margin_fraction"] == pytest.approx(0.5 / 8.0)
    assert intents[0]["entry_ttl_sec"] == 300

    expired_intents = _build_open_intents(
        sleeve_scores=[score],
        state={"open_sleeves": {}, "processed_signals": {}},
        total_weight=0.5,
        leverage_budget=8.0,
        allocation_mode="research_gross",
        exec_cfg=WaveExecutionConfig(leverage=8),
        entry_timeout_fraction=0.25,
        max_entry_wait_sec=300,
        entry_maker_max_deviation_pct=0.003,
        maker_refresh_interval_sec=60,
        now=pd.Timestamp("2020-01-23T00:05:00Z"),
    )
    assert expired_intents == []

    outcomes = asyncio.run(
        _execute_open_intents(
            intents=intents,
            client=None,
            executor=None,
            exec_cfg=WaveExecutionConfig(dry_run=True, leverage=8),
        )
    )
    assert outcomes[0]["ok"] is True
    assert outcomes[0]["order_status"] == "DRY_RUN"
    assert float(outcomes[0]["filled_quantity"]) > 0.0

    class FakeLiveClient:
        async def get_usdt_balance(self):
            return {"total": 100.0}

        async def get_ticker_price(self, symbol):
            return 10_000.0

    place_order = AsyncMock(
        return_value={"status": "FILLED", "filled_quantity": "0.005"}
    )
    attach_report = AsyncMock(side_effect=lambda **kwargs: kwargs["order_info"])
    with (
        patch(
            "execution.portfolio_live._place_portfolio_maker_order_with_deadline",
            new=place_order,
        ),
        patch(
            "execution.portfolio_live._attach_exchange_trade_report",
            new=attach_report,
        ),
    ):
        live_result = asyncio.run(
            _open_sleeve(
                client=FakeLiveClient(),
                executor=SimpleNamespace(),
                exec_cfg=WaveExecutionConfig(dry_run=False, leverage=8),
                sleeve=intents[0]["sleeve"],
                margin_fraction=intents[0]["margin_fraction"],
                entry_ttl_sec=intents[0]["entry_ttl_sec"],
            )
        )
    submitted = place_order.await_args.kwargs
    assert submitted["order_side"] == "SELL"
    assert submitted["position_side"] == "SHORT"
    assert float(submitted["quantity"]) == pytest.approx(0.005)
    assert live_result["notional"] == pytest.approx(50.0)


def test_dollar_short_runtime_contract_rejects_side_or_lifecycle_drift():
    valid = {
        "name": "dollar_rally_short",
        "source": str(RUNTIME_CONFIG),
        "side": "SHORT",
        "weight": 0.5,
    }
    spec = _load_sleeve_runtime_spec(valid)
    assert spec["hold_bars"] == 144
    assert spec["stride_bars"] == 12

    wrong_side = _load_sleeve_runtime_spec({**valid, "side": "LONG"})
    assert "configured sleeve side" in wrong_side["runtime_contract_error"]
    assert wrong_side["hold_bars"] == 144
    assert wrong_side["barrier_exit"] is None
    with pytest.raises(signals.DollarShortContractError, match="side"):
        signals.validate_dollar_short_runtime_config(
            json.loads(RUNTIME_CONFIG.read_text()), configured_side="LONG"
        )

    with tempfile.TemporaryDirectory() as tmp:
        changed = json.loads(RUNTIME_CONFIG.read_text())
        changed["hold_bars_5m"] = 143
        source = Path(tmp) / "changed.json"
        source.write_text(json.dumps(changed))
        drifted = _load_sleeve_runtime_spec({**valid, "source": str(source)})
        assert "hold_bars_5m" in drifted["runtime_contract_error"]
        assert drifted["hold_bars"] == 144
        assert drifted["dynamic_exit"] is None
        with pytest.raises(signals.DollarShortContractError, match="hold_bars_5m"):
            signals.validate_dollar_short_runtime_config(changed)

    missing = _load_sleeve_runtime_spec(
        {**valid, "source": "/missing/dollar-short-runtime.json"}
    )
    assert "is missing" in missing["runtime_contract_error"]
    assert missing["hold_bars"] == 144
    assert missing["stride_bars"] == 12
    assert (
        _portfolio_has_barrier_exits(
            {
                "base_sleeves": [
                    {**valid, "source": "/missing/dollar-short-runtime.json"}
                ]
            }
        )
        is False
    )


def test_dollar_short_recovery_registry_is_discoverable_but_not_live_authorized():
    portfolio = json.loads(RECOVERY_PORTFOLIO.read_text())

    _validate_portfolio_mode(portfolio, live=False)
    with pytest.raises(RuntimeError, match="not authorized for live orders"):
        _validate_portfolio_mode(portfolio, live=True)

    recovered = _known_portfolio_sleeves(portfolio)["dollar_rally_short"]
    assert recovered["side"] == "SHORT"
    assert recovered["weight"] == pytest.approx(0.5)
    assert recovered["hold_bars"] == 144
    assert recovered["stride_bars"] == 12
    assert recovered["runtime_contract_error"] is None


def test_dollar_short_scores_in_the_live_parallel_worker_path():
    async def run():
        decision_time = pd.Timestamp("2020-01-22 23:55:00Z")
        market = _legacy_market()
        market = market.loc[market["date"] <= decision_time].reset_index(drop=True)
        market["date"] = market["date"].dt.tz_convert(None)
        features = build_market_feature_frame(market, window_size=288)
        sleeve = {
            "name": "dollar_rally_short",
            "source": str(RUNTIME_CONFIG),
            "side": "SHORT",
            "weight": 0.5,
        }
        manager = AlphaProcessManager([sleeve], timeout_sec=30.0)
        try:
            return (
                await manager.score(
                    portfolio={"base_sleeves": [sleeve]},
                    enriched=market,
                    features=features,
                    exec_cfg=WaveExecutionConfig(leverage=8),
                    asof=decision_time + pd.Timedelta(minutes=5),
                )
            )[0]
        finally:
            await manager.shutdown()

    score = asyncio.run(run())
    assert score["active"] is True
    assert score["side"] == "SHORT"
    assert score["scoring_mode"] == "process"
    assert score["worker_pid"] != os.getpid()
