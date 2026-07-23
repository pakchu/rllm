from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from execution import portfolio_live as pl
from execution.binance_aggtrade_stream import AggTradeTick
from execution.wave_execution import WaveExecutionConfig


def _barrier(take_bps: int = 400, stop_bps: int = 250) -> dict[str, object]:
    barrier = pl._barrier_exit_from_config({"take_bps": take_bps, "stop_bps": stop_bps})
    assert barrier is not None
    return barrier


def _reason_kind(reason: str | None) -> str | None:
    return {"take": "take_profit", "stop": "stop_loss"}.get(reason, reason)


def _assert_fixed_400_250_barrier(barrier: dict[str, object] | None) -> None:
    assert barrier is not None
    assert barrier["take_bps"] == 400
    assert barrier["stop_bps"] == 250
    assert barrier["entry_execution"] == "market"
    assert barrier.get("same_bar_policy", barrier.get("same_bar_priority")) == "stop_before_take"


def _open_state(*, side: str = "LONG", entry_price: float = 100.0, filled_at: str = "2026-07-16T00:05:00Z") -> dict[str, object]:
    return {
        "name": "fresh_kimchi_fx",
        "side": side,
        "signal_id": f"fresh_kimchi_fx:{side}:2026-07-16T00:00:00Z",
        "barrier_exit": _barrier(),
        "entry_fill_price": entry_price,
        "entry_filled_at": filled_at,
    }


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-07-16T00:00:00Z", periods=5, freq="5min"),
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [120.0, 120.0, 103.0, 104.0, 100.0],
            "low": [80.0, 80.0, 98.0, 99.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )


def test_barrier_exit_from_fresh_kimchi_config_uses_fixed_400_250_contract() -> None:
    cfg = json.loads(Path("configs/shadow/fresh_kimchi_fx_2026-07-16.json").read_text())

    barrier = pl._barrier_exit_from_config(cfg)

    _assert_fixed_400_250_barrier(barrier)


def test_portfolio_poll_cadence_cannot_be_slower_than_sleeve_contract() -> None:
    portfolio = {
        "base_sleeves": [
            {
                "name": "fresh_kimchi_fx",
                "source": "configs/shadow/fresh_kimchi_fx_2026-07-16.json",
                "side": "AUTO",
                "weight": 2.0,
            }
        ]
    }
    assert pl._portfolio_barrier_poll_sec(portfolio, configured_poll_sec=5.0) == 1.0


def test_barrier_exit_from_config_returns_none_when_config_has_no_barrier_contract() -> None:
    assert pl._barrier_exit_from_config({"name": "fresh_kimchi_fx"}) is None


@pytest.mark.parametrize(
    "barrier",
    [
        {"take_bps": 400},
        {"stop_bps": 250},
        {"take_bps": 0, "stop_bps": 250},
        {"take_bps": 400, "stop_bps": -1},
        {"take_bps": "not-numeric", "stop_bps": 250},
        {"type": "trailing_stop", "take_bps": 400, "stop_bps": 250},
    ],
)
def test_barrier_exit_due_at_price_fails_closed_for_invalid_contracts(barrier: dict[str, object]) -> None:
    due, reason, reasons = pl._barrier_exit_due_at_price({**_open_state(), "barrier_exit": barrier}, 50.0)

    assert due is False
    assert reason is None
    assert any("fail_closed" in item for item in reasons)


@pytest.mark.parametrize(
    ("side", "observed_price", "expected_reason"),
    [
        ("LONG", 104.0, "take_profit"),
        ("LONG", 97.5, "stop_loss"),
        ("SHORT", 96.0, "take_profit"),
        ("SHORT", 102.5, "stop_loss"),
    ],
)
def test_barrier_exit_due_at_price_handles_long_and_short_take_profit_and_stop_loss(
    side: str, observed_price: float, expected_reason: str
) -> None:
    due, reason, reasons = pl._barrier_exit_due_at_price(_open_state(side=side), observed_price)

    assert due is True
    assert _reason_kind(reason) == expected_reason
    assert any(_reason_kind(reason.split("=")[0].replace("barrier_", "")) == expected_reason for reason in reasons)


@pytest.mark.parametrize(("side", "observed_price"), [("LONG", 100.5), ("SHORT", 100.5)])
def test_barrier_exit_due_at_price_waits_inside_take_profit_and_stop_loss(side: str, observed_price: float) -> None:
    due, reason, reasons = pl._barrier_exit_due_at_price(_open_state(side=side), observed_price)

    assert due is False
    assert reason is None
    assert any("inside" in item or "wait" in item for item in reasons)


def test_barrier_exit_due_in_bars_uses_stop_before_take_when_same_bar_hits_both() -> None:
    bars = pd.DataFrame(
        {
            "date": pd.date_range("2026-07-16T00:05:00Z", periods=2, freq="5min"),
            "open": [100.0, 100.0],
            "high": [100.0, 105.0],
            "low": [100.0, 95.0],
            "close": [100.0, 100.0],
        }
    )

    due, reason, reasons, exit_at = pl._barrier_exit_due_in_bars(_open_state(), bars, interval_minutes=5)

    assert due is True
    assert _reason_kind(reason) == "stop_loss"
    assert exit_at == pd.Timestamp("2026-07-16T00:10:00Z")
    assert any("stop_before_take" in item for item in reasons)


def test_barrier_exit_due_in_bars_excludes_bars_before_entry_and_partial_entry_bar() -> None:
    # The first two bars contain huge ranges but are before the fill or contain
    # only a partial entry interval. The first eligible full bar after a 00:07
    # fill is 00:10, which hits the long take-profit barrier.
    due, reason, reasons, exit_at = pl._barrier_exit_due_in_bars(
        _open_state(filled_at="2026-07-16T00:07:00Z"),
        _bars(),
        interval_minutes=5,
    )

    assert due is True
    assert _reason_kind(reason) == "take_profit"
    assert exit_at == pd.Timestamp("2026-07-16T00:15:00Z")


def test_barrier_exit_due_in_bars_uses_actual_average_fill_price_not_signal_fallback() -> None:
    state = {
        **_open_state(entry_price=110.0, filled_at="2026-07-16T00:05:00Z"),
        "reference_price": 100.0,
        "current_close": 100.0,
    }
    bars = pd.DataFrame(
        {
            "date": pd.date_range("2026-07-16T00:05:00Z", periods=2, freq="5min"),
            "open": [100.0, 100.0],
            "high": [110.0, 111.0],
            "low": [110.0, 108.0],
            "close": [110.0, 109.0],
        }
    )

    due, reason, reasons, exit_at = pl._barrier_exit_due_in_bars(state, bars, interval_minutes=5)

    assert due is False
    assert reason is None
    assert exit_at is None
    assert any("inside" in item for item in reasons)


def test_entry_fill_metadata_prefers_actual_avg_price_and_fill_timestamp_over_fallback() -> None:
    order_info = {
        "avg_price": "101.25",
        "filled_quantity": "0.010",
        "finished_at": "2026-07-16T00:06:30Z",
        "started_at": "2026-07-16T00:05:02Z",
        "raw_order": {
            "updateTime": 1_784_160_301_000,
            "trade_report": {
                "first_fill_at": "2026-07-16T00:05:00.125Z",
                "last_fill_at": "2026-07-16T00:05:00.250Z",
            },
        },
    }

    metadata = pl._entry_fill_metadata(order_info, fallback_price=100.0)

    assert metadata["entry_fill_price"] == 101.25
    assert pd.Timestamp(metadata["entry_filled_at"]) == pd.Timestamp("2026-07-16T00:05:00.125Z")


def test_entry_fill_metadata_falls_back_when_avg_price_is_missing_or_zero() -> None:
    metadata = pl._entry_fill_metadata(
        {"avg_price": "0", "filled_quantity": "0.010", "started_at": "2026-07-16T00:05:02Z"},
        fallback_price=100.0,
    )

    assert metadata["entry_fill_price"] == 100.0
    assert pd.Timestamp(metadata["entry_filled_at"]) == pd.Timestamp("2026-07-16T00:05:02Z")


def test_load_sleeve_runtime_spec_recovers_barrier_exit_from_source_config(tmp_path: Path) -> None:
    source = tmp_path / "fresh_kimchi_fx.json"
    source.write_text(
        json.dumps(
            {
                "name": "fresh_kimchi_fx",
                "policy_type": "bidirectional_gate",
                "hold_bars": 288,
                "stride_bars": 6,
                "take_bps": 400,
                "stop_bps": 250,
            }
        )
    )

    spec = pl._load_sleeve_runtime_spec({"name": "fresh_kimchi_fx", "source": str(source)})

    assert spec["hold_bars"] == 288
    assert spec["stride_bars"] == 6
    _assert_fixed_400_250_barrier(spec.get("barrier_exit"))


def test_score_sleeves_includes_barrier_exit_for_active_fresh_kimchi_sleeve(tmp_path: Path) -> None:
    dates = pd.date_range("2026-07-16T00:25:00Z", periods=7, freq="5min")
    enriched = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * len(dates),
            "high": [101.0] * len(dates),
            "low": [99.0] * len(dates),
            "close": [100.0] * len(dates),
        }
    )
    features = pd.DataFrame(
        {
            "alpha_x": [2.0] * len(dates),
        }
    )
    source = tmp_path / "fresh_kimchi_fx.json"
    source.write_text(
        json.dumps(
            {
                "name": "fresh_kimchi_fx",
                "gates": [{"feature": "alpha_x", "op": ">=", "threshold": 1.0}],
                "hold_bars": 288,
                "stride_bars": 6,
                "stride_offset_bars": 5,
                "entry_delay_bars": 1,
                "take_bps": 400,
                "stop_bps": 250,
            }
        )
    )
    portfolio = {"base_sleeves": [{"name": "fresh_kimchi_fx", "source": str(source), "side": "LONG", "weight": 1.0}]}

    scores = pl._score_sleeves(
        portfolio=portfolio,
        enriched=enriched,
        features=features,
        exec_cfg=WaveExecutionConfig(),
        asof=dates[-1],
    )

    assert len(scores) == 1
    assert scores[0]["active"] is True
    assert scores[0]["side"] == "LONG"
    _assert_fixed_400_250_barrier(scores[0].get("barrier_exit"))


def test_invalid_barrier_contract_blocks_signal_before_order_intent(tmp_path: Path) -> None:
    dates = pd.date_range("2026-07-16T00:25:00Z", periods=7, freq="5min")
    enriched = pd.DataFrame({"date": dates, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    features = pd.DataFrame({"alpha_x": [2.0] * len(dates)})
    source = tmp_path / "invalid_barrier.json"
    source.write_text(
        json.dumps(
            {
                "gates": [{"feature": "alpha_x", "op": ">=", "threshold": 1.0}],
                "hold_bars": 12,
                "stride_bars": 1,
                "entry_delay_bars": 1,
                "barrier_exit": {"take_bps": 400, "stop_bps": 250, "entry_execution": "maker"},
            }
        )
    )
    scores = pl._score_sleeves(
        portfolio={"base_sleeves": [{"name": "invalid", "source": str(source), "side": "LONG", "weight": 1.0}]},
        enriched=enriched,
        features=features,
        exec_cfg=WaveExecutionConfig(),
        asof=dates[-1],
    )
    assert scores[0]["active"] is False
    assert any("entry_execution_must_be_market" in reason for reason in scores[0]["reasons"])



def test_execute_close_intents_uses_market_order_for_barrier_exit_due() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.market_calls: list[dict[str, object]] = []
            self.ticker_calls = 0

        async def place_order(self, **kwargs):
            self.market_calls.append(kwargs)
            return {
                "orderId": 9001,
                "clientOrderId": kwargs["client_order_id"],
                "status": "FILLED",
                "executedQty": "0.010",
                "avgPrice": "97.50",
            }

        async def get_trades(self, symbol: str, limit: int = 1000):
            return [
                {
                    "orderId": 9001,
                    "qty": "0.010",
                    "price": "97.50",
                    "quoteQty": "0.975",
                    "realizedPnl": "0",
                    "commission": "0",
                    "commissionAsset": "USDT",
                    "time": 1_784_160_000_000,
                }
            ]

        async def get_ticker_price(self, symbol: str):
            self.ticker_calls += 1
            raise AssertionError("barrier market close must not request ticker price")

    class FailingExecutor:
        async def get_maker_price(self, *args, **kwargs):
            raise AssertionError("barrier market close must not use maker executor")

    async def run() -> None:
        client = FakeClient()
        outcomes = await pl._execute_close_intents(
            intents=[
                {
                    "key": "fresh_kimchi_fx",
                    "open_state": {
                        "name": "fresh_kimchi_fx",
                        "signal_id": "fresh_kimchi_fx:LONG:2026-07-16T00:00:00Z",
                        "side": "LONG",
                        "quantity": "0.010",
                    },
                    "barrier_exit_due": True,
                    "barrier_reason": "stop",
                    "time_exit_due": False,
                    "dynamic_exit_due": False,
                }
            ],
            client=client,
            executor=FailingExecutor(),
            exec_cfg=WaveExecutionConfig(dry_run=False, allow_live_orders=True),
            max_exit_wait_sec=30,
            exit_maker_max_deviation_pct=0.002,
            maker_refresh_interval_sec=60,
        )
        assert len(outcomes) == 1
        assert outcomes[0]["ok"] is True
        assert outcomes[0]["fully_closed"] is True
        assert outcomes[0]["filled_quantity"] == "0.010"
        assert len(client.market_calls) == 1
        assert client.market_calls[0]["side"] == "SELL"
        assert client.market_calls[0]["order_type"] == "MARKET"
        assert client.market_calls[0]["position_side"] == "LONG"
        assert "reduce_only" not in client.market_calls[0]
        assert client.ticker_calls == 0
        assert outcomes[0]["close_info"]["barrier_market_exit"] is True

    asyncio.run(run())


def test_execute_open_intents_uses_atomic_market_entry_for_barrier_sleeve() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.orders: list[dict[str, object]] = []

        async def get_usdt_balance(self):
            return {"total": 100.0}

        async def get_ticker_price(self, symbol: str):
            return 100.0

        async def get_open_orders(self, symbol: str):
            return []

        async def place_order(self, **kwargs):
            self.orders.append(kwargs)
            return {
                "orderId": 9200,
                "clientOrderId": kwargs["client_order_id"],
                "status": "FILLED",
                "executedQty": "0.4",
                "avgPrice": "100.25",
                "updateTime": 1_784_160_000_125,
            }

    class FailingExecutor:
        async def get_maker_price(self, *args, **kwargs):
            raise AssertionError("barrier entry must not wait on maker fills")

    async def run() -> None:
        client = FakeClient()
        outcomes = await pl._execute_open_intents(
            intents=[
                {
                    "sleeve": {
                        "name": "fresh_kimchi_fx",
                        "signal_id": "fresh_kimchi_fx:LONG:2026-07-16T00:00:00Z",
                        "side": "LONG",
                        "current_close": 100.0,
                        "barrier_exit": _barrier(),
                    },
                    "margin_fraction": 0.1,
                    "entry_ttl_sec": 300,
                }
            ],
            client=client,
            executor=FailingExecutor(),
            exec_cfg=WaveExecutionConfig(
                dry_run=False,
                allow_live_orders=True,
                leverage=4,
            ),
        )
        assert len(outcomes) == 1
        assert outcomes[0]["ok"] is True
        assert outcomes[0]["filled_quantity"] == "0.4"
        assert outcomes[0]["order_info"]["order"]["avg_price"] == "100.25"
        assert outcomes[0]["order_info"]["order"]["barrier_market_entry"] is True
        assert len(client.orders) == 1
        assert client.orders[0]["order_type"] == "MARKET"
        assert client.orders[0]["position_side"] == "LONG"
        assert "reduce_only" not in client.orders[0]
        assert client.orders[0]["new_order_resp_type"] == "RESULT"

    asyncio.run(run())


def test_legacy_wave_client_market_order_uses_signed_result_response() -> None:
    class LegacyClient:
        def __init__(self) -> None:
            self.params: dict[str, object] | None = None

        async def place_order(
            self,
            symbol,
            side,
            order_type,
            quantity=None,
            reduce_only=False,
            client_order_id=None,
            position_side=None,
        ):
            raise AssertionError("legacy client must use signed RESULT transport")

        async def get_symbol_info(self, symbol):
            return {"filters": []}

        def _round_quantity(self, symbol_info, quantity):
            return str(quantity)

        async def _private_request(self, method, path, params):
            self.params = dict(params)
            return {"orderId": 1, "status": "FILLED", "executedQty": params["quantity"], "avgPrice": "100"}

    async def run() -> None:
        client = LegacyClient()
        result = await pl._place_market_result_order(
            client=client,
            symbol="BTCUSDT",
            side="BUY",
            quantity=pl.Decimal("0.1"),
            position_side="LONG",
            reduce_only=False,
            client_order_id="rpf_test",
        )
        assert result["status"] == "FILLED"
        assert client.params is not None
        assert client.params["newOrderRespType"] == "RESULT"
        assert client.params["positionSide"] == "LONG"
        assert "reduceOnly" not in client.params

    asyncio.run(run())


def test_ack_market_order_with_unconfirmed_fill_is_not_classified_as_missed() -> None:
    class AckClient:
        async def get_open_orders(self, symbol: str):
            return []

        async def get_usdt_balance(self):
            return {"total": 100.0}

        async def get_ticker_price(self, symbol: str):
            return 100.0

        async def place_order(self, **kwargs):
            return {"orderId": 9500, "clientOrderId": kwargs["client_order_id"], "status": "NEW"}

        async def get_order(self, symbol: str, order_id=None, client_order_id=None):
            raise RuntimeError("temporary order query outage")

        async def get_trades(self, symbol: str, limit: int = 1000):
            raise RuntimeError("temporary trade query outage")

    async def run() -> None:
        outcomes = await pl._execute_open_intents(
            intents=[
                {
                    "sleeve": {
                        "name": "fresh_kimchi_fx",
                        "signal_id": "fresh_kimchi_fx:LONG:2026-07-16T00:00:00Z",
                        "side": "LONG",
                        "current_close": 100.0,
                        "barrier_exit": _barrier(),
                    },
                    "margin_fraction": 0.1,
                    "entry_ttl_sec": 300,
                }
            ],
            client=AckClient(),
            executor=object(),
            exec_cfg=WaveExecutionConfig(dry_run=False, allow_live_orders=True, leverage=4),
        )
        assert outcomes[0]["ok"] is False
        assert outcomes[0]["execution_uncertain"] is True
        assert outcomes[0]["reservation_status"] == "UNCERTAIN"
        assert outcomes[0]["error"] == "market_order_execution_uncertain"

    asyncio.run(run())


def test_market_entry_post_and_idempotency_lookup_outage_is_uncertain() -> None:
    class OutageClient:
        async def get_open_orders(self, symbol: str):
            return []

        async def get_usdt_balance(self):
            return {"total": 100.0}

        async def get_ticker_price(self, symbol: str):
            return 100.0

        async def place_order(self, **kwargs):
            raise TimeoutError("POST response lost")

        async def get_order(self, symbol: str, order_id=None, client_order_id=None):
            raise RuntimeError("order lookup unavailable")

        async def get_trades(self, symbol: str, limit: int = 1000):
            raise RuntimeError("trade lookup unavailable")

    async def run() -> None:
        outcomes = await pl._execute_open_intents(
            intents=[
                {
                    "sleeve": {
                        "name": "fresh_kimchi_fx",
                        "signal_id": "fresh_kimchi_fx:LONG:2026-07-16T00:00:00Z",
                        "side": "LONG",
                        "current_close": 100.0,
                        "barrier_exit": _barrier(),
                    },
                    "margin_fraction": 0.1,
                    "entry_ttl_sec": 300,
                }
            ],
            client=OutageClient(),
            executor=object(),
            exec_cfg=WaveExecutionConfig(dry_run=False, allow_live_orders=True, leverage=4),
        )
        assert outcomes[0]["ok"] is False
        assert outcomes[0]["execution_uncertain"] is True
        raw = outcomes[0]["order_info"]["order"]["raw_order"]
        assert raw["status"] == "UNKNOWN"
        assert "POST response lost" in raw["placement_error"]
        assert "order lookup unavailable" in raw["idempotency_lookup_error"]
        assert raw["clientOrderId"].startswith("rpf_")

    asyncio.run(run())


def test_market_close_post_and_idempotency_lookup_outage_is_uncertain() -> None:
    class OutageClient:
        async def place_order(self, **kwargs):
            raise TimeoutError("close POST response lost")

        async def get_order(self, symbol: str, order_id=None, client_order_id=None):
            raise RuntimeError("close order lookup unavailable")

        async def get_trades(self, symbol: str, limit: int = 1000):
            raise RuntimeError("close trade lookup unavailable")

    async def run() -> None:
        outcomes = await pl._execute_close_intents(
            intents=[
                {
                    "key": "fresh_kimchi_fx",
                    "open_state": {
                        "name": "fresh_kimchi_fx",
                        "signal_id": "fresh_kimchi_fx:LONG:2026-07-16T00:00:00Z",
                        "side": "LONG",
                        "quantity": "0.010",
                    },
                    "barrier_exit_due": True,
                    "barrier_reason": "stop",
                    "time_exit_due": False,
                    "dynamic_exit_due": False,
                }
            ],
            client=OutageClient(),
            executor=object(),
            exec_cfg=WaveExecutionConfig(dry_run=False, allow_live_orders=True),
            max_exit_wait_sec=30,
            exit_maker_max_deviation_pct=0.002,
            maker_refresh_interval_sec=60,
        )
        assert outcomes[0]["ok"] is False
        assert outcomes[0]["execution_uncertain"] is True
        raw = outcomes[0]["close_info"]["raw_order"]
        assert "close POST response lost" in raw["placement_error"]
        assert "close order lookup unavailable" in raw["idempotency_lookup_error"]

    asyncio.run(run())


def test_filled_market_entry_without_fill_price_or_time_is_uncertain() -> None:
    class MissingFillDetailClient:
        async def get_open_orders(self, symbol: str):
            return []

        async def get_usdt_balance(self):
            return {"total": 100.0}

        async def get_ticker_price(self, symbol: str):
            return 100.0

        async def place_order(self, **kwargs):
            return {
                "orderId": 9600,
                "clientOrderId": kwargs["client_order_id"],
                "status": "FILLED",
                "executedQty": "0.4",
                "avgPrice": "0",
            }

        async def get_trades(self, symbol: str, limit: int = 1000):
            raise RuntimeError("fill details unavailable")

    async def run() -> None:
        outcomes = await pl._execute_open_intents(
            intents=[
                {
                    "sleeve": {
                        "name": "fresh_kimchi_fx",
                        "signal_id": "fresh_kimchi_fx:LONG:2026-07-16T00:00:00Z",
                        "side": "LONG",
                        "current_close": 100.0,
                        "barrier_exit": _barrier(),
                    },
                    "margin_fraction": 0.1,
                    "entry_ttl_sec": 300,
                }
            ],
            client=MissingFillDetailClient(),
            executor=object(),
            exec_cfg=WaveExecutionConfig(dry_run=False, allow_live_orders=True, leverage=4),
        )
        assert outcomes[0]["ok"] is False
        assert outcomes[0]["execution_uncertain"] is True
        assert outcomes[0]["filled_quantity"] == "0.4"
        assert outcomes[0]["order_info"]["order"]["avg_price"] == "0"

    asyncio.run(run())


def test_wait_with_barrier_monitor_removes_hit_sleeve_and_logs_one_close(tmp_path: Path) -> None:
    state_file = tmp_path / "portfolio_state.json"
    state_file.write_text(
        json.dumps(
            {
                "open_sleeves": {
                    "fresh_kimchi_fx": {
                        "name": "fresh_kimchi_fx",
                        "signal_id": "fresh_kimchi_fx:LONG:2026-07-16T00:00:00Z",
                        "side": "LONG",
                        "quantity": "0.010",
                        "barrier_exit": _barrier(),
                        "entry_fill_price": 100.0,
                        "entry_filled_at": "2026-07-16T00:05:00Z",
                    }
                },
                "processed_signals": {},
            }
        )
    )

    class FakeClient:
        def __init__(self) -> None:
            self.market_calls = 0

        async def get_ticker_price(self, symbol: str):
            return 97.5

        async def place_order(self, **kwargs):
            self.market_calls += 1
            return {
                "orderId": 9100,
                "clientOrderId": kwargs["client_order_id"],
                "status": "FILLED",
                "executedQty": "0.010",
                "avgPrice": "97.50",
            }

        async def get_trades(self, symbol: str, limit: int = 1000):
            return [
                {
                    "orderId": 9100,
                    "qty": "0.010",
                    "price": "97.50",
                    "quoteQty": "0.975",
                    "realizedPnl": "0",
                    "commission": "0",
                    "commissionAsset": "USDT",
                    "time": 1_784_160_000_000,
                }
            ]

    class FailingExecutor:
        async def get_maker_price(self, *args, **kwargs):
            raise AssertionError("barrier monitor must close via market path")

    async def run() -> None:
        client = FakeClient()
        logs: list[dict[str, object]] = []

        def fake_log_trade_execution(*args, **kwargs) -> None:
            logs.append(kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(pl, "_log_trade_execution", fake_log_trade_execution)
            closed = await pl._wait_with_barrier_monitor(
                wait_sec=0.01,
                poll_sec=0.01,
                state_file=state_file,
                client=client,
                executor=FailingExecutor(),
                exec_cfg=WaveExecutionConfig(dry_run=False, allow_live_orders=True),
                engine=object(),
                strategy_name="rllm",
                execution_exchange="binance-testnet",
                db_lease=None,
            )

        assert closed == ["fresh_kimchi_fx"]
        assert client.market_calls == 1
        state = json.loads(state_file.read_text())
        assert state["open_sleeves"] == {}
        assert len(state["barrier_exit_history"]) == 1
        assert state["barrier_exit_history"][0]["fully_closed"] is True
        assert len(logs) == 1
        assert logs[0]["action"] == "CLOSE"
        assert logs[0]["order_type"] == "MARKET_BARRIER_STOP"
        assert logs[0]["status"] == "FILLED"

    asyncio.run(run())


def test_aggtrade_monitor_drains_ticks_without_open_barrier_positions(tmp_path: Path) -> None:
    state_file = tmp_path / "portfolio_state.json"
    state_file.write_text(json.dumps({"open_sleeves": {}, "processed_signals": {}}))

    class FakeStream:
        def __init__(self) -> None:
            self.collect_calls = 0

        async def collect(self, *, timeout_sec: float):
            self.collect_calls += 1
            await asyncio.sleep(min(timeout_sec, 0.001))
            return [AggTradeTick(price=100.0, event_time_ms=self.collect_calls)]

    async def run() -> None:
        stream = FakeStream()
        closed = await pl._wait_with_barrier_monitor(
            wait_sec=0.01,
            poll_sec=0.005,
            state_file=state_file,
            client=object(),
            executor=object(),
            exec_cfg=WaveExecutionConfig(dry_run=False, allow_live_orders=True),
            engine=object(),
            strategy_name="rllm",
            execution_exchange="binance",
            db_lease=None,
            trade_stream=stream,
        )
        assert closed == []
        assert stream.collect_calls > 0

    asyncio.run(run())


def test_aggtrade_monitor_closes_on_first_observed_touch_in_order(tmp_path: Path) -> None:
    state_file = tmp_path / "portfolio_state.json"
    state = _open_state()
    state.update({"quantity": "0.010", "barrier_stream_session_id": "stream-1", "barrier_stream_gap_count": 0})
    state_file.write_text(json.dumps({"open_sleeves": {"fresh_kimchi_fx": state}, "processed_signals": {}}))

    class FakeStream:
        session_id = "stream-1"
        gap_count = 0
        healthy = True

        def __init__(self) -> None:
            self.sent = False

        async def collect(self, *, timeout_sec: float):
            if self.sent:
                await asyncio.sleep(timeout_sec)
                return []
            self.sent = True
            first = int(pd.Timestamp("2026-07-16T00:06:00Z").timestamp() * 1000)
            return [
                AggTradeTick(price=104.0, event_time_ms=first),
                AggTradeTick(price=97.5, event_time_ms=first + 1_000),
            ]

    class FakeClient:
        async def get_ticker_price(self, symbol: str):
            raise AssertionError("healthy aggTrade path must not poll REST ticker")

        async def place_order(self, **kwargs):
            return {
                "orderId": 9300,
                "clientOrderId": kwargs["client_order_id"],
                "status": "FILLED",
                "executedQty": "0.010",
                "avgPrice": "104.0",
            }

        async def get_trades(self, symbol: str, limit: int = 1000):
            return []

    async def run() -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(pl, "_log_trade_execution", lambda *args, **kwargs: None)
            closed = await pl._wait_with_barrier_monitor(
                wait_sec=0.01,
                poll_sec=0.01,
                state_file=state_file,
                client=FakeClient(),
                executor=object(),
                exec_cfg=WaveExecutionConfig(dry_run=False, allow_live_orders=True),
                engine=object(),
                strategy_name="rllm",
                execution_exchange="binance-testnet",
                db_lease=None,
                trade_stream=FakeStream(),
            )
        assert closed == ["fresh_kimchi_fx"]
        persisted = json.loads(state_file.read_text())
        assert persisted["barrier_exit_history"][0]["reason"] == "take"

    asyncio.run(run())


def test_aggtrade_gap_forces_market_close_even_inside_barrier(tmp_path: Path) -> None:
    state_file = tmp_path / "portfolio_state.json"
    state = _open_state()
    state.update({"quantity": "0.010", "barrier_stream_session_id": "old-stream", "barrier_stream_gap_count": 0})
    state_file.write_text(json.dumps({"open_sleeves": {"fresh_kimchi_fx": state}, "processed_signals": {}}))

    class GapStream:
        session_id = "new-stream"
        gap_count = 0
        healthy = True

        async def collect(self, *, timeout_sec: float):
            raise AssertionError("continuity failure must close before consuming ticks")

    class FakeClient:
        async def get_ticker_price(self, symbol: str):
            return 100.0

        async def place_order(self, **kwargs):
            return {
                "orderId": 9400,
                "clientOrderId": kwargs["client_order_id"],
                "status": "FILLED",
                "executedQty": "0.010",
                "avgPrice": "100.0",
            }

        async def get_trades(self, symbol: str, limit: int = 1000):
            return []

    async def run() -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(pl, "_log_trade_execution", lambda *args, **kwargs: None)
            closed = await pl._wait_with_barrier_monitor(
                wait_sec=0.01,
                poll_sec=0.01,
                state_file=state_file,
                client=FakeClient(),
                executor=object(),
                exec_cfg=WaveExecutionConfig(dry_run=False, allow_live_orders=True),
                engine=object(),
                strategy_name="rllm",
                execution_exchange="binance-testnet",
                db_lease=None,
                trade_stream=GapStream(),
            )
        assert closed == ["fresh_kimchi_fx"]
        persisted = json.loads(state_file.read_text())
        assert persisted["barrier_exit_history"][0]["reason"] == "stream_gap_fail_safe"

    asyncio.run(run())
