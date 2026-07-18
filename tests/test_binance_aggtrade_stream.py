from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from execution.binance_aggtrade_stream import AggTradeTick, BinanceAggTradeStream


def test_stream_selects_network_url_and_preserves_tick_order() -> None:
    async def run() -> None:
        mainnet = BinanceAggTradeStream(symbol="BTCUSDT", testnet=False)
        testnet = BinanceAggTradeStream(symbol="BTCUSDT", testnet=True)
        assert mainnet.url == "wss://fstream.binance.com/market/ws/btcusdt@aggTrade"
        assert testnet.url == "wss://demo-fstream.binance.com/market/ws/btcusdt@aggTrade"
        first = AggTradeTick(price=100.0, event_time_ms=1, aggregate_trade_id=10)
        second = AggTradeTick(price=101.0, event_time_ms=2, aggregate_trade_id=11)
        mainnet.queue.put_nowait(first)
        mainnet.queue.put_nowait(second)
        assert await mainnet.collect(timeout_sec=0.01) == [first, second]

    asyncio.run(run())


def test_stream_health_requires_connected_lossless_session() -> None:
    stream = BinanceAggTradeStream(symbol="BTCUSDT", testnet=False, stale_after_sec=5.0)
    assert stream.healthy is False
    stream.connected.set()
    assert stream.healthy is False
    stream.ready.set()
    stream.last_message_monotonic = 100.0
    with patch("execution.binance_aggtrade_stream.time.monotonic", return_value=104.0):
        assert stream.healthy is True
    with patch("execution.binance_aggtrade_stream.time.monotonic", return_value=106.0):
        assert stream.healthy is False
    stream.last_message_monotonic = 200.0
    with patch("execution.binance_aggtrade_stream.time.monotonic", return_value=201.0):
        assert stream.healthy is True
    stream.overflowed = True
    assert stream.healthy is False


def test_start_rejects_handshake_without_first_market_event() -> None:
    async def run() -> None:
        stream = BinanceAggTradeStream(symbol="BTCUSDT", testnet=False)

        async def connected_without_data() -> None:
            stream.connected.set()
            await asyncio.Event().wait()

        stream._run = connected_without_data  # type: ignore[method-assign]
        try:
            with pytest.raises(RuntimeError, match="did not become ready"):
                await stream.start(timeout_sec=0.01)
        finally:
            await stream.close()

    asyncio.run(run())
