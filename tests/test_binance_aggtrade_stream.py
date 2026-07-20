from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from execution.binance_aggtrade_stream import (
    AggTradeTick,
    BinanceAggTradeStream,
    parse_aggtrade_payload,
)


def _payload(identifier: int, *, price: str = "100.0", maker: bool = False) -> dict[str, object]:
    return {
        "e": "aggTrade",
        "E": 1_609_459_200_100,
        "s": "BTCUSDT",
        "a": identifier,
        "p": price,
        "q": "2.5",
        "f": identifier * 10,
        "l": identifier * 10 + 2,
        "T": 1_609_459_200_000 + identifier,
        "m": maker,
    }


def test_payload_parser_preserves_frustration_fields() -> None:
    tick = parse_aggtrade_payload(_payload(7, price="101.5", maker=True))
    assert tick == AggTradeTick(
        price=101.5,
        event_time_ms=1_609_459_200_007,
        aggregate_trade_id=7,
        quantity=2.5,
        is_buyer_maker=True,
        first_trade_id=70,
        last_trade_id=72,
    )


def test_payload_parser_rejects_missing_or_invalid_frustration_fields() -> None:
    missing = _payload(7)
    missing.pop("m")
    with pytest.raises(ValueError, match="missing fields"):
        parse_aggtrade_payload(missing)
    invalid = _payload(7)
    invalid["p"] = "nan"
    with pytest.raises(ValueError, match="positive"):
        parse_aggtrade_payload(invalid)


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


def test_stream_fails_closed_on_aggregate_trade_id_gap() -> None:
    stream = BinanceAggTradeStream(symbol="BTCUSDT", testnet=False)
    assert stream._accept_contiguous_tick(parse_aggtrade_payload(_payload(10))) is True
    assert stream._accept_contiguous_tick(parse_aggtrade_payload(_payload(11))) is True
    assert stream._accept_contiguous_tick(parse_aggtrade_payload(_payload(13))) is False
    assert stream.discontinuous is True
    assert stream.gap_count == 1
    assert stream.last_error == "aggTrade ID discontinuity: expected 12, got 13"
    stream.connected.set()
    stream.ready.set()
    stream.last_message_monotonic = 1.0
    assert stream.healthy is False


def test_stream_fails_closed_on_duplicate_or_reversed_id() -> None:
    duplicate = BinanceAggTradeStream(symbol="BTCUSDT", testnet=False)
    assert duplicate._accept_contiguous_tick(parse_aggtrade_payload(_payload(10))) is True
    assert duplicate._accept_contiguous_tick(parse_aggtrade_payload(_payload(10))) is False
    assert duplicate.discontinuous is True

    reversed_stream = BinanceAggTradeStream(symbol="BTCUSDT", testnet=False)
    assert reversed_stream._accept_contiguous_tick(parse_aggtrade_payload(_payload(10))) is True
    assert reversed_stream._accept_contiguous_tick(parse_aggtrade_payload(_payload(9))) is False
    assert reversed_stream.discontinuous is True


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
