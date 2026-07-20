"""Loss-aware Binance USD-M aggregate-trade stream for live barriers."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any


LOG = logging.getLogger("portfolio_live.aggtrade")


@dataclass(frozen=True)
class AggTradeTick:
    price: float
    event_time_ms: int
    aggregate_trade_id: int | None = None
    quantity: float | None = None
    is_buyer_maker: bool | None = None
    first_trade_id: int | None = None
    last_trade_id: int | None = None


def parse_aggtrade_payload(payload: dict[str, Any]) -> AggTradeTick:
    """Parse the official USD-M aggTrade payload without dropping BAFR fields."""

    if payload.get("e") != "aggTrade":
        raise ValueError("payload is not an aggregate trade")
    required = ("a", "p", "q", "f", "l", "T", "m")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"aggTrade payload is missing fields: {missing}")
    price = float(payload["p"])
    quantity = float(payload["q"])
    aggregate_trade_id = int(payload["a"])
    first_trade_id = int(payload["f"])
    last_trade_id = int(payload["l"])
    event_time_ms = int(payload["T"])
    buyer_maker = payload["m"]
    if not math.isfinite(price) or not math.isfinite(quantity) or price <= 0.0 or quantity <= 0.0:
        raise ValueError("aggTrade price and quantity must be positive")
    if aggregate_trade_id < 0 or first_trade_id < 0 or last_trade_id < first_trade_id:
        raise ValueError("aggTrade identifiers are invalid")
    if event_time_ms < 0:
        raise ValueError("aggTrade timestamp is invalid")
    if not isinstance(buyer_maker, bool):
        raise ValueError("aggTrade buyer-maker flag must be boolean")
    return AggTradeTick(
        price=price,
        event_time_ms=event_time_ms,
        aggregate_trade_id=aggregate_trade_id,
        quantity=quantity,
        is_buyer_maker=buyer_maker,
        first_trade_id=first_trade_id,
        last_trade_id=last_trade_id,
    )


class BinanceAggTradeStream:
    """Maintain an ordered public trade queue and expose continuity state."""

    # USD-M market streams moved behind the routed ``/market`` endpoint in
    # 2026.  The legacy unrouted path can still complete a WebSocket handshake
    # while silently producing no aggTrade messages.
    MAINNET_URL = "wss://fstream.binance.com/market/ws/{symbol}@aggTrade"
    TESTNET_URL = "wss://demo-fstream.binance.com/market/ws/{symbol}@aggTrade"

    def __init__(
        self,
        *,
        symbol: str,
        testnet: bool,
        max_queue_size: int = 200_000,
        stale_after_sec: float = 5.0,
    ) -> None:
        template = self.TESTNET_URL if testnet else self.MAINNET_URL
        self.url = template.format(symbol=str(symbol).lower())
        self.session_id = f"agg-{time.time_ns()}"
        self.queue: asyncio.Queue[AggTradeTick] = asyncio.Queue(maxsize=max(1, int(max_queue_size)))
        self.connected = asyncio.Event()
        self.ready = asyncio.Event()
        self.closed = False
        self.gap_count = 0
        self.overflowed = False
        self.discontinuous = False
        self.previous_aggregate_trade_id: int | None = None
        self.last_event_time_ms: int | None = None
        self.last_message_monotonic: float | None = None
        self.last_error: str | None = None
        self.stale_after_sec = max(0.1, float(stale_after_sec))
        self._task: asyncio.Task[None] | None = None

    @property
    def healthy(self) -> bool:
        if (
            not self.connected.is_set()
            or not self.ready.is_set()
            or self.overflowed
            or self.discontinuous
            or self.closed
        ):
            return False
        if self.last_message_monotonic is None:
            return False
        return (time.monotonic() - self.last_message_monotonic) <= self.stale_after_sec

    async def start(self, *, timeout_sec: float = 10.0) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"aggtrade-{self.session_id}")
        try:
            # A successful handshake is not enough: unrouted Binance Futures
            # URLs can connect without delivering market events.  Do not let a
            # barrier sleeve enter until at least one valid trade is observed.
            await asyncio.wait_for(self.ready.wait(), timeout=max(0.1, float(timeout_sec)))
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Binance aggTrade stream did not become ready: {self.last_error or self.url}") from exc

    async def close(self) -> None:
        self.closed = True
        self.connected.clear()
        self.ready.clear()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def collect(self, *, timeout_sec: float) -> list[AggTradeTick]:
        """Wait for one tick, then drain the currently ordered backlog."""

        ticks: list[AggTradeTick] = []
        try:
            ticks.append(await asyncio.wait_for(self.queue.get(), timeout=max(0.01, float(timeout_sec))))
        except asyncio.TimeoutError:
            return ticks
        while True:
            try:
                ticks.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                return ticks

    def _accept_contiguous_tick(self, tick: AggTradeTick) -> bool:
        identifier = tick.aggregate_trade_id
        if identifier is None:
            self.discontinuous = True
            self.last_error = "aggTrade message has no aggregate trade ID"
            return False
        previous = self.previous_aggregate_trade_id
        if previous is not None and identifier != previous + 1:
            self.discontinuous = True
            self.gap_count += 1
            self.last_error = (
                f"aggTrade ID discontinuity: expected {previous + 1}, got {identifier}"
            )
            return False
        self.previous_aggregate_trade_id = identifier
        return True

    async def _run(self) -> None:
        import websockets

        reconnect_attempt = 0
        while not self.closed:
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as socket:
                    reconnect_attempt = 0
                    self.ready.clear()
                    self.connected.set()
                    async for message in socket:
                        if self.closed:
                            return
                        try:
                            payload: dict[str, Any] = json.loads(message)
                        except (json.JSONDecodeError, TypeError) as exc:
                            self.discontinuous = True
                            self.last_error = f"invalid aggTrade message: {exc}"
                            self.connected.clear()
                            break
                        if payload.get("e") != "aggTrade":
                            continue
                        try:
                            tick = parse_aggtrade_payload(payload)
                        except (KeyError, TypeError, ValueError) as exc:
                            self.discontinuous = True
                            self.last_error = f"invalid aggTrade payload: {exc}"
                            self.connected.clear()
                            break
                        if not self._accept_contiguous_tick(tick):
                            self.connected.clear()
                            break
                        self.last_event_time_ms = tick.event_time_ms
                        self.last_message_monotonic = time.monotonic()
                        if self.queue.full():
                            self.overflowed = True
                            self.connected.clear()
                            self.last_error = "aggTrade queue overflow"
                            break
                        self.queue.put_nowait(tick)
                        self.ready.set()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                LOG.warning("aggTrade stream error: %s", self.last_error)
            finally:
                self.connected.clear()
                self.ready.clear()
            if self.closed or self.overflowed or self.discontinuous:
                return
            self.gap_count += 1
            await asyncio.sleep(min(30.0, float(2**min(reconnect_attempt, 5))))
            reconnect_attempt += 1
