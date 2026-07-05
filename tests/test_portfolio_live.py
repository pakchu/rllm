import asyncio
import unittest
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd

from execution.portfolio_live import (
    PORTFOLIO_ORDER_PREFIX,
    _cancel_portfolio_orders_for_sleeve,
    _cancel_stale_portfolio_orders,
    _entry_ttl_seconds,
    _margin_fraction_for_weight,
    _place_portfolio_maker_order_with_deadline,
    _portfolio_client_order_id,
    _portfolio_sleeve_key,
)


class FakeExecutor:
    async def get_maker_price(self, side, ws_orderbook=None):
        return 100.0


class FakeClient:
    def __init__(self, *, order_statuses=None, open_orders=None):
        self.order_statuses = list(order_statuses or [])
        self.open_orders = list(open_orders or [])
        self.cancelled = []
        self.placed = []

    async def place_order(self, **kwargs):
        self.placed.append(kwargs)
        return {"orderId": 101, "clientOrderId": kwargs["client_order_id"], "status": "NEW"}

    async def get_order(self, symbol, order_id=None, client_order_id=None):
        if self.order_statuses:
            return self.order_statuses.pop(0)
        return {"orderId": order_id, "status": "NEW", "executedQty": "0", "avgPrice": "0"}

    async def cancel_order(self, symbol, order_id=None, client_order_id=None):
        self.cancelled.append({"symbol": symbol, "order_id": order_id, "client_order_id": client_order_id})
        return {"status": "CANCELED", "clientOrderId": client_order_id}

    async def get_open_orders(self, symbol=None):
        return list(self.open_orders)


class PortfolioLiveSafetyTests(unittest.TestCase):
    def test_research_gross_margin_preserves_weight(self):
        margin = _margin_fraction_for_weight(
            weight=2.475,
            total_weight=4.95,
            leverage_budget=6,
            allocation_mode="research_gross",
        )
        self.assertAlmostEqual(margin, 0.4125)
        self.assertAlmostEqual(margin * 6, 2.475)

    def test_entry_ttl_uses_stride_cycle_with_cap(self):
        sleeve = {"stride_bars": 6, "hold_bars": 144}
        self.assertEqual(
            _entry_ttl_seconds(sleeve, interval_minutes=5, timeout_fraction=0.25, max_entry_wait_sec=300),
            300,
        )
        self.assertEqual(
            _entry_ttl_seconds(sleeve, interval_minutes=5, timeout_fraction=0.10, max_entry_wait_sec=300),
            180,
        )

    def test_client_order_id_encodes_sleeve_and_stays_short(self):
        cid = _portfolio_client_order_id("alpha:2026-07-05T00:00:00", sleeve_name="rex_short", now_sec=1234567890)
        self.assertTrue(cid.startswith(PORTFOLIO_ORDER_PREFIX + "_1234567890_"))
        self.assertIn(_portfolio_sleeve_key("rex_short"), cid)
        self.assertLessEqual(len(cid), 36)

    def test_post_only_entry_cancels_unfilled_by_deadline(self):
        async def run():
            client = FakeClient(order_statuses=[{"orderId": 101, "status": "NEW", "executedQty": "0", "avgPrice": "0"}])
            result = await _place_portfolio_maker_order_with_deadline(
                client=client,
                executor=FakeExecutor(),
                exec_cfg=SimpleNamespace(symbol="BTCUSDT"),
                order_side="BUY",
                quantity=Decimal("0.01"),
                position_side="LONG",
                signal_id="sig-1",
                sleeve_name="rex_short",
                ttl_sec=0,
                poll_interval_sec=0.05,
            )
            self.assertEqual(result["status"], "TIMEOUT_CANCELLED")
            self.assertEqual(result["filled_quantity"], "0")
            self.assertEqual(len(client.cancelled), 1)
            self.assertIsNone(client.cancelled[0]["order_id"])
            self.assertTrue(client.cancelled[0]["client_order_id"].startswith(PORTFOLIO_ORDER_PREFIX + "_"))

        asyncio.run(run())

    def test_cancel_stale_only_touches_portfolio_prefix(self):
        async def run():
            stale_cid = _portfolio_client_order_id("sig-old", sleeve_name="rex_short", now_sec=100)
            fresh_cid = _portfolio_client_order_id("sig-new", sleeve_name="rex_short", now_sec=9999999999)
            client = FakeClient(
                open_orders=[
                    {"orderId": 1, "clientOrderId": stale_cid, "time": 100_000},
                    {"orderId": 2, "clientOrderId": "otherbot_abc", "time": 100_000},
                    {"orderId": 3, "clientOrderId": fresh_cid, "time": 9_999_999_999_000},
                ]
            )
            cancelled = await _cancel_stale_portfolio_orders(
                client=client,
                symbol="BTCUSDT",
                now=pd.Timestamp(1970, 1, 1, 0, 10, tz="UTC"),
                max_age_sec=300,
            )
            self.assertEqual([c["client_order_id"] for c in cancelled], [stale_cid])
            self.assertEqual([c["client_order_id"] for c in client.cancelled], [stale_cid])

        asyncio.run(run())

    def test_new_signal_replaces_same_sleeve_order_only(self):
        async def run():
            target = _portfolio_client_order_id("sig-old", sleeve_name="rex_short", now_sec=100)
            other_sleeve = _portfolio_client_order_id("sig-old", sleeve_name="alpha_long", now_sec=100)
            client = FakeClient(
                open_orders=[
                    {"orderId": 1, "clientOrderId": target, "time": 100_000},
                    {"orderId": 2, "clientOrderId": other_sleeve, "time": 100_000},
                ]
            )
            cancelled = await _cancel_portfolio_orders_for_sleeve(
                client=client,
                symbol="BTCUSDT",
                sleeve_name="rex_short",
                reason="new_signal_replaces_stale_entry",
            )
            self.assertEqual([c["client_order_id"] for c in cancelled], [target])
            self.assertEqual([c["client_order_id"] for c in client.cancelled], [target])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
