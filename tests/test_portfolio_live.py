import asyncio
import unittest
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd

from execution.portfolio_live import (
    PORTFOLIO_ORDER_PREFIX,
    PortfolioLiveConfig,
    _cancel_portfolio_orders_for_sleeve,
    _add_live_volume_wave_features,
    _add_portfolio_oi_features,
    _cancel_stale_portfolio_orders,
    _entry_ttl_seconds,
    _completed_decision_data_asof,
    _margin_fraction_for_weight,
    _load_sleeve_runtime_spec,
    _place_portfolio_maker_order_with_deadline,
    _portfolio_client_order_id,
    _portfolio_sleeve_key,
    _reconcile_exchange_flat_sleeves,
    _recover_exchange_positions_into_state,
    _summarize_exchange_trade_fills,
    _gate_clauses_pass,
    _gate_pass,
    _freshness_requirements_for_decision,
    _portfolio_uses_feature,
    _validate_portfolio_mode,
)


class FakeExecutor:
    async def get_maker_price(self, side, ws_orderbook=None):
        return 100.0


class FakeClient:
    def __init__(self, *, order_statuses=None, open_orders=None, cancel_responses=None):
        self.order_statuses = list(order_statuses or [])
        self.open_orders = list(open_orders or [])
        self.cancel_responses = list(cancel_responses or [])
        self.cancelled = []
        self.placed = []

    async def place_order(self, **kwargs):
        self.placed.append(kwargs)
        return {"orderId": 100 + len(self.placed), "clientOrderId": kwargs["client_order_id"], "status": "NEW"}

    async def get_order(self, symbol, order_id=None, client_order_id=None):
        if self.order_statuses:
            return self.order_statuses.pop(0)
        return {"orderId": order_id, "status": "NEW", "executedQty": "0", "avgPrice": "0"}

    async def cancel_order(self, symbol, order_id=None, client_order_id=None):
        self.cancelled.append({"symbol": symbol, "order_id": order_id, "client_order_id": client_order_id})
        if self.cancel_responses:
            response = dict(self.cancel_responses.pop(0))
            response.setdefault("clientOrderId", client_order_id)
            return response
        return {"status": "CANCELED", "clientOrderId": client_order_id}

    async def get_open_orders(self, symbol=None):
        return list(self.open_orders)


class PortfolioLiveSafetyTests(unittest.TestCase):
    def test_shadow_candidate_is_hard_blocked_from_live_orders(self):
        import json

        with open("configs/live/portfolio_added_alpha_shadow_candidate_2026-07-16.json") as handle:
            portfolio = json.load(handle)
        _validate_portfolio_mode(portfolio, live=False)
        with self.assertRaisesRegex(RuntimeError, "not authorized for live orders"):
            _validate_portfolio_mode(portfolio, live=True)

    def test_live_anchor_remains_authorized(self):
        import json

        with open("configs/live/portfolio_gross385_trainmdd40_2026-07-12.json") as handle:
            portfolio = json.load(handle)
        _validate_portfolio_mode(portfolio, live=True)

    def test_gate_clauses_are_or_of_and_groups(self):
        row = pd.Series({"a": 2.0, "b": 0.0, "c": 3.0, "d": 4.0})
        clauses = [
            [{"feature": "a", "op": ">=", "threshold": 1.0}, {"feature": "b", "op": ">=", "threshold": 1.0}],
            [{"feature": "c", "op": ">=", "threshold": 2.0}, {"feature": "d", "op": "<=", "threshold": 5.0}],
        ]
        passed, reasons = _gate_clauses_pass(row, clauses)
        self.assertTrue(passed)
        self.assertEqual(reasons[-1], "gate_clauses:any:pass")

    def test_gross385_portfolio_discovers_clause_features(self):
        import json

        with open("configs/live/portfolio_gross385_trainmdd40_2026-07-12.json") as handle:
            portfolio = json.load(handle)
        self.assertTrue(_portfolio_uses_feature(portfolio, "funding_rate"))
        self.assertTrue(_portfolio_uses_feature(portfolio, "premium_index_change"))

    def test_freshness_wait_excludes_fx_and_uses_no_oi_boundary_gate(self):
        requirements = _freshness_requirements_for_decision(
            symbol="BTCUSDT",
            expected_bar=pd.Timestamp("2026-07-10T12:00:00Z"),
            required_1m=pd.Timestamp("2026-07-10T12:04:00Z"),
        )

        keys = {requirement.key for requirement in requirements}
        self.assertNotIn("bars_polygon:USDKRW:1m", keys)
        self.assertNotIn("open_interest_binance:BTCUSDT:1m", keys)

    def test_current_portfolio_skips_unused_activity_flow_feature(self):
        import json

        with open("configs/live/portfolio_gross610_dynamic_top1_2026-07-08.json") as handle:
            portfolio = json.load(handle)

        self.assertFalse(_portfolio_uses_feature(portfolio, "activity_flow_htf"))

    def test_completed_decision_cutoff_excludes_next_incomplete_candle(self):
        expected = pd.Timestamp("2026-07-10T12:00:00Z")
        self.assertEqual(
            _completed_decision_data_asof(expected, interval_minutes=5),
            pd.Timestamp("2026-07-10T12:04:00Z"),
        )

    def test_runtime_spec_preserves_dynamic_exit_for_restart_recovery(self):
        spec = _load_sleeve_runtime_spec(
            {
                "source": "configs/live/oi_alt_ratio72_dyn_exit_candidate.json",
            }
        )
        self.assertEqual(spec["hold_bars"], 288)
        self.assertEqual(spec["dynamic_exit"]["name"], "vwap_overheat")
        self.assertEqual(spec["dynamic_exit"]["min_bars"], 48)

    def test_exchange_fill_summary_reports_realized_pnl_and_fees(self):
        report = _summarize_exchange_trade_fills(
            [
                {
                    "orderId": 7,
                    "qty": "0.002",
                    "price": "100",
                    "quoteQty": "0.2",
                    "realizedPnl": "-0.01",
                    "commission": "0.00004",
                    "commissionAsset": "USDT",
                    "time": 1_000,
                }
            ]
        )
        self.assertEqual(report["quantity"], "0.002")
        self.assertEqual(report["avg_price"], "1E+2")
        self.assertEqual(report["realized_pnl"], "-0.01")
        self.assertEqual(report["net_realized_pnl"], "-0.01004")

    def test_restart_reconciles_exchange_flat_as_attributed_close(self):
        async def run():
            signal_id = "rex_dual_regime_auto:LONG:2026-07-08T11:55:00"
            cid = _portfolio_client_order_id(signal_id, sleeve_name="rex_dual_regime_auto", now_sec=100)

            class ReconcileClient:
                async def get_positions(self, symbol=None):
                    return [{"positionSide": "LONG", "positionAmt": "0"}]

                async def get_trades(self, symbol, limit=1000):
                    return [
                        {
                            "orderId": 77,
                            "positionSide": "LONG",
                            "side": "SELL",
                            "qty": "0.003",
                            "price": "101",
                            "quoteQty": "0.303",
                            "realizedPnl": "0.003",
                            "commission": "0.00006",
                            "commissionAsset": "USDT",
                            "time": int(pd.Timestamp("2026-07-08T12:00:00Z").timestamp() * 1000),
                        }
                    ]

                async def get_order(self, symbol, order_id=None):
                    return {"orderId": order_id, "clientOrderId": cid}

            state = {
                "open_sleeves": {
                    "rex_dual_regime_auto": {
                        "name": "rex_dual_regime_auto",
                        "side": "LONG",
                        "signal_id": signal_id,
                        "signal_date": "2026-07-08T11:55:00",
                        "exit_at": "2026-07-09T00:00:00Z",
                        "quantity": "0.003",
                    }
                }
            }
            rows = await _reconcile_exchange_flat_sleeves(
                state=state,
                client=ReconcileClient(),
                exec_cfg=SimpleNamespace(symbol="BTCUSDT"),
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["order_info"]["status"], "FILLED_RECONCILED")
            self.assertEqual(rows[0]["order_info"]["trade_report"]["realized_pnl"], "0.003")

        asyncio.run(run())

    def test_exchange_recovery_restores_dynamic_exit_spec(self):
        async def run():
            name = "oi_alt_ratio72_dyn_exit"
            signal_id = f"{name}:LONG:2026-07-08T14:30:00"
            cid = _portfolio_client_order_id(signal_id, sleeve_name=name, now_sec=100)
            entry_ms = int(pd.Timestamp("2026-07-08T14:30:00Z").timestamp() * 1000)

            class RecoveryClient:
                async def get_positions(self, symbol=None):
                    return [{"positionSide": "LONG", "positionAmt": "0.003", "entryPrice": "100"}]

                async def _private_request(self, method, path, params):
                    return [{"orderId": 88, "positionSide": "LONG", "side": "BUY", "time": entry_ms, "price": "100"}]

                async def get_order(self, symbol, order_id=None):
                    return {"orderId": order_id, "clientOrderId": cid, "time": entry_ms}

            state = {"open_sleeves": {}, "processed_signals": {}}
            portfolio = {
                "base_sleeves": [
                    {
                        "name": name,
                        "source": "configs/live/oi_alt_ratio72_dyn_exit_candidate.json",
                        "side": "LONG",
                        "weight": 0.35,
                    }
                ]
            }
            rows = await _recover_exchange_positions_into_state(
                state=state,
                client=RecoveryClient(),
                exec_cfg=SimpleNamespace(symbol="BTCUSDT", interval_minutes=5, max_holding_bars=144),
                portfolio=portfolio,
                leverage_budget=7,
                allocation_mode="research_gross",
            )
            self.assertEqual(len(rows), 1)
            restored = state["open_sleeves"][name]
            self.assertEqual(restored["exit_at"], "2026-07-09 14:35:00+00:00")
            self.assertEqual(restored["dynamic_exit"]["name"], "vwap_overheat")

        asyncio.run(run())

    def test_rex_selector_default_model_is_text_only(self):
        self.assertEqual(PortfolioLiveConfig(portfolio_config=__import__("pathlib").Path("p.json"), execution_config=__import__("pathlib").Path("e.json")).rex_selector_model_name, "qwen2.5-1.5b-instruct")

    def test_default_freshness_wait_covers_observed_upbit_commit_tail(self):
        cfg = PortfolioLiveConfig(portfolio_config=__import__("pathlib").Path("p.json"), execution_config=__import__("pathlib").Path("e.json"))
        self.assertEqual(cfg.max_freshness_wait_sec, 50.0)

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

    def test_post_only_refresh_reorders_only_uncancelled_remainder(self):
        async def run():
            client = FakeClient(
                order_statuses=[{"orderId": 101, "status": "NEW", "executedQty": "0.004", "avgPrice": "100"}],
                cancel_responses=[
                    {"status": "CANCELED", "executedQty": "0.006", "avgPrice": "100"},
                    {"status": "CANCELED", "executedQty": "0.004", "avgPrice": "100"},
                ],
            )
            result = await _place_portfolio_maker_order_with_deadline(
                client=client,
                executor=FakeExecutor(),
                exec_cfg=SimpleNamespace(symbol="BTCUSDT"),
                order_side="BUY",
                quantity=Decimal("0.01"),
                position_side="LONG",
                signal_id="sig-partial",
                sleeve_name="rex_short",
                ttl_sec=2,
                refresh_interval_sec=1,
                poll_interval_sec=0.05,
            )
            self.assertEqual(result["status"], "FILLED")
            self.assertEqual(Decimal(result["filled_quantity"]), Decimal("0.010"))
            self.assertGreaterEqual(len(client.placed), 2)
            self.assertEqual(Decimal(str(client.placed[0]["quantity"])), Decimal("0.01"))
            self.assertEqual(Decimal(str(client.placed[1]["quantity"])), Decimal("0.004"))

        asyncio.run(run())


    def test_live_alt_pool_volume_ratio_uses_ingested_pool(self):
        dates = pd.date_range("2026-07-01", periods=320, freq="5min")
        enriched = pd.DataFrame(
            {
                "date": dates,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
                "quote_asset_volume": 1000.0,
            }
        )
        features = pd.DataFrame(index=enriched.index)
        alt_rows = []
        for i, ts in enumerate(dates):
            for symbol in ("ETHUSDT", "SOLUSDT"):
                alt_rows.append({"date": ts, "symbol": symbol, "quote_asset_volume": 1000.0 + i * 10.0})
        out = _add_live_volume_wave_features(enriched, features, {}, pd.DataFrame(alt_rows))
        self.assertIn("vg_alt_btc_qv_ratio_z_72", out.columns)
        self.assertIn("vg_alt_btc_qv_ratio_z_288", out.columns)
        self.assertNotEqual(float(out["vg_alt_btc_qv_ratio_z_72"].iloc[-1]), 0.0)
        self.assertNotEqual(float(out["vg_alt_btc_qv_ratio_z_288"].iloc[-1]), 0.0)


    def test_gate_fails_closed_when_source_availability_missing(self):
        cases = [
            (
                pd.Series({"premium_index_zscore": -3.0, "premium_available": 0.0}),
                [{"feature": "premium_index_zscore", "op": "<=", "threshold": -2.0}],
                "premium_available",
            ),
            (
                pd.Series({"oi_minus_px_4h_z": 3.0, "open_interest_available": 0.0}),
                [{"feature": "oi_minus_px_4h_z", "op": ">=", "threshold": 2.0}],
                "open_interest_available",
            ),
            (
                pd.Series({"vg_alt_btc_qv_ratio_z_72": 3.0, "alt_pool_available": 0.0}),
                [{"feature": "vg_alt_btc_qv_ratio_z_72", "op": ">=", "threshold": 2.0}],
                "alt_pool_available",
            ),
        ]
        for row, gates, flag in cases:
            ok, reasons = _gate_pass(row, gates)
            self.assertFalse(ok)
            self.assertTrue(any(flag in reason and reason.endswith(":fail") for reason in reasons), reasons)

    def test_oi_features_do_not_forward_fill_unavailable_live_rows(self):
        n = 70
        enriched = pd.DataFrame(
            {
                "date": pd.date_range("2026-07-01", periods=n, freq="5min"),
                "close": 100.0,
                "open_interest": [1000.0] * (n - 1) + [float("nan")],
                "open_interest_available": [1.0] * (n - 1) + [0.0],
            }
        )
        features = pd.DataFrame(index=enriched.index)
        out = _add_portfolio_oi_features(enriched, features)
        self.assertEqual(float(out["open_interest_available"].iloc[-1]), 0.0)
        self.assertTrue(pd.isna(out["oi_ret_30m"].iloc[-1]))

    def test_live_volume_features_mark_missing_alt_pool_unavailable(self):
        dates = pd.date_range("2026-07-01", periods=320, freq="5min")
        enriched = pd.DataFrame(
            {
                "date": dates,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
                "quote_asset_volume": 1000.0,
            }
        )
        out = _add_live_volume_wave_features(enriched, pd.DataFrame(index=enriched.index), {}, None)
        self.assertIn("alt_pool_available", out.columns)
        self.assertEqual(float(out["alt_pool_available"].iloc[-1]), 0.0)

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
