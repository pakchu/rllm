import asyncio
import io
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from execution.wave_execution import WaveExecutionConfig
from execution.portfolio_live import (
    AlphaProcessManager,
    PORTFOLIO_ORDER_PREFIX,
    PortfolioLiveConfig,
    _acquire_portfolio_db_lease,
    _acquire_portfolio_runner_lock,
    _build_open_intents,
    _cancel_portfolio_orders_for_sleeve,
    _add_live_volume_wave_features,
    _add_portfolio_oi_features,
    _cancel_stale_portfolio_orders,
    _close_sleeve,
    _entry_ttl_seconds,
    _execution_exchange_scope,
    _ensure_trade_executions_table,
    _execute_close_intents,
    _execute_open_intents,
    _finish_trade_intents,
    _completed_decision_data_asof,
    _margin_fraction_for_weight,
    _make_executor,
    _load_sleeve_runtime_spec,
    _place_portfolio_maker_order_with_deadline,
    _portfolio_client_order_id,
    _portfolio_db_lease_key,
    _portfolio_gate_features,
    _portfolio_sleeve_key,
    _reconcile_exchange_flat_sleeves,
    _recover_exchange_positions_into_state,
    _score_sleeves,
    _summarize_exchange_trade_fills,
    _terminate_process_executor,
    _gate_clauses_pass,
    _gate_pass,
    _freshness_requirements_for_decision,
    _interval_slot,
    _portfolio_uses_feature,
    _release_portfolio_runner_lock,
    _release_portfolio_db_lease,
    _reserve_trade_intents,
    _assert_portfolio_db_lease,
    _validate_portfolio_mode,
    parse_args,
)


def _sleeping_process_worker(seconds):
    time.sleep(seconds)
    return seconds


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
    def test_cli_help_renders_percent_literals(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["portfolio_live", "--help"]), redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                parse_args()
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("100% margin budget", output.getvalue())
        self.assertIn("calibrated to 0.3%", output.getvalue())

    def test_exchange_scope_separates_testnet_and_rejects_network_mismatch(self):
        self.assertEqual(_execution_exchange_scope("binance", testnet=False), "binance")
        self.assertEqual(_execution_exchange_scope("binance", testnet=True), "binance-testnet")
        self.assertEqual(
            _execution_exchange_scope("binance-testnet", testnet=True),
            "binance-testnet",
        )
        with self.assertRaisesRegex(ValueError, "cannot use mainnet"):
            _execution_exchange_scope("binance-mainnet", testnet=True)
        with self.assertRaisesRegex(ValueError, "cannot use testnet"):
            _execution_exchange_scope("binance-testnet", testnet=False)

    def test_executor_initialization_failure_closes_exchange_client(self):
        async def run():
            instances = []

            class Client:
                def __init__(self, **kwargs):
                    self.closed = False
                    instances.append(self)

                async def sync_time(self):
                    return 0

                async def is_hedge_mode(self, force_refresh=False):
                    return False

                async def aclose(self):
                    self.closed = True

            class Executor:
                def __init__(self, **kwargs):
                    pass

            cfg = WaveExecutionConfig(dry_run=False, allow_live_orders=True, testnet=True)
            with patch(
                "execution.portfolio_live._load_api_credentials",
                return_value=("key", "secret"),
            ), patch(
                "execution.portfolio_live.load_wave_execution_classes",
                return_value=(Client, Executor),
            ):
                with self.assertRaisesRegex(RuntimeError, "hedge mode"):
                    await _make_executor(cfg)
            self.assertEqual(len(instances), 1)
            self.assertTrue(instances[0].closed)

        asyncio.run(run())

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

    def test_live_alpha_stride_contract_matches_historical_grid(self):
        anchor = pd.Timestamp("2020-01-01T02:55:00Z")
        cases = [
            ("configs/live/oi_upbit_ratio288_low_candidate.json", 6, 5),
            ("configs/live/new_long_minimal_funding_premium_candidate.json", 12, 11),
            ("configs/live/rex_veto_7_candidate.json", 24, 11),
        ]
        for path, stride, offset in cases:
            with self.subTest(path=path):
                with open(path) as handle:
                    candidate = json.load(handle)
                self.assertEqual(candidate["stride_bars"], stride)
                self.assertEqual(candidate["stride_offset_bars"], offset)
                self.assertEqual(candidate["entry_delay_bars"], 1)
                self.assertTrue(
                    _interval_slot(
                        anchor,
                        stride_bars=stride,
                        interval_minutes=5,
                        stride_offset_bars=offset,
                    )
                )

    def test_serial_scorer_preserves_sleeve_order_and_does_not_mutate_snapshot(self):
        dates = pd.date_range("2026-07-16T00:00:00Z", periods=20, freq="5min")
        enriched = pd.DataFrame(
            {
                "date": dates,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
            }
        )
        features = pd.DataFrame({"alpha_x": [0.0] * 19 + [2.0]})
        enriched_before = enriched.copy(deep=True)
        features_before = features.copy(deep=True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = []
            for name, threshold in (("second", 3.0), ("first", 1.0)):
                source = root / f"{name}.json"
                source.write_text(
                    json.dumps(
                        {
                            "name": name,
                            "gates": [{"feature": "alpha_x", "op": ">=", "threshold": threshold}],
                            "hold_bars": 12,
                            "stride_bars": 1,
                            "stride_offset_bars": 0,
                            "entry_delay_bars": 1,
                        }
                    )
                )
                sources.append(source)

            portfolio = {
                "base_sleeves": [
                    {"name": "second", "source": str(sources[0]), "side": "LONG", "weight": 0.5},
                    {"name": "first", "source": str(sources[1]), "side": "LONG", "weight": 0.5},
                ]
            }
            scores = _score_sleeves(
                portfolio=portfolio,
                enriched=enriched,
                features=features,
                exec_cfg=WaveExecutionConfig(),
                asof=dates[-1],
            )

        self.assertEqual([score["name"] for score in scores], ["second", "first"])
        self.assertEqual([score["active"] for score in scores], [False, True])
        pd.testing.assert_frame_equal(enriched, enriched_before)
        pd.testing.assert_frame_equal(features, features_before)

    def test_scorer_fails_closed_for_missing_config_and_unsupported_entry_delay(self):
        dates = pd.date_range("2026-07-16T00:00:00Z", periods=20, freq="5min")
        enriched = pd.DataFrame(
            {"date": dates, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        )
        features = pd.DataFrame({"alpha_x": [2.0] * len(dates)})
        with tempfile.TemporaryDirectory() as tmp:
            delayed = Path(tmp) / "delayed.json"
            delayed.write_text(
                json.dumps(
                    {
                        "gates": [{"feature": "alpha_x", "op": ">=", "threshold": 1.0}],
                        "hold_bars": 12,
                        "stride_bars": 1,
                        "entry_delay_bars": 2,
                    }
                )
            )
            portfolio = {
                "base_sleeves": [
                    {"name": "missing", "source": str(Path(tmp) / "missing.json"), "side": "LONG", "weight": 0.5},
                    {"name": "delayed", "source": str(delayed), "side": "LONG", "weight": 0.5},
                ]
            }
            scores = _score_sleeves(
                portfolio=portfolio,
                enriched=enriched,
                features=features,
                exec_cfg=WaveExecutionConfig(),
                asof=dates[-1],
            )

        self.assertFalse(scores[0]["active"])
        self.assertTrue(any(reason.startswith("source_config=missing:") for reason in scores[0]["reasons"]))
        self.assertFalse(scores[1]["active"])
        self.assertIn("entry_delay_bars=2:unsupported_fail_closed", scores[1]["reasons"])

    def test_process_manager_uses_one_worker_per_sleeve_and_isolates_failure(self):
        async def run():
            dates = pd.date_range("2026-07-16T00:00:00Z", periods=20, freq="5min")
            enriched = pd.DataFrame(
                {"date": dates, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
            )
            features = pd.DataFrame({"alpha_x": [0.0] * 19 + [2.0]})
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sleeves = []
                for name in ("alpha_a", "alpha_b"):
                    source = root / f"{name}.json"
                    source.write_text(
                        json.dumps(
                            {
                                "gates": [{"feature": "alpha_x", "op": ">=", "threshold": 1.0}],
                                "hold_bars": 12,
                                "stride_bars": 1,
                                "stride_offset_bars": 0,
                                "entry_delay_bars": 1,
                            }
                        )
                    )
                    sleeves.append({"name": name, "source": str(source), "side": "LONG", "weight": 0.4})
                broken = root / "broken.json"
                broken.write_text("{not-json")
                sleeves.append({"name": "broken", "source": str(broken), "side": "SHORT", "weight": 0.2})
                portfolio = {"base_sleeves": sleeves}
                manager = AlphaProcessManager(sleeves, timeout_sec=30.0)
                try:
                    scores = await manager.score(
                        portfolio=portfolio,
                        enriched=enriched,
                        features=features,
                        exec_cfg=WaveExecutionConfig(),
                        asof=dates[-1],
                    )
                finally:
                    await manager.shutdown()

            self.assertEqual([score["name"] for score in scores], ["alpha_a", "alpha_b", "broken"])
            self.assertEqual([score["active"] for score in scores], [True, True, False])
            worker_pids = [scores[0]["worker_pid"], scores[1]["worker_pid"]]
            self.assertEqual(len(set(worker_pids)), 2)
            self.assertNotIn(os.getpid(), worker_pids)
            self.assertEqual(scores[2]["scoring_mode"], "process_fail_closed")
            self.assertTrue(any("alpha_worker_error=JSONDecodeError" in reason for reason in scores[2]["reasons"]))

        asyncio.run(run())

    def test_timed_out_worker_pool_is_terminated_before_replacement(self):
        executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        future = executor.submit(_sleeping_process_worker, 60.0)
        deadline = time.monotonic() + 10.0
        processes = []
        while time.monotonic() < deadline:
            processes = list((getattr(executor, "_processes", None) or {}).values())
            if processes and all(process.is_alive() for process in processes):
                break
            time.sleep(0.01)
        self.assertTrue(processes)
        self.assertTrue(all(process.is_alive() for process in processes))

        _terminate_process_executor(executor, terminate_grace_sec=0.2)

        self.assertTrue(all(not process.is_alive() for process in processes))
        self.assertTrue(future.done())

    def test_runner_lock_prevents_two_parent_coordinators(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            first = _acquire_portfolio_runner_lock(state_file)
            try:
                with self.assertRaisesRegex(RuntimeError, "already owns state lock"):
                    _acquire_portfolio_runner_lock(state_file)
            finally:
                _release_portfolio_runner_lock(first)
            second = _acquire_portfolio_runner_lock(state_file)
            _release_portfolio_runner_lock(second)

    def test_db_lease_is_cross_host_and_scoped_by_exchange_symbol(self):
        class FakeResult:
            def __init__(self, value):
                self.value = value

            def scalar_one(self):
                return self.value

        class FakeConnection:
            def __init__(self, acquired=True, backend_pid=1234):
                self.acquired = acquired
                self.backend_pid = backend_pid
                self.closed = False
                self.commits = 0
                self.calls = []

            def execute(self, statement, params=None):
                sql = str(statement)
                self.calls.append((sql, dict(params or {})))
                if "pg_try_advisory_lock" in sql:
                    return FakeResult(self.acquired)
                if "pg_advisory_unlock" in sql:
                    return FakeResult(True)
                if "pg_backend_pid" in sql:
                    return FakeResult(self.backend_pid)
                raise AssertionError(sql)

            def commit(self):
                self.commits += 1

            def close(self):
                self.closed = True

        class FakeEngine:
            def __init__(self, connection):
                self.connection = connection

            def connect(self):
                return self.connection

        fake_sqlalchemy = SimpleNamespace(text=lambda statement: statement)
        key = _portfolio_db_lease_key(
            strategy_name="rllm", exchange="binance-mainnet", symbol="BTCUSDT"
        )
        self.assertEqual(
            key,
            _portfolio_db_lease_key(
                strategy_name="rllm", exchange="binance-mainnet", symbol="BTCUSDT"
            ),
        )
        self.assertNotEqual(
            key,
            _portfolio_db_lease_key(
                strategy_name="rllm", exchange="binance-testnet", symbol="BTCUSDT"
            ),
        )

        connection = FakeConnection()
        with patch.dict(sys.modules, {"sqlalchemy": fake_sqlalchemy}):
            lease = _acquire_portfolio_db_lease(
                FakeEngine(connection),
                strategy_name="rllm",
                exchange="binance-mainnet",
                symbol="BTCUSDT",
            )
            _assert_portfolio_db_lease(lease)
            _release_portfolio_db_lease(lease)
        self.assertTrue(connection.closed)
        self.assertTrue(any("pg_advisory_unlock" in sql for sql, _ in connection.calls))

        rejected = FakeConnection(acquired=False)
        with patch.dict(sys.modules, {"sqlalchemy": fake_sqlalchemy}):
            with self.assertRaisesRegex(RuntimeError, "already owned"):
                _acquire_portfolio_db_lease(
                    FakeEngine(rejected),
                    strategy_name="rllm",
                    exchange="binance-mainnet",
                    symbol="BTCUSDT",
                )
        self.assertTrue(rejected.closed)

    def test_execution_schema_migration_takes_global_transaction_lock_first(self):
        class FakeEngine:
            def __init__(self):
                self.calls = []

            def begin(self):
                engine = self

                class Context:
                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return False

                    def execute(self, statement, params=None):
                        engine.calls.append((str(statement), dict(params or {})))

                return Context()

        engine = FakeEngine()
        with patch.dict(sys.modules, {"sqlalchemy": SimpleNamespace(text=lambda statement: statement)}):
            _ensure_trade_executions_table(engine)

        self.assertIn("pg_advisory_xact_lock", engine.calls[0][0])
        self.assertIn("lock_key", engine.calls[0][1])
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS trade_executions" in sql for sql, _ in engine.calls))
        self.assertTrue(any("DO $migration$" in sql for sql, _ in engine.calls))

    def test_open_intent_builder_filters_state_without_mutating_scores(self):
        scores = [
            {"name": "new", "signal_id": "new:1", "active": True, "side": "LONG", "weight": 0.6, "current_close": 100.0, "hold_bars": 12, "stride_bars": 1},
            {"name": "open", "signal_id": "open:1", "active": True, "side": "LONG", "weight": 0.2, "current_close": 100.0, "hold_bars": 12, "stride_bars": 1},
            {"name": "done", "signal_id": "done:1", "active": True, "side": "SHORT", "weight": 0.2, "current_close": 100.0, "hold_bars": 12, "stride_bars": 1},
        ]
        before = json.loads(json.dumps(scores))
        intents = _build_open_intents(
            sleeve_scores=scores,
            state={"open_sleeves": {"open": {}}, "processed_signals": {"done": "done:1"}},
            total_weight=1.0,
            leverage_budget=6.0,
            allocation_mode="research_gross",
            exec_cfg=WaveExecutionConfig(leverage=6),
            entry_timeout_fraction=0.25,
            max_entry_wait_sec=300,
            entry_maker_max_deviation_pct=0.003,
            maker_refresh_interval_sec=60,
        )
        self.assertEqual([intent["sleeve"]["name"] for intent in intents], ["new"])
        self.assertAlmostEqual(intents[0]["margin_fraction"], 0.1)
        self.assertEqual(scores, before)

    def test_open_order_tasks_run_concurrently_and_isolate_one_failure(self):
        async def run():
            starts = {}

            async def fake_cancel(**kwargs):
                return []

            async def fake_open(*, sleeve, **kwargs):
                starts[sleeve["name"]] = asyncio.get_running_loop().time()
                await asyncio.sleep(0.05)
                if sleeve["name"] == "bad":
                    raise RuntimeError("exchange rejected")
                return {"filled_quantity": "0.01", "order": {"status": "FILLED"}}

            intents = [
                {
                    "sleeve": {"name": name, "signal_id": f"{name}:1", "side": "LONG", "current_close": 100.0},
                    "margin_fraction": 0.1,
                    "entry_ttl_sec": 30,
                }
                for name in ("good_a", "bad", "good_b")
            ]
            with patch("execution.portfolio_live._cancel_portfolio_orders_for_sleeve", new=fake_cancel), patch(
                "execution.portfolio_live._open_sleeve", new=fake_open
            ):
                outcomes = await _execute_open_intents(
                    intents=intents,
                    client=object(),
                    executor=object(),
                    exec_cfg=WaveExecutionConfig(dry_run=False, allow_live_orders=True),
                )
            self.assertEqual([outcome["ok"] for outcome in outcomes], [True, False, True])
            self.assertLess(max(starts.values()) - min(starts.values()), 0.03)
            self.assertIn("exchange rejected", outcomes[1]["error"])

        asyncio.run(run())

    def test_open_order_fails_closed_when_stale_order_scan_fails(self):
        async def run():
            opened = []

            async def fake_cancel(**kwargs):
                return [{"status": "scan_failed", "error": "exchange unavailable"}]

            async def fake_open(**kwargs):
                opened.append(kwargs)
                return {"filled_quantity": "0.01", "order": {"status": "FILLED"}}

            intent = {
                "sleeve": {
                    "name": "alpha",
                    "signal_id": "alpha:1",
                    "side": "LONG",
                    "current_close": 100.0,
                },
                "margin_fraction": 0.1,
                "entry_ttl_sec": 30,
            }
            with patch("execution.portfolio_live._cancel_portfolio_orders_for_sleeve", new=fake_cancel), patch(
                "execution.portfolio_live._open_sleeve", new=fake_open
            ):
                outcomes = await _execute_open_intents(
                    intents=[intent],
                    client=object(),
                    executor=object(),
                    exec_cfg=WaveExecutionConfig(dry_run=False, allow_live_orders=True),
                )

            self.assertEqual(opened, [])
            self.assertFalse(outcomes[0]["ok"])
            self.assertEqual(outcomes[0]["replaced"][0]["status"], "scan_failed")
            self.assertIn("cancellation not confirmed", outcomes[0]["error"])

        asyncio.run(run())

    def test_open_order_fails_closed_when_stale_order_cancel_fails(self):
        async def run():
            opened = []

            async def fake_cancel(**kwargs):
                return [
                    {
                        "status": "cancel_failed",
                        "client_order_id": "rllm_pf_alpha_old",
                        "error": "timeout",
                    }
                ]

            async def fake_open(**kwargs):
                opened.append(kwargs)
                return {"filled_quantity": "0.01", "order": {"status": "FILLED"}}

            intent = {
                "sleeve": {
                    "name": "alpha",
                    "signal_id": "alpha:1",
                    "side": "LONG",
                    "current_close": 100.0,
                },
                "margin_fraction": 0.1,
                "entry_ttl_sec": 30,
            }
            with patch("execution.portfolio_live._cancel_portfolio_orders_for_sleeve", new=fake_cancel), patch(
                "execution.portfolio_live._open_sleeve", new=fake_open
            ):
                outcomes = await _execute_open_intents(
                    intents=[intent],
                    client=object(),
                    executor=object(),
                    exec_cfg=WaveExecutionConfig(dry_run=False, allow_live_orders=True),
                )

            self.assertEqual(opened, [])
            self.assertFalse(outcomes[0]["ok"])
            self.assertEqual(outcomes[0]["replaced"][0]["status"], "cancel_failed")
            self.assertIn("cancellation not confirmed", outcomes[0]["error"])

        asyncio.run(run())

    def test_close_order_tasks_run_concurrently_and_isolate_one_failure(self):
        async def run():
            starts = {}

            class Client:
                async def get_ticker_price(self, symbol):
                    return 100.0

            async def fake_close(*, sleeve_state, **kwargs):
                starts[sleeve_state["name"]] = asyncio.get_running_loop().time()
                await asyncio.sleep(0.05)
                if sleeve_state["name"] == "bad":
                    raise RuntimeError("close rejected")
                return {"status": "FILLED", "filled_quantity": sleeve_state["quantity"]}

            async def fake_report(**kwargs):
                return kwargs["order_info"]

            intents = [
                {
                    "key": name,
                    "open_state": {"name": name, "signal_id": f"{name}:1", "side": "LONG", "quantity": "0.01"},
                    "time_exit_due": True,
                    "dynamic_exit_due": False,
                }
                for name in ("good_a", "bad", "good_b")
            ]
            with patch("execution.portfolio_live._close_sleeve", new=fake_close), patch(
                "execution.portfolio_live._attach_exchange_trade_report", new=fake_report
            ):
                outcomes = await _execute_close_intents(
                    intents=intents,
                    client=Client(),
                    executor=object(),
                    exec_cfg=WaveExecutionConfig(dry_run=False, allow_live_orders=True),
                    max_exit_wait_sec=30,
                    exit_maker_max_deviation_pct=0.002,
                    maker_refresh_interval_sec=60,
                )
            self.assertEqual([outcome["ok"] for outcome in outcomes], [True, False, True])
            self.assertLess(max(starts.values()) - min(starts.values()), 0.03)
            self.assertIn("close rejected", outcomes[1]["error"])

        asyncio.run(run())

    def test_close_preserves_partial_fill_when_taker_fallback_fails(self):
        async def run():
            class Client:
                async def place_market(self, **kwargs):
                    raise RuntimeError("market unavailable")

            async def fake_maker(**kwargs):
                return {"status": "PARTIAL_CANCELLED", "filled_quantity": "0.004"}

            with patch("execution.portfolio_live._place_portfolio_maker_order_with_deadline", new=fake_maker):
                result = await _close_sleeve(
                    client=Client(),
                    executor=object(),
                    sleeve_state={"name": "alpha", "signal_id": "alpha:1", "side": "LONG", "quantity": "0.01"},
                    exec_cfg=WaveExecutionConfig(),
                    ttl_sec=30,
                    reference_price=100.0,
                    max_deviation_pct=0.002,
                    refresh_interval_sec=60,
                )
            self.assertEqual(result["status"], "PARTIAL_TAKER_FALLBACK_FAILED")
            self.assertEqual(result["filled_quantity"], "0.004")
            self.assertIn("market unavailable", result["taker_fallback_error"])

        asyncio.run(run())

    def test_db_reservation_batch_is_atomic_and_duplicate_safe(self):
        class FakeResult:
            def __init__(self, row=None):
                self.row = row

            def mappings(self):
                return self

            def one_or_none(self):
                return self.row

        class FakeEngine:
            def __init__(self):
                self.keys = set()
                self.begin_count = 0
                self.updates = []

            def begin(self):
                engine = self

                class Context:
                    def __enter__(self):
                        engine.begin_count += 1
                        return self

                    def __exit__(self, *args):
                        return False

                    def execute(self, statement, params):
                        if str(statement).lstrip().startswith("INSERT"):
                            key = (
                                params["strategy_name"],
                                params["sub_strategy_name"],
                                params["signal_id"],
                                "OPEN",
                                params["exchange"],
                                params["symbol"],
                            )
                            if key in engine.keys:
                                return FakeResult()
                            engine.keys.add(key)
                            return FakeResult(
                                {"sub_strategy_name": params["sub_strategy_name"], "signal_id": params["signal_id"]}
                            )
                        engine.updates.append(dict(params))
                        return FakeResult()

                return Context()

        intents = [
            {"sleeve": {"name": name, "signal_id": f"{name}:1", "side": "LONG", "weight": 0.5}, "margin_fraction": 0.1, "entry_ttl_sec": 30}
            for name in ("a", "b")
        ]
        engine = FakeEngine()
        fake_sqlalchemy = SimpleNamespace(text=lambda statement: statement)
        with patch.dict(sys.modules, {"sqlalchemy": fake_sqlalchemy}):
            first = _reserve_trade_intents(
                engine,
                strategy_name="rllm",
                exchange="binance",
                symbol="BTCUSDT",
                owner_id="owner-1",
                intents=intents,
            )
            second = _reserve_trade_intents(
                engine,
                strategy_name="rllm",
                exchange="binance",
                symbol="BTCUSDT",
                owner_id="owner-2",
                intents=intents,
            )
            other_symbol = _reserve_trade_intents(
                engine,
                strategy_name="rllm",
                exchange="binance",
                symbol="ETHUSDT",
                owner_id="owner-3",
                intents=intents,
            )
        self.assertEqual(first, {("a", "a:1"), ("b", "b:1")})
        self.assertEqual(second, set())
        self.assertEqual(other_symbol, {("a", "a:1"), ("b", "b:1")})
        outcomes = [
            {"intent": intents[0], "reservation_status": "FILLED", "order_status": "FILLED", "filled_quantity": "0.01", "error": None},
            {"intent": intents[1], "reservation_status": "ERROR", "order_status": "ERROR", "filled_quantity": "0", "error": "boom"},
        ]
        with patch.dict(sys.modules, {"sqlalchemy": fake_sqlalchemy}):
            _finish_trade_intents(
                engine,
                strategy_name="rllm",
                exchange="binance",
                symbol="BTCUSDT",
                owner_id="owner-1",
                outcomes=outcomes,
            )
        self.assertEqual(engine.begin_count, 4)
        self.assertEqual([update["status"] for update in engine.updates], ["FILLED", "ERROR"])

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

    def test_live_anchor_waits_only_for_sources_used_by_its_alpha_gates(self):
        with open("configs/live/portfolio_gross385_trainmdd40_2026-07-12.json") as handle:
            portfolio = json.load(handle)
        features = _portfolio_gate_features(portfolio)
        requirements = _freshness_requirements_for_decision(
            symbol="BTCUSDT",
            expected_bar=pd.Timestamp("2026-07-10T12:00:00Z"),
            required_1m=pd.Timestamp("2026-07-10T12:04:00Z"),
            include_premium=any(feature.startswith("premium_") for feature in features),
            include_upbit=any(feature.startswith(("vg_upbit_", "kimchi_")) for feature in features),
            include_alt_pool=any(feature.startswith("vg_alt_") for feature in features),
        )
        keys = {requirement.key for requirement in requirements}
        self.assertIn("bars_binance:BTCUSDT:1m", keys)
        self.assertIn("bars_binance_premium:BTCUSDT:1m", keys)
        self.assertIn("bars_upbit:KRW-BTC:1m", keys)
        self.assertFalse(any(requirement.source == "binance_alt_pool" for requirement in requirements))

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
                    return [{"orderId": 88, "positionSide": "LONG", "side": "BUY", "time": entry_ms, "price": "99"}]

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
            self.assertEqual(restored["entry_fill_price"], 100.0)

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

    def test_post_only_refresh_reports_quantity_weighted_fill_price(self):
        async def run():
            class SequencedExecutor:
                def __init__(self):
                    self.prices = iter((100.0, 110.0))

                async def get_maker_price(self, side, ws_orderbook=None):
                    return next(self.prices)

            client = FakeClient(
                order_statuses=[
                    {"orderId": 101, "status": "NEW", "executedQty": "0.004", "avgPrice": "100"},
                ],
                cancel_responses=[
                    {"status": "CANCELED", "executedQty": "0.006", "avgPrice": "100"},
                    {"status": "CANCELED", "executedQty": "0.004", "avgPrice": "110"},
                ],
            )
            result = await _place_portfolio_maker_order_with_deadline(
                client=client,
                executor=SequencedExecutor(),
                exec_cfg=SimpleNamespace(symbol="BTCUSDT"),
                order_side="BUY",
                quantity=Decimal("0.01"),
                position_side="LONG",
                signal_id="sig-weighted-price",
                sleeve_name="fresh_kimchi_fx",
                ttl_sec=2,
                refresh_interval_sec=1,
                poll_interval_sec=0.05,
            )
            self.assertEqual(result["status"], "FILLED")
            self.assertEqual(Decimal(result["filled_quantity"]), Decimal("0.010"))
            self.assertEqual(Decimal(result["avg_price"]), Decimal("104"))

        asyncio.run(run())

    def test_post_only_reconciles_cumulative_average_quote_not_delta_average(self):
        async def run():
            client = FakeClient(
                order_statuses=[
                    {"orderId": 101, "status": "NEW", "executedQty": "0.001", "avgPrice": "100"},
                ],
                cancel_responses=[
                    {"status": "CANCELED", "executedQty": "0.002", "avgPrice": "110"},
                ],
            )
            result = await _place_portfolio_maker_order_with_deadline(
                client=client,
                executor=FakeExecutor(),
                exec_cfg=SimpleNamespace(symbol="BTCUSDT"),
                order_side="BUY",
                quantity=Decimal("0.002"),
                position_side="LONG",
                signal_id="sig-cumulative-average",
                sleeve_name="legacy-maker",
                ttl_sec=2,
                refresh_interval_sec=1,
                poll_interval_sec=0.05,
            )
            self.assertEqual(result["status"], "FILLED")
            self.assertEqual(Decimal(result["filled_quantity"]), Decimal("0.002"))
            self.assertEqual(Decimal(result["avg_price"]), Decimal("110"))

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
