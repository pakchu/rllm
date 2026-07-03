import os
import tempfile
import unittest
from pathlib import Path

from execution.wave_execution import (
    ExecutionDecision,
    WaveExecutionBridge,
    WaveExecutionConfig,
    _StaticSignalGenerator,
    _load_api_credentials,
    _validate_config,
    _validate_decision,
    evaluate_execution_gate,
    load_env_file,
)


class DummyExecutor:
    def __init__(self):
        self.calls = []

    async def initialize(self):
        self.calls.append(("initialize", {}))

    async def handle_signal(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    async def close_position(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"ok": True}


class DummyClient:
    async def sync_time(self):
        return 0

    async def aclose(self):
        return None


class WaveExecutionTests(unittest.IsolatedAsyncioTestCase):
    def test_static_signal_generator_params(self):
        gen = _StaticSignalGenerator(atr_period=15, pt_mult=3.75, max_holding_bars=288)
        self.assertEqual(gen.BEST_PARAMS["atr_period"], 15)
        self.assertEqual(gen.BEST_PARAMS["holding_period"], 288)
        self.assertEqual(gen.generate_signal(), ("HOLD", 0.5, None))

    def test_validate_decision_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            _validate_decision(ExecutionDecision("LONG", 1.5, 100.0, 1.0, "x"))
        with self.assertRaises(ValueError):
            _validate_decision(ExecutionDecision("LONG", 0.5, 0.0, 1.0, "x"))
        with self.assertRaises(ValueError):
            _validate_decision(ExecutionDecision("LONG", 0.5, 100.0, 1.0, ""))

    async def test_dry_run_does_not_call_executor(self):
        executor = DummyExecutor()
        bridge = WaveExecutionBridge(
            config=WaveExecutionConfig(dry_run=True),
            client=DummyClient(),
            executor=executor,
        )
        result = await bridge.execute_decision(ExecutionDecision("LONG", 0.7, 100.0, 2.0, "sig-1"))
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["action"], "BLOCKED")
        self.assertIn("BEAR required", result["gate_reason"])
        self.assertEqual(executor.calls, [])

    async def test_live_mode_delegates_to_wave_executor(self):
        executor = DummyExecutor()
        bridge = WaveExecutionBridge(
            config=WaveExecutionConfig(dry_run=False, allow_live_orders=True, manual_regime="BEAR", interval_minutes=5),
            client=DummyClient(),
            executor=executor,
        )
        result = await bridge.execute_decision(ExecutionDecision("SHORT", 0.8, 100.0, 2.0, "sig-2"))
        self.assertFalse(result["dry_run"])
        self.assertEqual(len(executor.calls), 2)
        args, kwargs = executor.calls[1]
        self.assertEqual(args[:5], ("SHORT", 0.8, 100.0, 2.0, "sig-2"))
        self.assertEqual(args[5], 5)
        self.assertEqual(kwargs, {"ws_client": None})

    def test_manual_bear_regime_gate_approves_trade(self):
        cfg = WaveExecutionConfig(dry_run=True, manual_regime="BEAR")
        decision = ExecutionDecision("SHORT", 0.8, 100.0, 1.0, "sig-bear")
        gate = evaluate_execution_gate(cfg, decision)
        self.assertTrue(gate.allowed)
        self.assertEqual(gate.action, "EXECUTE_SIGNAL")

    def test_manual_regime_gate_blocks_non_bear_trade(self):
        cfg = WaveExecutionConfig(dry_run=True, manual_regime="UNKNOWN")
        decision = ExecutionDecision("SHORT", 0.8, 100.0, 1.0, "sig-unknown")
        gate = evaluate_execution_gate(cfg, decision)
        self.assertFalse(gate.allowed)
        self.assertIn("BEAR required", gate.reason)

    def test_allowed_signal_gate_can_make_bear_pilot_short_only(self):
        cfg = WaveExecutionConfig(dry_run=True, manual_regime="BEAR", allowed_signals=("SHORT",))
        decision = ExecutionDecision("LONG", 0.8, 100.0, 1.0, "sig-long")
        gate = evaluate_execution_gate(cfg, decision)
        self.assertFalse(gate.allowed)
        self.assertIn("not in allowed_signals", gate.reason)

    def test_live_orders_require_explicit_allow_flag(self):
        cfg = WaveExecutionConfig(dry_run=False, manual_regime="BEAR", allow_live_orders=False)
        decision = ExecutionDecision("SHORT", 0.8, 100.0, 1.0, "sig-live")
        gate = evaluate_execution_gate(cfg, decision)
        self.assertFalse(gate.allowed)
        self.assertIn("allow_live_orders", gate.reason)

    def test_config_rejects_invalid_regime(self):
        with self.assertRaises(ValueError):
            _validate_config(WaveExecutionConfig(manual_regime="CRAB"))  # type: ignore[arg-type]

    def test_loads_wave_trading_env_without_overriding_existing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("BINANCE_TESTNET_API_KEY=file-key\nBINANCE_TESTNET_API_SECRET=file-secret\n")
            old_key = os.environ.get("BINANCE_TESTNET_API_KEY")
            old_secret = os.environ.get("BINANCE_TESTNET_API_SECRET")
            try:
                os.environ["BINANCE_TESTNET_API_KEY"] = "existing-key"
                os.environ.pop("BINANCE_TESTNET_API_SECRET", None)
                loaded = load_env_file(env_file)
                self.assertEqual(loaded["BINANCE_TESTNET_API_KEY"], "file-key")
                self.assertEqual(os.environ["BINANCE_TESTNET_API_KEY"], "existing-key")
                self.assertEqual(os.environ["BINANCE_TESTNET_API_SECRET"], "file-secret")
            finally:
                if old_key is None:
                    os.environ.pop("BINANCE_TESTNET_API_KEY", None)
                else:
                    os.environ["BINANCE_TESTNET_API_KEY"] = old_key
                if old_secret is None:
                    os.environ.pop("BINANCE_TESTNET_API_SECRET", None)
                else:
                    os.environ["BINANCE_TESTNET_API_SECRET"] = old_secret

    def test_credentials_can_be_loaded_from_wave_trading_env_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("BINANCE_TESTNET_API_KEY=k\nBINANCE_TESTNET_API_SECRET=s\n")
            old_dry = os.environ.get("RLLM_EXECUTION_DRY_RUN")
            old_key = os.environ.pop("BINANCE_TESTNET_API_KEY", None)
            old_secret = os.environ.pop("BINANCE_TESTNET_API_SECRET", None)
            try:
                os.environ["RLLM_EXECUTION_DRY_RUN"] = "false"
                self.assertEqual(_load_api_credentials(True, str(root)), ("k", "s"))
            finally:
                if old_dry is None:
                    os.environ.pop("RLLM_EXECUTION_DRY_RUN", None)
                else:
                    os.environ["RLLM_EXECUTION_DRY_RUN"] = old_dry
                if old_key is not None:
                    os.environ["BINANCE_TESTNET_API_KEY"] = old_key
                if old_secret is not None:
                    os.environ["BINANCE_TESTNET_API_SECRET"] = old_secret


if __name__ == "__main__":
    unittest.main()
