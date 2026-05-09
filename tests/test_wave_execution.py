import unittest

from execution.wave_execution import (
    ExecutionDecision,
    WaveExecutionBridge,
    WaveExecutionConfig,
    _StaticSignalGenerator,
    _validate_decision,
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
        self.assertEqual(result["action"], "EXECUTE_SIGNAL")
        self.assertEqual(executor.calls, [])

    async def test_live_mode_delegates_to_wave_executor(self):
        executor = DummyExecutor()
        bridge = WaveExecutionBridge(
            config=WaveExecutionConfig(dry_run=False, interval_minutes=5),
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


if __name__ == "__main__":
    unittest.main()
