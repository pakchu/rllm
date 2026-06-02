import json
import tempfile
import unittest
from pathlib import Path

from training.eval_text_trader import _action_json, _candidate_actions, evaluate_text_trader, parse_trader_json


class TestEvalTextTrader(unittest.TestCase):
    def test_candidate_actions_cover_no_trade_and_trade_horizons(self):
        actions = _candidate_actions([48, 96])
        self.assertEqual(actions[0], {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0})
        self.assertEqual(actions[-1], {"gate": "TRADE", "side": "SHORT", "hold_bars": 96})
        self.assertEqual(len(actions), 5)
        self.assertEqual(_action_json(actions[-1]), '{"gate":"TRADE","hold_bars":96,"side":"SHORT"}')

    def test_parse_trader_json_normalizes_invalid_outputs(self):
        self.assertEqual(parse_trader_json('x {"gate":"TRADE","side":"LONG","hold_bars":96} y'), {"gate": "TRADE", "side": "LONG", "hold_bars": 96})
        self.assertEqual(parse_trader_json('{"gate":"NO_TRADE","side":"SHORT","hold_bars":288}'), {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0})
        self.assertEqual(parse_trader_json('bad'), {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0})

    def test_target_echo_eval_reports_perfect_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "trader.jsonl"
            out = Path(td) / "report.json"
            rows = [
                {"task": "trader", "prompt": "P", "target": '{"gate":"NO_TRADE","side":"NONE","hold_bars":0}'},
                {"task": "trader", "prompt": "P", "target": '{"gate":"TRADE","side":"SHORT","hold_bars":96}'},
            ]
            data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            report = evaluate_text_trader(eval_jsonl=str(data), output=str(out), prediction_mode="target_echo")
            self.assertEqual(report["metrics"]["gate_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["hold_accuracy_when_target_trade"], 1.0)
            self.assertEqual(report["metrics"]["exact_action_accuracy"], 1.0)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
