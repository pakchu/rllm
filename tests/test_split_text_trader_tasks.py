import json
import tempfile
import unittest
from pathlib import Path

from training.split_text_trader_tasks import split_trader_jsonl, split_trader_rows


class TestSplitTextTraderTasks(unittest.TestCase):
    def test_split_rows_keeps_gate_all_and_side_trade_only(self):
        rows = [
            {"prompt": "Header\nAnalyzer summary: {a}", "target": '{"gate":"NO_TRADE","side":"NONE"}', "date": "d0"},
            {"prompt": "Header\nAnalyzer summary: {b}", "target": '{"gate":"TRADE","side":"SHORT"}', "date": "d1"},
        ]
        gate, side = split_trader_rows(rows)
        self.assertEqual(len(gate), 2)
        self.assertEqual(len(side), 1)
        self.assertEqual(json.loads(gate[0]["target"]), {"gate": "NO_TRADE"})
        self.assertEqual(json.loads(side[0]["target"]), {"side": "SHORT"})
        self.assertIn("key gate", gate[0]["prompt"])
        self.assertIn("key side", side[0]["prompt"])
        self.assertIn("{b}", side[0]["prompt"])

    def test_cli_helper_writes_outputs_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "trader.jsonl"
            gate = Path(td) / "gate.jsonl"
            side = Path(td) / "side.jsonl"
            summary = Path(td) / "summary.json"
            inp.write_text(json.dumps({"prompt": "Analyzer summary: {}", "target": '{"gate":"TRADE","side":"LONG"}'}) + "\n")
            rep = split_trader_jsonl(input_jsonl=str(inp), gate_output=str(gate), side_output=str(side), summary_output=str(summary))
            self.assertEqual(rep["records"], {"input": 1, "gate": 1, "side": 1})
            self.assertTrue(gate.exists())
            self.assertTrue(side.exists())
            self.assertTrue(summary.exists())


if __name__ == "__main__":
    unittest.main()
