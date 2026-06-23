import json
import tempfile
import unittest
from pathlib import Path

from training.apply_month_validation_gate import MonthValidationGateCfg, apply_gate


class TestApplyMonthValidationGate(unittest.TestCase):
    def test_blocks_month_below_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pred = td / "pred.jsonl"
            summary = td / "summary.json"
            out = td / "out.jsonl"
            pred.write_text("\n".join([
                json.dumps({"date": "2026-01-01 00:00:00", "prediction": {"gate": "TRADE", "side": "LONG", "hold_bars": 288}}),
                json.dumps({"date": "2026-02-01 00:00:00", "prediction": {"gate": "TRADE", "side": "SHORT", "hold_bars": 288}}),
            ]) + "\n")
            summary.write_text(json.dumps({"months": [{"month": "2026-01", "selected": {"score": -1.0}}, {"month": "2026-02", "selected": {"score": 2.0}}]}))
            report = apply_gate(MonthValidationGateCfg(predictions_jsonl=str(pred), rolling_summary_json=str(summary), output_jsonl=str(out), threshold=0.5))
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(rows[0]["prediction"]["gate"], "NO_TRADE")
            self.assertEqual(rows[1]["prediction"]["gate"], "TRADE")
            self.assertEqual(report["trade_signals_before"], 2)
            self.assertEqual(report["trade_signals_after"], 1)


if __name__ == "__main__":
    unittest.main()
