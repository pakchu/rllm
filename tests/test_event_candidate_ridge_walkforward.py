import json
import tempfile
import unittest
from pathlib import Path

from training.event_candidate_ridge_walkforward import _no_trade_predictions, _passes_validation, EventCandidateRidgeWalkForwardCfg


class TestEventCandidateRidgeWalkForward(unittest.TestCase):
    def test_no_trade_predictions_emit_one_row_per_signal(self):
        rows = [
            {"date": "2024-01-01 00:00:00", "signal_pos": 1, "side": "LONG"},
            {"date": "2024-01-01 00:00:00", "signal_pos": 1, "side": "SHORT"},
            {"date": "2024-01-02 00:00:00", "signal_pos": 2, "side": "LONG"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pred.jsonl"
            summary = _no_trade_predictions(rows, str(out), reason="unit")
            loaded = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(summary["rows"], 2)
        self.assertEqual([r["prediction"]["gate"] for r in loaded], ["NO_TRADE", "NO_TRADE"])

    def test_validation_gate_collects_reasons(self):
        cfg = EventCandidateRidgeWalkForwardCfg(
            input_jsonl="x",
            market_csv="m",
            output="o",
            min_val_trades=5,
            min_val_cagr_pct=10,
            min_val_ratio=1,
            max_val_strict_mdd_pct=15,
            max_val_p_value=0.2,
        )
        passed, reasons = _passes_validation(
            {"trade_entries": 4, "cagr_pct": 1, "cagr_to_strict_mdd": 0.5, "strict_mdd_pct": 20},
            {"p_value_mean_ret_approx": 0.5},
            cfg,
        )
        self.assertFalse(passed)
        self.assertIn("val_trades_below_min", reasons)
        self.assertIn("val_mdd_above_max", reasons)


if __name__ == "__main__":
    unittest.main()
