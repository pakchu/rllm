import json
import tempfile
import unittest
from pathlib import Path

from training.event_candidate_pairwise_walkforward import EventCandidatePairwiseWalkForwardCfg, _no_trade_predictions, make_folds


class TestEventCandidatePairwiseWalkForward(unittest.TestCase):
    def test_make_folds_uses_half_open_calendar_windows(self):
        cfg = EventCandidatePairwiseWalkForwardCfg(input_jsonl="x", market_csv="m", output="o", fit_months=12, val_months=6, test_months=6, step_months=6)
        folds = make_folds("2022-01-15 00:00:00", "2024-02-01 00:00:00", cfg)
        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[0].fit_start, "2022-01-01 00:00:00")
        self.assertEqual(folds[0].fit_end, "2023-01-01 00:00:00")
        self.assertEqual(folds[0].val_start, "2023-01-01 00:00:00")
        self.assertEqual(folds[0].test_start, "2023-07-01 00:00:00")
        self.assertEqual(folds[1].fit_start, "2022-07-01 00:00:00")

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
        self.assertEqual([r["signal_pos"] for r in loaded], [1, 2])


if __name__ == "__main__":
    unittest.main()
