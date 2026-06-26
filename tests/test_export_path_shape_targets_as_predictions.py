import json
import tempfile
import unittest
from pathlib import Path

from training.export_path_shape_targets_as_predictions import PathShapeTargetExportCfg, _parse_target, export_targets


class TestExportPathShapeTargetsAsPredictions(unittest.TestCase):
    def test_parse_target_uses_max_hold_bars(self):
        out = _parse_target('{"gate":"TRADE","side":"LONG","max_hold_bars":144,"target_pct":1.0,"stop_pct":0.6}')
        self.assertEqual(out["hold_bars"], 144)
        self.assertEqual(out["side"], "LONG")

    def test_export_writes_prediction_rows(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.jsonl"
            out = Path(td) / "out.jsonl"
            inp.write_text(json.dumps({"date": "2025-01-01 00:00:00", "signal_pos": 1, "target": {"gate": "NO_TRADE"}}) + "\n")
            rep = export_targets(PathShapeTargetExportCfg(str(inp), str(out)))
            row = json.loads(out.read_text().strip())
            self.assertEqual(rep["rows"], 1)
            self.assertEqual(row["prediction"]["gate"], "NO_TRADE")
            self.assertTrue(row["leakage_guard"]["prediction_is_future_target_echo"])


if __name__ == "__main__":
    unittest.main()
