import json
import tempfile
import unittest
from pathlib import Path

from training.compose_gate_side_policy import compose_gate_side_reports


class TestComposeGateSidePolicy(unittest.TestCase):
    def test_compose_gate_and_side_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gate_path = tmp / "gate.json"
            side_path = tmp / "side.json"
            out_path = tmp / "composed.json"

            gate_report = {
                "action_schema": "trade_gate",
                "action_scores": [
                    {
                        "date": "2025-01-01 00:00:00",
                        "target": "TRADE",
                        "next_return": 0.01,
                        "adjusted_scores": {"TRADE": -0.1, "NO_TRADE": -1.0},
                    },
                    {
                        "date": "2025-01-01 00:05:00",
                        "target": "NO_TRADE",
                        "next_return": 0.0,
                        "adjusted_scores": {"TRADE": -2.0, "NO_TRADE": -0.1},
                    },
                ],
            }
            side_report = {
                "action_schema": "trade_side",
                "action_scores": [
                    {
                        "date": "2025-01-01 00:00:00",
                        "target": "LONG",
                        "next_return": 0.01,
                        "adjusted_scores": {"LONG": -0.2, "SHORT": -1.5},
                    }
                ],
            }
            gate_path.write_text(json.dumps(gate_report))
            side_path.write_text(json.dumps(side_report))

            out = compose_gate_side_reports(
                gate_report_path=str(gate_path),
                side_report_path=str(side_path),
                output_path=str(out_path),
            )

            self.assertEqual(out["action_schema"], "buy_hold_sell")
            self.assertEqual(out["composition"]["side_date_overlap"], 1)
            self.assertEqual(len(out["action_scores"]), 2)
            self.assertEqual(out["action_scores"][0]["target"], "BUY")
            self.assertEqual(out["action_scores"][0]["pred"], "BUY")
            self.assertEqual(out["action_scores"][1]["target"], "HOLD")
            self.assertEqual(out["action_scores"][1]["pred"], "HOLD")
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
