import json
import tempfile
import unittest
from pathlib import Path

from training.apply_vlm_bias import apply_vlm_bias_report


class TestApplyVlmBias(unittest.TestCase):
    def test_apply_trade_side_biases_rewrites_predictions_and_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "report.json"
            out = Path(td) / "out.json"
            src.write_text(
                json.dumps(
                    {
                        "action_schema": "trade_side",
                        "metrics": {},
                        "action_scores": [
                            {
                                "date": "2025-01-01 00:00:00",
                                "target": "SHORT",
                                "pred": "LONG",
                                "scores": {"LONG": 1.0, "SHORT": 0.4},
                            },
                            {
                                "date": "2025-01-01 00:05:00",
                                "target": "LONG",
                                "pred": "LONG",
                                "scores": {"LONG": 0.2, "SHORT": 0.1},
                            },
                        ],
                    }
                )
            )

            report = apply_vlm_bias_report(
                input_report=str(src),
                output=str(out),
                biases={"LONG": -1.0, "SHORT": 0.0},
            )

            self.assertTrue(out.exists())
            self.assertEqual([r["pred"] for r in report["action_scores"]], ["SHORT", "SHORT"])
            self.assertEqual(
                report["action_scores"][0]["adjusted_scores"],
                {"LONG": 0.0, "SHORT": 0.4},
            )
            self.assertEqual(report["metrics"]["pred_counts"], {"LONG": 0, "SHORT": 2})
            self.assertEqual(report["bias_application"]["biases"], {"LONG": -1.0, "SHORT": 0.0})


if __name__ == "__main__":
    unittest.main()
