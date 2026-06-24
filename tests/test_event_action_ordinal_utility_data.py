import json
import tempfile
import unittest
from pathlib import Path

from training.event_action_ordinal_utility_data import EventActionOrdinalUtilityConfig, build_ordinal_jsonl, build_ordinal_rows, label_for_row


def row(util, mae=0.01):
    return {
        "date": "2025-01-01",
        "signal_pos": 1,
        "prompt": "\n".join([
            "You are an action value judge for BTCUSDT futures.",
            "Output exactly one label: TAKE or SKIP.",
            "Date: 2025-01-01",
            "Past-only state: {}",
            "Candidate action: {\"side\":\"LONG\"}",
        ]),
        "action": {"side": "LONG", "hold_bars": 72},
        "action_audit": {"rank_utility": util, "mae": mae},
    }


class TestEventActionOrdinalUtilityData(unittest.TestCase):
    def test_label_thresholds(self):
        cfg = EventActionOrdinalUtilityConfig("in", "out", avoid_below=-0.01, mid_at=0.004, high_at=0.012, max_mae_high=0.018)
        self.assertEqual(label_for_row(row(-0.02), cfg), "AVOID")
        self.assertEqual(label_for_row(row(0.0), cfg), "LOW")
        self.assertEqual(label_for_row(row(0.006), cfg), "MID")
        self.assertEqual(label_for_row(row(0.02, mae=0.01), cfg), "HIGH")
        self.assertEqual(label_for_row(row(0.02, mae=0.03), cfg), "MID")

    def test_prompt_removes_binary_instruction_and_keeps_candidate(self):
        cfg = EventActionOrdinalUtilityConfig("in", "out")
        rows = build_ordinal_rows([row(0.02)], cfg)
        self.assertEqual(rows[0]["target"], "HIGH")
        self.assertNotIn("TAKE or SKIP", rows[0]["prompt"])
        self.assertIn("AVOID, LOW, MID, or HIGH", rows[0]["prompt"])
        self.assertIn("Candidate action:", rows[0]["prompt"])
        self.assertTrue(rows[0]["leakage_guard"]["target_uses_future_utility_for_training_only"])

    def test_cli_writes_jsonl_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "value.jsonl"
            out = Path(tmp) / "ord.jsonl"
            summary = Path(tmp) / "summary.json"
            inp.write_text(json.dumps(row(0.02)) + "\n")
            report = build_ordinal_jsonl(input_jsonl=str(inp), output_jsonl=str(out), summary_output=str(summary))
            self.assertEqual(report["rows"], 1)
            self.assertEqual(report["target_counts"], {"HIGH": 1})
            self.assertTrue(out.exists())
            self.assertTrue(summary.exists())


if __name__ == "__main__":
    unittest.main()
