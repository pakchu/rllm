import json
import tempfile
import unittest
from pathlib import Path

from training.text_preference_data import build_preference_jsonl, build_preference_pairs


class TestTextPreferenceData(unittest.TestCase):
    def test_build_preference_pairs_keeps_labels_out_of_prompt(self):
        rows = [
            {
                "date": "2025-01-01",
                "signal_pos": 10,
                "prompt": "Analyzer summary: {\"risk_state\":\"NORMAL\"}",
                "target": '{"gate":"TRADE","side":"LONG","hold_bars":96}',
            },
            {
                "date": "2025-01-02",
                "signal_pos": 20,
                "prompt": "Analyzer summary: {\"risk_state\":\"STRESS\"}",
                "target": '{"gate":"NO_TRADE","side":"NONE","hold_bars":0}',
            },
        ]
        pairs = build_preference_pairs(rows, hold_candidates=(48, 96), max_pairs_per_row=2)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all("preferred_step_bars" not in p["prompt"] for p in pairs))
        first = json.loads(pairs[0]["chosen"])
        self.assertEqual(first["gate"], "TRADE")
        self.assertEqual(first["side"], "LONG")
        self.assertEqual(first["hold_bars"], 96)
        rejected = [json.loads(p["rejected"]) for p in pairs[:2]]
        self.assertIn("NO_TRADE", {r["gate"] for r in rejected})
        self.assertIn("SHORT", {r["side"] for r in rejected})
        self.assertTrue(pairs[0]["leakage_guard"]["chosen_rejected_use_future_path_labels_for_training_only"])

    def test_cli_builder_writes_jsonl_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "trader.jsonl"
            out = Path(tmp) / "pref.jsonl"
            summary = Path(tmp) / "summary.json"
            inp.write_text(
                json.dumps(
                    {
                        "prompt": "Analyzer summary: {}",
                        "target": '{"gate":"TRADE","side":"SHORT","hold_bars":48}',
                    }
                )
                + "\n"
            )
            report = build_preference_jsonl(
                input_jsonl=str(inp),
                output_jsonl=str(out),
                summary_output=str(summary),
                hold_candidates="48,96",
                max_pairs_per_row=2,
            )
            self.assertEqual(report["pairs"], 2)
            self.assertTrue(out.exists())
            self.assertTrue(summary.exists())
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertIn("chosen", rows[0])
            self.assertIn("rejected", rows[0])


if __name__ == "__main__":
    unittest.main()
