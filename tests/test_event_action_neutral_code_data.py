import json
import tempfile
import unittest
from pathlib import Path

from training.event_action_neutral_code_data import NeutralCodeConfig, build_code_rows, build_neutral_code_jsonl


class TestEventActionNeutralCodeData(unittest.TestCase):
    def test_build_code_rows_rewrites_targets_and_prompt(self):
        rows = [{"prompt": "Output exactly one label: AVOID, LOW, MID, or HIGH.\nDefinitions: old\nCandidate action: {}", "target": "HIGH", "date": "d", "signal_pos": 1}]
        out = build_code_rows(rows, NeutralCodeConfig("in", "out"))
        self.assertEqual(out[0]["target"], "Q4")
        self.assertEqual(out[0]["semantic_target"], "HIGH")
        self.assertIn("Output exactly one code", out[0]["prompt"])
        self.assertNotIn("Output exactly one label", out[0]["prompt"])
        self.assertNotIn("Definitions: old", out[0]["prompt"])
        self.assertTrue(out[0]["leakage_guard"]["semantic_label_not_output_token"])

    def test_cli_writes_jsonl_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "ord.jsonl"
            out = Path(tmp) / "code.jsonl"
            summary = Path(tmp) / "summary.json"
            inp.write_text(json.dumps({"prompt": "p", "target": "AVOID", "date": "d", "signal_pos": 1}) + "\n")
            report = build_neutral_code_jsonl(input_jsonl=str(inp), output_jsonl=str(out), summary_output=str(summary))
            self.assertEqual(report["target_counts"], {"Q1": 1})
            self.assertTrue(out.exists())
            self.assertTrue(summary.exists())


if __name__ == "__main__":
    unittest.main()
