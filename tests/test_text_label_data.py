import json
import tempfile
import unittest
from pathlib import Path

from training.text_label_data import build_label_jsonl, build_label_rows
from training.eval_text_label import evaluate_text_label, parse_label


class TestTextLabelData(unittest.TestCase):
    def test_build_plain_gate_rows(self):
        rows = [{"prompt": "Header\nAnalyzer summary: S", "target": '{"gate":"NO_TRADE"}', "date": "d"}]
        out = build_label_rows(rows, key="gate")
        self.assertEqual(out[0]["target"], "NO_TRADE")
        self.assertIn("Output exactly one label", out[0]["prompt"])
        self.assertNotIn("{\"gate\"", out[0]["target"])

    def test_parse_label_and_target_echo(self):
        self.assertEqual(parse_label("NO_TRADE\n", key="gate"), "NO_TRADE")
        self.assertEqual(parse_label("TRADE.", key="gate"), "TRADE")
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.jsonl"
            out = Path(td) / "out.jsonl"
            rep = Path(td) / "rep.json"
            inp.write_text(json.dumps({"prompt": "Analyzer summary: S", "target": '{"gate":"TRADE"}'}) + "\n")
            summary = build_label_jsonl(input_jsonl=str(inp), output_jsonl=str(out), key="gate", summary_output=str(rep))
            self.assertEqual(summary["target_counts"], {"TRADE": 1})
            eval_rep = evaluate_text_label(eval_jsonl=str(out), output=str(Path(td) / "eval.json"), key="gate")
            self.assertEqual(eval_rep["metrics"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
