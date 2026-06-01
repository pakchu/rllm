import json
import tempfile
import unittest
from pathlib import Path

from training.eval_text_json_key import evaluate_text_json_key, parse_key_json


class TestEvalTextJsonKey(unittest.TestCase):
    def test_parse_key_json(self):
        self.assertEqual(parse_key_json('x {"side":"SHORT"}', key="side"), "SHORT")
        self.assertEqual(parse_key_json('{"gate":"TRADE"}', key="gate"), "TRADE")
        self.assertEqual(parse_key_json('bad', key="gate"), "NO_TRADE")

    def test_target_echo_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "side.jsonl"
            out = Path(td) / "report.json"
            rows = [
                {"prompt": "P", "target": '{"side":"LONG"}'},
                {"prompt": "P", "target": '{"side":"SHORT"}'},
            ]
            data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            report = evaluate_text_json_key(eval_jsonl=str(data), output=str(out), key="side")
            self.assertEqual(report["metrics"]["accuracy"], 1.0)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
