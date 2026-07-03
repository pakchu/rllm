import json
import tempfile
import unittest
from pathlib import Path

from training.eval_text_json_key import _candidate_json, evaluate_text_json_key, parse_key_json


class TestEvalTextJsonKey(unittest.TestCase):
    def test_parse_key_json(self):
        self.assertEqual(parse_key_json('x {"side":"SHORT"}', key="side"), "SHORT")
        self.assertEqual(parse_key_json('{"gate":"TRADE"}', key="gate"), "TRADE")
        self.assertEqual(parse_key_json('bad', key="gate"), "NO_TRADE")
        self.assertEqual(parse_key_json('{"gate":"BAD"}', key="gate"), "NO_TRADE")
        self.assertEqual(parse_key_json('{"action":"SHORT"}', key="action"), "SHORT")
        self.assertEqual(parse_key_json('{"action":"BAD"}', key="action"), "NO_TRADE")
        self.assertEqual(parse_key_json('{"side_pair":"inverse"}', key="side_pair"), "INVERSE")
        self.assertEqual(parse_key_json('{"side_pair":"bad"}', key="side_pair"), "NORMAL")
        self.assertEqual(parse_key_json('{"direction_regime":"LOW_SCORE_WINS"}', key="direction_regime"), "LOW_SCORE_WINS")
        self.assertEqual(parse_key_json('{"direction_regime":"BAD"}', key="direction_regime"), "ABSTAIN")
        self.assertEqual(parse_key_json('{"trust_score_rank":"LOW"}', key="trust_score_rank"), "LOW")

    def test_candidate_json_single_key_shape(self):
        self.assertEqual(_candidate_json("side", "LONG"), '{"side": "LONG"}')
        self.assertEqual(_candidate_json("action", "SHORT"), '{"action": "SHORT"}')
        self.assertEqual(_candidate_json("side_pair", "INVERSE"), '{"side_pair": "inverse"}')
        self.assertEqual(_candidate_json("direction_regime", "LOW_SCORE_WINS"), '{"direction_regime": "LOW_SCORE_WINS"}')
        self.assertEqual(_candidate_json("trust_score_rank", "LOW"), '{"trust_score_rank": "LOW"}')

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
