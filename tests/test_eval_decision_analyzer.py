import json
import tempfile
import unittest
from pathlib import Path

from training.eval_decision_analyzer import evaluate_decision_analyzer, parse_decision_json


class TestEvalDecisionAnalyzer(unittest.TestCase):
    def test_parse_decision_json_defaults_invalid_action_to_abstain(self):
        parsed = parse_decision_json('{"decision":"TRADE_TREND","action_side":"NONE","confidence":"HIGH","rationale_class":"EDGE_PERSIST_CONTINUATION"}')
        self.assertEqual(parsed["decision"], "ABSTAIN")
        self.assertEqual(parsed["action_side"], "NONE")

    def test_target_echo_writes_predictions_and_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "decision.jsonl"
            out = root / "eval.json"
            pred = root / "pred.jsonl"
            rows = [
                {
                    "task": "decision_analyzer",
                    "date": "2025-01-01 00:00:00",
                    "prompt": "past context",
                    "target": json.dumps({"decision": "TRADE_TREND", "action_side": "LONG", "confidence": "HIGH", "rationale_class": "EDGE_PERSIST_CONTINUATION"}),
                },
                {
                    "task": "decision_analyzer",
                    "date": "2025-01-02 00:00:00",
                    "prompt": "past context",
                    "target": json.dumps({"decision": "ABSTAIN", "action_side": "NONE", "confidence": "LOW", "rationale_class": "LOW_CONFIDENCE_SKIP"}),
                },
            ]
            src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            report = evaluate_decision_analyzer(eval_jsonl=str(src), output=str(out), prediction_output=str(pred), prediction_mode="target_echo")
            self.assertEqual(report["metrics"]["exact_all_keys_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["decision_action_accuracy"], 1.0)
            written = [json.loads(line) for line in pred.read_text().splitlines()]
            self.assertEqual(len(written), 2)
            self.assertIn("prediction", written[0])


if __name__ == "__main__":
    unittest.main()
