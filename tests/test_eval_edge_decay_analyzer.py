import json
import tempfile
import unittest
from pathlib import Path

from training.eval_edge_decay_analyzer import evaluate_edge_decay_analyzer, parse_edge_decay_json


def _row(edge="EDGE_PERSIST"):
    return {
        "task": "edge_decay_analyzer",
        "date": "2025-01-01",
        "prompt": "past only",
        "target": json.dumps(
            {
                "trend_side": "LONG",
                "edge_decay_label": edge,
                "transition_label": "TREND_CONTINUATION",
                "risk_label": "LOW_ADVERSE_EXCURSION",
                "recommended_router_hint": "ALLOW_TREND_SPECIALIST",
            },
            separators=(",", ":"),
        ),
    }


class TestEvalEdgeDecayAnalyzer(unittest.TestCase):
    def test_parse_edge_decay_json_defaults_invalid_fields(self):
        parsed = parse_edge_decay_json('{"trend_side":"bad","edge_decay_label":"edge_persist"}')
        self.assertEqual(parsed["trend_side"], "NONE")
        self.assertEqual(parsed["edge_decay_label"], "EDGE_PERSIST")
        self.assertEqual(parsed["recommended_router_hint"], "LOW_CONFIDENCE_ROUTER")

    def test_target_echo_writes_prediction_jsonl_for_router_backtest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "eval.jsonl"
            out = root / "report.json"
            pred = root / "pred.jsonl"
            inp.write_text(json.dumps(_row()) + "\n" + json.dumps(_row("EDGE_DECAY")) + "\n")
            report = evaluate_edge_decay_analyzer(
                eval_jsonl=str(inp),
                output=str(out),
                prediction_output=str(pred),
                prediction_mode="target_echo",
            )
            self.assertEqual(report["metrics"]["exact_all_keys_accuracy"], 1.0)
            lines = [json.loads(x) for x in pred.read_text().splitlines()]
            self.assertIn("prediction", lines[0])
            self.assertTrue(report["leakage_guard"]["target_echo_is_oracle_only"])


if __name__ == "__main__":
    unittest.main()
