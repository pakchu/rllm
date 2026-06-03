import json
import tempfile
import unittest
from pathlib import Path

from training.decision_feature_learnability import (
    NaiveBayesModel,
    evaluate,
    flatten_summary_features,
    main,
    parse_jsonish,
    prediction_record,
    record_features,
)


def _row(decision: str, regime: str = "TREND", side: str = "LONG", date: str = "2025-01-01 00:00:00"):
    summary = {
        "regime": regime,
        "trend_alignment": "BULL" if side == "LONG" else "BEAR",
        "evidence": {"momentum_8h_pct": 1.2 if side == "LONG" else -1.2, "range_position": 0.7},
        "context_tags": ["TAG_A", regime],
    }
    action = "NONE" if decision == "ABSTAIN" else side
    return {
        "date": date,
        "signal_pos": 10,
        "prompt": "Header\nPast-only analyzer summary: " + json.dumps(summary, separators=(",", ":")),
        "target": json.dumps({"decision": decision, "action_side": action}, separators=(",", ":")),
        "source_edge_target": {"trend_side": side},
    }


class TestDecisionFeatureLearnability(unittest.TestCase):
    def test_parses_summary_from_prompt_and_flattens_features(self):
        row = _row("TRADE_TREND", "TREND", "LONG")
        summary = parse_jsonish(row["prompt"])
        feats = flatten_summary_features(summary)
        self.assertEqual(summary["regime"], "TREND")
        self.assertEqual(feats["regime"], "TREND")
        self.assertEqual(feats["context_tags__has__TREND"], "1")
        self.assertIn("evidence.momentum_8h_pct", feats)
        self.assertEqual(record_features(row)["trend_alignment"], "BULL")

    def test_naive_bayes_learns_separable_toy_rows(self):
        train = [
            _row("TRADE_TREND", "TREND", "LONG", "2025-01-01 00:00:00"),
            _row("TRADE_TREND", "TREND", "LONG", "2025-01-01 00:05:00"),
            _row("ABSTAIN", "RANGE", "LONG", "2025-01-01 00:10:00"),
            _row("ABSTAIN", "RANGE", "LONG", "2025-01-01 00:15:00"),
        ]
        model = NaiveBayesModel.fit(train)
        out = evaluate(model, train)
        self.assertGreaterEqual(out["accuracy"], 0.75)
        pred = prediction_record(train[0], model.predict(train[0]))
        self.assertIn("prediction", pred)
        payload = json.loads(pred["prediction"])
        self.assertIn(payload["decision"], {"TRADE_TREND", "FADE_TREND", "ABSTAIN"})
        self.assertEqual(payload["action_side"], "LONG")

    def test_prediction_side_uses_past_trend_not_target_action_side(self):
        row = _row("TRADE_TREND", "TREND", "SHORT")
        row["target"] = json.dumps({"decision": "TRADE_TREND", "action_side": "LONG"})
        pred = prediction_record(row, "TRADE_TREND")
        self.assertEqual(json.loads(pred["prediction"])["action_side"], "SHORT")
        no_side = dict(row)
        no_side["source_edge_target"] = {}
        self.assertEqual(json.loads(prediction_record(no_side, "TRADE_TREND")["prediction"])["decision"], "ABSTAIN")

    def test_cli_writes_report_and_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            out = root / "report.json"
            preds = root / "preds"
            rows = [_row("TRADE_TREND", "TREND", "LONG"), _row("ABSTAIN", "RANGE", "LONG")]
            text = "\n".join(json.dumps(r) for r in rows) + "\n"
            train.write_text(text)
            val.write_text(text)
            import sys

            old = sys.argv
            try:
                sys.argv = [
                    "prog",
                    "--train-jsonl",
                    str(train),
                    "--val-jsonl",
                    str(val),
                    "--output",
                    str(out),
                    "--prediction-output-dir",
                    str(preds),
                ]
                main()
            finally:
                sys.argv = old
            report = json.loads(out.read_text())
            self.assertIn("splits", report)
            self.assertTrue((preds / "val_predictions.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
