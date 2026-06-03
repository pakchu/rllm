import json
import tempfile
import unittest
from pathlib import Path

from training.analyzer_state_edge_report import (
    bucket_key,
    evaluate_policy,
    main,
    prediction_record,
    select_bucket_policy,
)


def _row(
    regime: str,
    trend_side: str,
    same_ret: float,
    opp_ret: float,
    date: str = "2025-01-01 00:00:00",
):
    summary = {
        "regime": regime,
        "trend_alignment": "BULL" if trend_side == "LONG" else "BEAR",
        "location": "MID",
        "volatility_level": "LOW",
        "risk_state": "CALM",
        "sequence_stats": {"wide_or_extreme": 0},
    }
    return {
        "date": date,
        "signal_pos": 10,
        "prompt": "Past-only analyzer summary: " + json.dumps(summary, separators=(",", ":")),
        "target": json.dumps({"decision": "ABSTAIN", "action_side": "NONE"}),
        "source_edge_target": {"trend_side": trend_side},
        "path_diagnostics": {
            "long_same": {"net_return": same_ret, "mae": 0.002},
            "long_opposite": {"net_return": opp_ret, "mae": 0.003},
        },
    }


class TestAnalyzerStateEdgeReport(unittest.TestCase):
    def test_selects_bucket_policy_on_train_only(self):
        fields = ("regime", "trend_alignment")
        train = [
            _row("TREND", "LONG", 0.01, -0.005, "2025-01-01 00:00:00"),
            _row("TREND", "LONG", 0.008, -0.004, "2025-01-01 00:05:00"),
            _row("REVERSAL", "LONG", -0.004, 0.009, "2025-01-01 00:10:00"),
            _row("REVERSAL", "LONG", -0.003, 0.008, "2025-01-01 00:15:00"),
        ]
        policy = select_bucket_policy(train, fields, min_train_count=2, min_mean_return=0.001, require_positive_ci=False, max_buckets=10)
        self.assertEqual(policy[bucket_key(train[0], fields)], "TREND")
        self.assertEqual(policy[bucket_key(train[2], fields)], "FADE")
        out = evaluate_policy(train, fields, policy)
        self.assertEqual(out["trades"], 4)
        self.assertEqual(out["action_counts"]["TREND"], 2)
        self.assertEqual(out["action_counts"]["FADE"], 2)

    def test_prediction_uses_past_trend_side_not_target_side(self):
        fields = ("regime",)
        row = _row("REVERSAL", "SHORT", -0.01, 0.02)
        row["target"] = json.dumps({"decision": "FADE_TREND", "action_side": "SHORT"})
        policy = {bucket_key(row, fields): "FADE"}
        pred = prediction_record(row, fields, policy)
        payload = json.loads(pred["prediction"])
        self.assertEqual(payload["decision"], "FADE_TREND")
        self.assertEqual(payload["action_side"], "LONG")

    def test_cli_writes_report_and_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            oos = root / "oos.jsonl"
            out = root / "report.json"
            preds = root / "preds"
            rows = [_row("TREND", "LONG", 0.01, -0.002), _row("TREND", "LONG", 0.008, -0.001)]
            text = "\n".join(json.dumps(r) for r in rows) + "\n"
            train.write_text(text)
            val.write_text(text)
            oos.write_text(text)
            import sys

            old = sys.argv
            try:
                sys.argv = [
                    "prog",
                    "--train-jsonl",
                    str(train),
                    "--val-jsonl",
                    str(val),
                    "--oos-jsonl",
                    str(oos),
                    "--output",
                    str(out),
                    "--prediction-output-dir",
                    str(preds),
                    "--bucket-fields",
                    "regime,trend_alignment",
                    "--min-train-count",
                    "2",
                    "--min-mean-return",
                    "0.001",
                ]
                main()
            finally:
                sys.argv = old
            report = json.loads(out.read_text())
            self.assertEqual(report["selection"]["selected_buckets"], 1)
            self.assertTrue((preds / "oos_predictions.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
