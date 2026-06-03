import json
import tempfile
import unittest
from pathlib import Path

from training.online_state_memory_report import choose_action, main, prediction_record, run_online
from training.analyzer_state_edge_report import _path_net
from training.online_state_memory_report import example_from_row


def _row(pos: int, regime: str, trend_side: str, same_ret: float, opp_ret: float):
    summary = {
        "regime": regime,
        "trend_alignment": "BULL" if trend_side == "LONG" else "BEAR",
        "location": "MID",
        "volatility_level": "LOW",
        "risk_state": "CALM",
        "sequence_stats": {"wide_or_extreme": 0, "rally_or_up": 4, "drop_or_down": 1},
    }
    return {
        "date": f"2025-01-01 00:{pos % 60:02d}:00",
        "signal_pos": pos,
        "prompt": "Past-only analyzer summary: " + json.dumps(summary, separators=(",", ":")),
        "target": json.dumps({"decision": "ABSTAIN", "action_side": "NONE"}),
        "source_edge_target": {"trend_side": trend_side},
        "path_diagnostics": {
            "long_same": {"net_return": same_ret, "mae": 0.002},
            "long_opposite": {"net_return": opp_ret, "mae": 0.003},
        },
    }


class TestOnlineStateMemoryReport(unittest.TestCase):
    def test_memory_uses_only_matured_prior_examples(self):
        fields = ("regime", "trend_alignment")
        rows = {
            "train": [_row(0, "TREND", "LONG", 0.01, -0.01)],
            "val": [_row(10, "TREND", "LONG", 0.01, -0.01), _row(120, "TREND", "LONG", 0.01, -0.01)],
        }
        report, preds = run_online(
            rows,
            fields,
            hold_bars=100,
            top_k=8,
            min_similarity=1.0,
            min_neighbors=1,
            min_mean_return=0.001,
            mae_penalty=0.0,
            recency_halflife_bars=0.0,
        )
        first = json.loads(preds["val"][0]["prediction"])
        second = json.loads(preds["val"][1]["prediction"])
        self.assertEqual(first["decision"], "ABSTAIN")
        self.assertEqual(second["decision"], "TRADE_TREND")
        self.assertEqual(report["val"]["trades"], 1)

    def test_prediction_side_uses_past_trend_side(self):
        row = _row(200, "REVERSAL", "SHORT", -0.01, 0.02)
        row["target"] = json.dumps({"decision": "FADE_TREND", "action_side": "SHORT"})
        pred = prediction_record(row, "FADE")
        payload = json.loads(pred["prediction"])
        self.assertEqual(payload["decision"], "FADE_TREND")
        self.assertEqual(payload["action_side"], "LONG")

    def test_choose_action_selects_best_similar_memory(self):
        fields = ("regime", "trend_alignment")
        row = _row(200, "TREND", "LONG", 0.0, 0.0)
        memory = [example_from_row(_row(0, "TREND", "LONG", 0.01, -0.01), fields)]
        action, dbg = choose_action(
            row,
            [m for m in memory if m is not None],
            fields,
            top_k=4,
            min_similarity=1.0,
            min_neighbors=1,
            min_mean_return=0.001,
            mae_penalty=0.0,
            recency_halflife_bars=0.0,
        )
        self.assertEqual(action, "TREND")
        self.assertEqual(dbg["neighbors_considered"], 1)

    def test_cli_writes_report_and_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            out = root / "report.json"
            preds = root / "preds"
            train.write_text(json.dumps(_row(0, "TREND", "LONG", 0.01, -0.01)) + "\n")
            val.write_text(json.dumps(_row(120, "TREND", "LONG", 0.01, -0.01)) + "\n")
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
                    "--similarity-fields",
                    "regime,trend_alignment",
                    "--hold-bars",
                    "100",
                    "--min-neighbors",
                    "1",
                    "--min-similarity",
                    "1.0",
                ]
                main()
            finally:
                sys.argv = old
            report = json.loads(out.read_text())
            self.assertIn("leakage_guard", report)
            self.assertTrue((preds / "val_predictions.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
