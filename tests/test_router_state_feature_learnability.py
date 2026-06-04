import argparse
import json
import tempfile
import unittest
from pathlib import Path

from training.router_state_feature_learnability import KeyNaiveBayesModel, evaluate_key, run_report


def _row(label: str, regime: str):
    summary = {"regime": regime, "momentum": 1.0 if regime == "TREND" else -1.0}
    target = {"primary_route": label, "horizon_policy": "SHORT_STEP" if label != "SKIP" else "SKIP_STEP"}
    return {"prompt": "Past-only analyzer summary: " + json.dumps(summary), "target": json.dumps(target), "date": "2025-01-01"}


class TestRouterStateFeatureLearnability(unittest.TestCase):
    def test_key_model_learns_separable_rows(self):
        rows = [_row("TREND", "TREND"), _row("TREND", "TREND"), _row("SKIP", "RANGE"), _row("SKIP", "RANGE")]
        model = KeyNaiveBayesModel.fit(rows, key="primary_route")
        out = evaluate_key(model, rows)
        self.assertGreaterEqual(out["accuracy"], 0.75)
        self.assertIn("TREND", out["prediction_counts"])

    def test_run_report_writes_keywise_splits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            out = root / "out.json"
            rows = [_row("TREND", "TREND"), _row("SKIP", "RANGE")]
            text = "\n".join(json.dumps(r) for r in rows) + "\n"
            train.write_text(text)
            val.write_text(text)
            args = argparse.Namespace(train_jsonl=str(train), val_jsonl=str(val), oos_jsonl="", output=str(out), keys="primary_route,horizon_policy", alpha=1.0)
            report = run_report(args)
            self.assertIn("primary_route", report["per_key"])
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
