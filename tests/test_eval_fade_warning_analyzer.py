import json
import tempfile
import unittest
from pathlib import Path

from training.eval_fade_warning_analyzer import evaluate_fade_warning_analyzer, parse_fade_warning_json


def _row(fade="FADE_STRONG", skip="TRADEABLE_FADE"):
    target = {"trend_side": "LONG", "fade_warning": fade, "skip_reason": skip, "trend_continuation_quality": "NO_CONTINUATION"}
    return {"task": "fade_warning_analyzer", "prompt": "Past-only analyzer summary: {}", "target": json.dumps(target)}


class TestEvalFadeWarningAnalyzer(unittest.TestCase):
    def test_parser_normalizes_bad_values(self):
        parsed = parse_fade_warning_json('{"fade_warning":"bad","skip_reason":"bad"}')
        self.assertEqual(parsed["fade_warning"], "NO_FADE_WARNING")
        self.assertEqual(parsed["skip_reason"], "NO_EDGE")

    def test_target_echo_eval_reports_exact_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "eval.jsonl"
            out = root / "report.json"
            preds = root / "preds.jsonl"
            rows = [_row("FADE_STRONG", "TRADEABLE_FADE"), _row("NO_FADE_WARNING", "NO_EDGE")]
            data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            report = evaluate_fade_warning_analyzer(eval_jsonl=str(data), output=str(out), prediction_output=str(preds), prediction_mode="target_echo")
            self.assertEqual(report["metrics"]["num_samples"], 2)
            self.assertEqual(report["metrics"]["exact_all_keys_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["exact_primary_keys_accuracy"], 1.0)
            self.assertTrue(report["leakage_guard"]["target_echo_is_oracle_only"])
            self.assertEqual(len(preds.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
