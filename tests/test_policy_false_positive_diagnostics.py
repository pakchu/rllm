import json
import tempfile
import unittest
from pathlib import Path

from training.policy_false_positive_diagnostics import _extract_prompt_features, run, PolicyFalsePositiveDiagnosticsCfg


class TestPolicyFalsePositiveDiagnostics(unittest.TestCase):
    def test_extract_prompt_features_reads_dash_colon_lines(self):
        prompt = "train_fitted_prior:\n- dxy_low_depth_bucket: medium\n- kimchi_prior_signal: SHORT\n"
        self.assertEqual(_extract_prompt_features(prompt)["dxy_low_depth_bucket"], "medium")

    def test_run_counts_false_positive_by_bucket(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            eval_path = root / "eval.jsonl"
            pred_path = root / "pred.jsonl"
            target = {"activate": False, "action": "NO_TRADE", "exit_profile": "AVOID", "confidence": "LOW", "reason_code": "x"}
            eval_path.write_text(json.dumps({"date": "2025-01-01", "signal_pos": 1, "prompt": "- kimchi_signal_strength_bucket: near", "target": json.dumps(target)}) + "\n")
            pred_path.write_text(json.dumps({"date": "2025-01-01", "signal_pos": 1, "policy_prediction": {"activate": True, "action": "SHORT", "exit_profile": "FAST"}}) + "\n")
            out = root / "out.json"
            report = run(PolicyFalsePositiveDiagnosticsCfg(eval_jsonl=str(eval_path), predictions_jsonl=str(pred_path), output=str(out), min_count=1))
            self.assertEqual(report["category_counts"]["false_positive_SHORT"], 1)
            self.assertEqual(report["feature_stats"]["kimchi_signal_strength_bucket"]["values"][0]["false_positive"], 1)


if __name__ == "__main__":
    unittest.main()
