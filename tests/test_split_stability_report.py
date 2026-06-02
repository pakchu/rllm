import unittest

from training.split_stability_report import _passes_gate, _summarize_candidate


class TestSplitStabilityReport(unittest.TestCase):
    def test_summarize_candidate_computes_gap(self):
        s = _summarize_candidate({"test_metrics": {"trades": 10, "cagr_to_mdd_proxy": 5}, "eval_metrics": {"trades": 10, "cagr_to_mdd_proxy": 2}}, 1)
        self.assertEqual(s["generalization_gap"]["ratio_eval_minus_test"], -3)

    def test_passes_gate_requires_eval_ratio(self):
        summary = {"eval": {"trades": 30, "ratio": 4, "strict_mdd": 0.1}, "generalization_gap": {"ratio_eval_minus_test": -1}}
        self.assertTrue(_passes_gate(summary, min_eval_trades=30, min_eval_ratio=3, max_eval_mdd=0.15, max_ratio_gap=3))
