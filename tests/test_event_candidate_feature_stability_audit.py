import unittest

from training.event_candidate_feature_stability_audit import CandidateFeatureStabilityCfg, _metrics


class TestCandidateFeatureStabilityAudit(unittest.TestCase):
    def test_metrics_reports_positive_spread(self):
        rows = [{"feature_snapshot": {"x": float(i)}, "reward": {"rank_utility": float(i)}} for i in range(20)]
        m = _metrics(rows, "x", CandidateFeatureStabilityCfg(input_jsonl="x", output="o", min_rows=5))
        self.assertGreater(m["spearman_ic"], 0.99)
        self.assertGreater(m["q_high_minus_low"], 0.0)


if __name__ == "__main__":
    unittest.main()
