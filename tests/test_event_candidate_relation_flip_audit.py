import unittest

from training.event_candidate_relation_flip_audit import _metrics


class TestEventCandidateRelationFlipAudit(unittest.TestCase):
    def test_metrics_reports_ic_and_quantile_spread(self):
        rows = [
            {"feature_snapshot": {"x": float(i)}, "reward": {"rank_utility": float(i)}}
            for i in range(20)
        ]
        m = _metrics(rows, "x", q=0.2, min_rows=5)
        self.assertGreater(m["spearman_ic"], 0.99)
        self.assertGreater(m["q_high_minus_low"], 0.0)

    def test_metrics_respects_min_rows(self):
        rows = [{"feature_snapshot": {"x": 1.0}, "reward": {"rank_utility": 1.0}}]
        self.assertEqual(_metrics(rows, "x", q=0.2, min_rows=5), {"n": 1})


if __name__ == "__main__":
    unittest.main()
