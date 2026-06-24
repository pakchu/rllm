import unittest

from training.event_candidate_drift_audit import _drift_summary, _stats


class TestEventCandidateDriftAudit(unittest.TestCase):
    def test_stats_reports_positive_fraction(self):
        out = _stats([-1, 0, 2])
        self.assertEqual(out["n"], 3)
        self.assertAlmostEqual(out["positive_frac"], 1 / 3)

    def test_drift_summary_compares_2026_to_prior_mean(self):
        rows = [
            {"year": "2024", "field": "family", "value": "a", "mean": 1.0},
            {"year": "2025", "field": "family", "value": "a", "mean": 3.0},
            {"year": "2026", "field": "family", "value": "a", "mean": -1.0},
        ]
        out = _drift_summary(rows, key_fields=("field", "value"), top_n=1)
        self.assertEqual(out[0]["key"], {"field": "family", "value": "a"})
        self.assertAlmostEqual(out[0]["prior_mean"], 2.0)
        self.assertAlmostEqual(out[0]["delta_2026_vs_prior"], -3.0)


if __name__ == "__main__":
    unittest.main()
