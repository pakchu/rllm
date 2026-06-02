import unittest

from training.export_drift_aware_policy_labels import _aggregate_outcomes, classify_drift_risk


class TestExportDriftAwarePolicyLabels(unittest.TestCase):
    def test_classify_drift_risk_high_on_bad_recent_mean(self):
        risk = classify_drift_risk(
            baseline={"mean_net_return": 0.01},
            recent={"samples": 6, "mean_net_return": -0.001, "win_rate": 0.5, "mean_mae": 0.01},
            min_recent_samples=6,
            high_min_mean_net=0.0,
            high_min_win_rate=0.45,
            high_max_mean_mae=0.02,
            medium_mean_drop=0.004,
        )
        self.assertEqual(risk, "HIGH")

    def test_classify_drift_risk_medium_on_baseline_drop(self):
        risk = classify_drift_risk(
            baseline={"mean_net_return": 0.01},
            recent={"samples": 6, "mean_net_return": 0.005, "win_rate": 0.6, "mean_mae": 0.01},
            min_recent_samples=6,
            high_min_mean_net=0.0,
            high_min_win_rate=0.45,
            high_max_mean_mae=0.02,
            medium_mean_drop=0.004,
        )
        self.assertEqual(risk, "MEDIUM")

    def test_aggregate_outcomes_empty_is_zero_sample(self):
        self.assertEqual(_aggregate_outcomes([])["samples"], 0)
