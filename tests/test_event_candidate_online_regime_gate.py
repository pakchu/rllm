import unittest

from training.event_candidate_online_regime_gate import OnlineRegimeGateCfg, _select_rule


class TestEventCandidateOnlineRegimeGate(unittest.TestCase):
    def test_select_rule_uses_prior_improvement(self):
        prior = [
            {"pretest_range_pos": 0.9, "test_ratio": -2.0, "is_bad": True},
            {"pretest_range_pos": 0.2, "test_ratio": 3.0, "is_bad": False},
            {"pretest_range_pos": 0.3, "test_ratio": 2.0, "is_bad": False},
        ]
        cfg = OnlineRegimeGateCfg(fold_regime_audit="a", predictions_root="p", market_csv="m", output="o", min_prior_folds=3, candidate_features="pretest_range_pos", candidate_thresholds="0.8")
        selected = _select_rule(prior, cfg)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["rule"]["threshold"], 0.8)
        self.assertGreater(selected["improvement"], 0.0)


if __name__ == "__main__":
    unittest.main()
