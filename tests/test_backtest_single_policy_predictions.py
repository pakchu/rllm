import unittest

from training.backtest_single_policy_predictions import _policy_from_row


class TestBacktestSinglePolicyPredictions(unittest.TestCase):
    def test_policy_prediction_is_required_by_default(self):
        with self.assertRaisesRegex(ValueError, "requires policy_prediction"):
            _policy_from_row({"target": '{"action":"LONG","exit_profile":"FAST"}'})

    def test_target_echo_requires_explicit_oracle_flag(self):
        policy = _policy_from_row(
            {"target": '{"action":"SHORT","exit_profile":"FAST","regime":"TREND_DOWN","edge_quality":"MODERATE","risk":"MID","confidence":"MID"}'},
            allow_target_echo=True,
        )
        self.assertEqual(policy["action"], "SHORT")
        self.assertEqual(policy["exit_profile"], "FAST")

    def test_policy_prediction_takes_precedence_over_target(self):
        policy = _policy_from_row(
            {
                "target": '{"action":"SHORT","exit_profile":"FAST"}',
                "policy_prediction": {"action": "LONG", "exit_profile": "NORMAL", "regime": "TREND_UP", "edge_quality": "MODERATE", "risk": "LOW", "confidence": "MID"},
            }
        )
        self.assertEqual(policy["action"], "LONG")
        self.assertEqual(policy["exit_profile"], "NORMAL")


if __name__ == "__main__":
    unittest.main()
