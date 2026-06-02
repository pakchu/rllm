import unittest

from training.eval_calibrated_policy_model import _agreement, _metrics_from_actions, parse_policy_json


class TestEvalCalibratedPolicyModel(unittest.TestCase):
    def test_parse_policy_json_extracts_first_json_and_validates_trade(self):
        parsed = parse_policy_json('x {"gate":"TRADE","side":"LONG","hold_bars":96} y', allowed_holds=(48, 96))
        self.assertEqual(parsed["gate"], "TRADE")
        self.assertEqual(parsed["hold_bars"], 96)
        invalid = parse_policy_json('{"gate":"TRADE","side":"LONG","hold_bars":999}', allowed_holds=(48, 96))
        self.assertEqual(invalid["gate"], "NO_TRADE")

    def test_metrics_from_actions_skips_overlap_and_uses_mae_mdd(self):
        records = [
            {"date": "d0", "signal_pos": 0, "key": "k", "actions": {"LONG_3": {"side": "LONG", "hold_bars": 3, "net_return": 0.1, "mae": 0.01, "utility": 0.09}}},
            {"date": "d1", "signal_pos": 1, "key": "k", "actions": {"LONG_3": {"side": "LONG", "hold_bars": 3, "net_return": 0.1, "mae": 0.01, "utility": 0.09}}},
            {"date": "d4", "signal_pos": 4, "key": "k", "actions": {"LONG_3": {"side": "LONG", "hold_bars": 3, "net_return": -0.01, "mae": 0.2, "utility": -0.21}}},
        ]
        actions = [{"gate": "TRADE", "side": "LONG", "hold_bars": 3}] * 3
        metrics = _metrics_from_actions(records, actions)
        self.assertEqual(metrics["trades"], 2)
        self.assertEqual(metrics["model_overlap_skips"], 1)
        self.assertGreater(metrics["strict_mdd_proxy"], 0.19)

    def test_agreement_counts_gate_and_exact(self):
        records = [{"x": 1}, {"x": 2}]
        pred = [{"gate": "TRADE", "side": "LONG", "hold_bars": 48}, {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}]
        target = [{"gate": "TRADE", "side": "LONG", "hold_bars": 48}, {"gate": "TRADE", "side": "LONG", "hold_bars": 96}]
        out = _agreement(records, pred, target)
        self.assertEqual(out["gate_accuracy"], 0.5)
        self.assertEqual(out["exact_action_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
