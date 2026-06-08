import unittest

from training.eval_single_policy import parse_policy_json, policy_to_action


class TestEvalSinglePolicy(unittest.TestCase):
    def test_parse_policy_json_coerces_no_trade_exit(self):
        p = parse_policy_json('{"action":"NO_TRADE","exit_profile":"TRAIL","regime":"TREND_UP","edge_quality":"NONE","risk":"LOW","confidence":"LOW"}')
        self.assertEqual(p["action"], "NO_TRADE")
        self.assertEqual(p["exit_profile"], "AVOID")
        self.assertEqual(policy_to_action(p), {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0})

    def test_parse_policy_json_maps_trade_exit_to_strict_action(self):
        p = parse_policy_json('prefix {"action":"SHORT","exit_profile":"FAST","regime":"TREND_DOWN","edge_quality":"MODERATE","risk":"MID","confidence":"MID"}')
        self.assertEqual(policy_to_action(p), {"gate": "TRADE", "side": "SHORT", "hold_bars": 72})


if __name__ == "__main__":
    unittest.main()
