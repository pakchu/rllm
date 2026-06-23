import json
import unittest

from training.eval_dxy_kimchi_policy import _candidate_policy_jsons, parse_dxy_kimchi_policy, policy_to_prediction


class TestEvalDxyKimchiPolicy(unittest.TestCase):
    def test_parse_normalizes_inactive_policy(self):
        p = parse_dxy_kimchi_policy('{"activate":false,"action":"LONG","exit_profile":"FAST"}')
        self.assertFalse(p["activate"])
        self.assertEqual(p["action"], "NO_TRADE")
        self.assertEqual(p["exit_profile"], "AVOID")

    def test_candidate_policy_jsons_cover_abstain_long_short(self):
        candidates = [p for p, _ in _candidate_policy_jsons()]
        self.assertEqual([c["action"] for c in candidates], ["NO_TRADE", "LONG", "SHORT"])
        self.assertTrue(all(c["reason_code"] != "model_activate" for c in candidates))

    def test_policy_to_prediction_maps_active_side(self):
        pred = policy_to_prediction({"activate": True, "action": "SHORT", "exit_profile": "FAST"}, horizon=144)
        self.assertEqual(pred, {"gate": "TRADE", "side": "SHORT", "hold_bars": 144})
        no = policy_to_prediction({"activate": False, "action": "NO_TRADE"}, horizon=144)
        self.assertEqual(no["gate"], "NO_TRADE")


if __name__ == "__main__":
    unittest.main()
