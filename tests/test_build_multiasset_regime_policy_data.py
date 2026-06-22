import json
import unittest

from training.build_multiasset_regime_preference_data import _response
from training.build_multiasset_regime_sft_data import _prompt, _target, policy_response


class TestBuildMultiassetRegimePolicyData(unittest.TestCase):
    def test_sft_target_is_single_policy_json(self):
        target = json.loads(_target({"selected": "utility1_pos", "scores": {"utility1_pos": 4.5, "cash": 1.0}}))
        self.assertEqual(target["policy"], "utility1_pos")
        self.assertTrue(target["allow_trade"])
        self.assertEqual(target["evidence_strength"], "high")
        self.assertNotIn("analyzer", target)
        self.assertNotIn("trader", target)

    def test_preference_response_uses_same_single_policy_schema(self):
        target = json.loads(_response("cash", {"utility3_pos": 2.0, "cash": 0.5}))
        self.assertEqual(target["policy"], "cash")
        self.assertFalse(target["allow_trade"])
        self.assertLess(target["score_margin"], 0)
        self.assertNotIn("analyzer", target)
        self.assertNotIn("trader", target)

    def test_prompt_does_not_request_two_stage_roles(self):
        prompt = _prompt(
            {"month": "2026-01-01"},
            [{"month": "2025-12-01", "selected": "cash", "sim": {"ret_pct": 0.0, "strict_mdd_pct": 0.0}, "scores": {"cash": 1.0}}],
            {"market_return": "neutral"},
        )
        lower = prompt.lower()
        self.assertIn("single compact rllm monthly policy", lower)
        self.assertNotIn("analyzer/trader", lower)
        self.assertIn("policy, allow_trade", prompt)

    def test_policy_response_penalizes_non_best_margin(self):
        target = json.loads(policy_response("utility1_inv", {"cash": 3.0, "utility1_inv": 1.0}, reason_code="x"))
        self.assertEqual(target["policy"], "utility1_inv")
        self.assertEqual(target["reason_code"], "x")
        self.assertEqual(target["score_margin"], -2.0)


if __name__ == "__main__":
    unittest.main()
