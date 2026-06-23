import unittest

from training.augment_context_with_validation_reliability import _inject_prompt_tokens, reliability_bucket


class TestValidationReliabilityAugment(unittest.TestCase):
    def test_reliability_bucket_edges(self):
        self.assertEqual(reliability_bucket(None)[0], "unknown_pre_roll")
        self.assertEqual(reliability_bucket(0.5)[0], "reliable_normal")
        self.assertEqual(reliability_bucket(-500.01)[0], "inverse_candidate")
        self.assertEqual(reliability_bucket(-500.0)[0], "weak_or_decaying")

    def test_inject_prompt_before_policy_intent(self):
        prompt = "header\ncausal_state_tokens:\n- a: b\nPolicy intent: wait"
        out = _inject_prompt_tokens(prompt, {"side_map_reliability": "inverse_candidate"})
        self.assertIn("- side_map_reliability: inverse_candidate\nPolicy intent", out)


if __name__ == "__main__":
    unittest.main()
