import json
import unittest

from training.build_event_side_rationale_preference import candidate_response


class TestEvalEventSideRationalePreferenceContract(unittest.TestCase):
    def test_candidate_response_shape_contains_side_and_rationale(self):
        row = {"score_tokens": {"score_side_gap": "high"}, "state_tokens": {"trend_alignment": "aligned_down"}}
        obj = json.loads(candidate_response(row, "normal"))
        self.assertEqual(obj["side_pair"], "normal")
        self.assertIn("rationale_class", obj)
        self.assertIn("causal_evidence", obj)


if __name__ == "__main__":
    unittest.main()
