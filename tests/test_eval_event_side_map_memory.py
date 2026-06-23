import unittest

from training.eval_event_side_map_memory import _majority, _signature


class TestEvalEventSideMapMemory(unittest.TestCase):
    def test_majority_prefers_unreliable_tie(self):
        self.assertEqual(_majority(["normal", "unreliable"]), "unreliable")

    def test_signature(self):
        row = {"generated_side": "LONG", "score_tokens": {"score_side_gap": "high"}, "state_tokens": {"trend_alignment": "up"}}
        self.assertIn("side=LONG", _signature(row, "token_signature"))
        self.assertIn("score_side_gap=high", _signature(row, "score_only"))


if __name__ == "__main__":
    unittest.main()
