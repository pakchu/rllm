import unittest

from training.eval_text_json_key import _candidate_json


class TestEvalTextJsonKeyCandidateJson(unittest.TestCase):
    def test_side_map_candidate_json_uses_lowercase_value(self):
        self.assertEqual(_candidate_json("side_map", "NORMAL"), '{"side_map": "normal"}')


if __name__ == "__main__":
    unittest.main()
