import unittest

from training.eval_text_json_key import parse_key_json


class TestEvalTextJsonKeySideMap(unittest.TestCase):
    def test_parse_side_map(self):
        self.assertEqual(parse_key_json('{"side_map":"normal"}', key="side_map"), "NORMAL")
        self.assertEqual(parse_key_json('{"side_map":"bad"}', key="side_map"), "UNRELIABLE")


if __name__ == "__main__":
    unittest.main()
