import unittest

from training.train_text_sft import _target_counter


class TestTrainTextSftSideMapCounts(unittest.TestCase):
    def test_target_counter_side_map(self):
        rows = [{"target": '{"side_map":"normal"}'}, {"target": '{"side_map":"inverse"}'}]
        counts = _target_counter(rows)
        self.assertEqual(counts["side_map=normal"], 1)
        self.assertEqual(counts["side_map=inverse"], 1)


if __name__ == "__main__":
    unittest.main()
