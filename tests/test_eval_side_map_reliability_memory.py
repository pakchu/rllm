import unittest

from training.eval_side_map_reliability_memory import _history_labels, _majority


class TestSideMapReliabilityMemoryEval(unittest.TestCase):
    def test_history_labels(self):
        row = {"prompt": "- month=2025-01 label=inverse pass_cagr_bucket=negative\n- month=2025-02 label=normal"}
        self.assertEqual(_history_labels(row), ["inverse", "normal"])

    def test_majority(self):
        self.assertEqual(_majority(["normal", "inverse", "normal"]), "normal")
        self.assertEqual(_majority([], "unreliable"), "unreliable")


if __name__ == "__main__":
    unittest.main()
