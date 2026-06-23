import unittest

from training.build_side_map_reliability_sft import _ret_bucket, _score_bucket, _split


class TestBuildSideMapReliabilitySft(unittest.TestCase):
    def test_score_bucket(self):
        self.assertEqual(_score_bucket(None), "unknown")
        self.assertEqual(_score_bucket(0.5), "positive")
        self.assertEqual(_score_bucket(-500.1), "severe_decay")
        self.assertEqual(_score_bucket(0.1), "weak_positive")

    def test_split(self):
        self.assertEqual(_split("2024-12", "2024-12", "2025-12"), "train")
        self.assertEqual(_split("2025-01", "2024-12", "2025-12"), "val")
        self.assertEqual(_split("2026-01", "2024-12", "2025-12"), "eval")

    def test_ret_bucket(self):
        self.assertEqual(_ret_bucket(120), "very_positive")
        self.assertEqual(_ret_bucket(-60), "very_negative")


if __name__ == "__main__":
    unittest.main()
