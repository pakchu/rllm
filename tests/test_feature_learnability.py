import json
import tempfile
import unittest
from pathlib import Path

from training.analyze_feature_learnability import _metrics, _parse_feature_list, run, parse_args
import numpy as np


class TestFeatureLearnability(unittest.TestCase):
    def test_metrics_balanced_recall(self):
        out = _metrics(np.array([1, 1, 0, 0]), np.array([0.8, 0.4, 0.3, 0.7]))
        self.assertEqual(out["samples"], 4)
        self.assertAlmostEqual(out["accuracy"], 0.5)
        self.assertAlmostEqual(out["balanced_recall"], 0.5)

    def test_feature_list_defaults(self):
        self.assertIn("trend_96", _parse_feature_list(None))
        self.assertEqual(_parse_feature_list("a,b"), ("a", "b"))


if __name__ == "__main__":
    unittest.main()
