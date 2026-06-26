import unittest

import numpy as np

from training.sparse_setup_failure_cluster_miner import FailureClusterMinerCfg, _coverage_rule, _rows_by_cluster


class TestSparseSetupFailureClusterMiner(unittest.TestCase):
    def test_rows_by_cluster_splits_train_good_and_test_bad(self):
        events = [
            {"fold": "train", "reward": {"utility": 1.0, "mae_pct": 1.0}},
            {"fold": "test", "reward": {"utility": -1.0, "mae_pct": 2.0}},
            {"fold": "eval", "reward": {"utility": -1.0, "mae_pct": 2.0}},
        ]
        good, bad, counts = _rows_by_cluster(events, {"train"}, {"test"}, FailureClusterMinerCfg("s", "m", "o", "[]", "[]", "[]"))
        self.assertEqual(len(good), 1)
        self.assertEqual(len(bad), 1)
        self.assertEqual(counts["other"], 1)

    def test_coverage_rule_uses_high_threshold_when_bad_is_higher(self):
        row = {"effect_d_bad_minus_good": 1.0, "bad_p25": 8.0, "bad_p75": 10.0}
        rule = _coverage_rule(row, np.asarray([1.0, 2.0, 9.0]), np.asarray([8.0, 9.0, 10.0]))
        self.assertEqual(rule["direction"], "ge")
        self.assertGreater(rule["coverage_edge"], 0.0)


if __name__ == "__main__":
    unittest.main()
