import unittest

import pandas as pd

import numpy as np

from training.rolling_sparse_setup_miner import _build_predicate_cache, _feature_columns, SparseSetupCfg, run


class TestRollingSparseSetupMiner(unittest.TestCase):
    def test_feature_columns_include_fx_component_features(self):
        features = pd.DataFrame({
            "mkt__fx_eurusd_zscore": [0.0, 1.0, 2.0],
            "mkt__btckrw_momentum": [0.0, 0.1, 0.2],
            "mkt__external_any_available": [1.0, 1.0, 1.0],
            "mkt__trend_12": [0.0, 0.1, 0.2],
        })

        cols = _feature_columns(features)

        self.assertIn("mkt__fx_eurusd_zscore", cols)
        self.assertIn("mkt__btckrw_momentum", cols)
        self.assertIn("mkt__trend_12", cols)
        self.assertNotIn("mkt__external_any_available", cols)

    def test_predicate_cache_reuses_fold_threshold_masks(self):
        values = np.asarray([0.0, 1.0, 2.0, 3.0])
        finite_y = np.asarray([True, True, True, True])
        fold_meta = [{"train": np.asarray([True, True, False, False]), "eval": np.asarray([False, False, True, True])}]

        cache = _build_predicate_cache(
            cols=["a"],
            X={"a": values},
            fold_meta=fold_meta,
            finite_y=finite_y,
            q=0.5,
            min_train_rows=1,
        )

        low = cache[("a", "low", 0)]
        high = cache[("a", "high", 0)]
        self.assertEqual(low["threshold"], 0.5)
        self.assertEqual(high["threshold"], 0.5)
        self.assertEqual(low["mask"].tolist(), [True, False, False, False])
        self.assertEqual(high["mask"].tolist(), [False, True, True, True])

    def test_feature_columns_include_price_action_extreme_features(self):
        features = pd.DataFrame({
            "pa__pa_ext_144_to_max_high_pct": [-0.1, -0.2, -0.3],
            "pa__pa_ext_144_range_pos": [0.1, 0.2, 0.3],
            "mkt__external_any_available": [1.0, 1.0, 1.0],
        })

        cols = _feature_columns(features)

        self.assertIn("pa__pa_ext_144_to_max_high_pct", cols)
        self.assertIn("pa__pa_ext_144_range_pos", cols)

    def test_score_event_folds_respects_train_split_min_positive_folds(self):
        from training.rolling_sparse_setup_miner import _score_event_folds

        cfg = SparseSetupCfg(input_csv="i", output="o", min_positive_folds=3, min_fold_events=1)
        folds = [
            {"n": 2, "mean_pct": 1.0, "t_stat": 1.0},
            {"n": 2, "mean_pct": 1.0, "t_stat": 1.0},
            {"n": 2, "mean_pct": 1.0, "t_stat": 1.0},
            {"n": 2, "mean_pct": -0.5, "t_stat": -1.0},
        ]

        self.assertGreater(_score_event_folds(folds, cfg), 0.0)


if __name__ == "__main__":
    unittest.main()
