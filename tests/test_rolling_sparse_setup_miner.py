import unittest

import pandas as pd

from training.rolling_sparse_setup_miner import _feature_columns, SparseSetupCfg, run


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


if __name__ == "__main__":
    unittest.main()
