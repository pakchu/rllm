import unittest

import numpy as np
import pandas as pd

from training.sparse_setup_regime_gate_tte import RegimeGateTTECfg, _event_reward, _select_best_events, _selection_score, _standardize_apply, _standardize_fit


class TestSparseSetupRegimeGateTTE(unittest.TestCase):
    def test_event_reward_long_positive_path(self):
        market = pd.DataFrame({
            "open": [100.0, 100.0, 102.0, 104.0, 104.0],
            "high": [101.0, 101.0, 103.0, 105.0, 105.0],
            "low": [99.0, 99.0, 101.0, 103.0, 103.0],
        })
        ev = {"signal_pos": 0, "side": 1, "horizon": 2}
        reward = _event_reward(ev, market, RegimeGateTTECfg("s", "m", "o", "[]", "[]", "[]", fee_rate=0.0, slippage_rate=0.0))
        self.assertGreater(reward["net_return_pct"], 0.0)
        self.assertGreater(reward["utility"], 0.0)

    def test_select_best_events_applies_threshold_per_signal(self):
        rows = [
            {"signal_pos": 1, "candidate_index": 0},
            {"signal_pos": 1, "candidate_index": 1},
            {"signal_pos": 2, "candidate_index": 2},
        ]
        selected = _select_best_events(rows, np.asarray([0.1, 0.5, -0.2]), threshold=0.0)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["candidate_index"], 1)

    def test_standardize_handles_constant_columns(self):
        x = np.asarray([[1.0, 2.0], [1.0, 4.0]])
        z, mu, sd = _standardize_fit(x)
        applied = _standardize_apply(x, mu, sd)
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue(np.isfinite(applied).all())
        self.assertEqual(sd[0], 1.0)

    def test_selection_score_rejects_low_trade_cagr_explosion(self):
        sim = {"trade_entries": 1, "cagr_pct": 1e12, "strict_mdd_pct": 1.0, "cagr_to_strict_mdd": 1e12}
        self.assertLess(_selection_score(sim, min_trades=20), -998.0)


if __name__ == "__main__":
    unittest.main()
