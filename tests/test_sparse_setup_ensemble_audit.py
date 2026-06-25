import unittest

import numpy as np
import pandas as pd

from training.sparse_setup_ensemble_audit import EnsembleCfg, _candidate_events


class TestSparseSetupEnsembleAudit(unittest.TestCase):
    def _market(self, n=160):
        return pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=n, freq="5min"),
                "open": np.linspace(100.0, 120.0, n),
                "high": np.linspace(100.5, 120.5, n),
                "low": np.linspace(99.5, 119.5, n),
            }
        )

    def test_candidate_events_respects_stored_zero_sample_fold(self):
        market = self._market()
        dates = pd.to_datetime(market["date"])
        features = pd.DataFrame({"fa": np.ones(len(market)), "fb": np.ones(len(market))})
        cand = {
            "horizon": 3,
            "quantile": 0.5,
            "features": [{"name": "fa", "side": "high"}, {"name": "fb", "side": "high"}],
            "strict_folds": [
                {
                    "fold": "f1",
                    "side": "LONG",
                    "thresholds": {"fa": {"threshold": 0.0}, "fb": {"threshold": 0.0}},
                    "result": {"sim": {"samples": 0, "trade_entries": 0}},
                }
            ],
        }
        report = {"folds": [{"name": "f1", "eval_start": "2020-01-01 10:00:00", "eval_end": "2020-01-01 12:00:00"}]}

        events = _candidate_events(cand=cand, report=report, dates=dates, features=features, market=market, cfg=EnsembleCfg("s", "m", "o"))

        self.assertEqual(events, [])

    def test_candidate_events_replays_stored_threshold_and_side(self):
        market = self._market()
        dates = pd.to_datetime(market["date"])
        values = np.zeros(len(market))
        values[110:115] = 10.0
        values[130:135] = 10.0
        features = pd.DataFrame({"fa": values, "fb": values})
        cand = {
            "horizon": 3,
            "quantile": 0.5,
            "features": [{"name": "fa", "side": "high"}, {"name": "fb", "side": "high"}],
            "strict_folds": [
                {
                    "fold": "f1",
                    "side": "SHORT",
                    "thresholds": {"fa": {"threshold": 9.0}, "fb": {"threshold": 9.0}},
                    "result": {"sim": {"samples": 5, "trade_entries": 5}},
                }
            ],
        }
        report = {"folds": [{"name": "f1", "eval_start": "2020-01-01 10:00:00", "eval_end": "2020-01-01 12:00:00"}]}

        events = _candidate_events(cand=cand, report=report, dates=dates, features=features, market=market, cfg=EnsembleCfg("s", "m", "o"))

        self.assertTrue(events)
        self.assertTrue(all(e["side"] == -1 for e in events))
        self.assertTrue(all(e["thresholds"]["fa"] == 9.0 for e in events))


if __name__ == "__main__":
    unittest.main()
