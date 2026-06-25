import unittest

import numpy as np
import pandas as pd

from training.sparse_setup_failure_veto_tte import FailureVetoTTECfg, _candidate_vetoes, _select_events, _veto_mask


class TestSparseSetupFailureVetoTTE(unittest.TestCase):
    def test_veto_mask_blocks_flagged_rows(self):
        features = pd.DataFrame({"fr__bad": [0.0, 1.0, 0.0]})
        rows = [{"signal_pos": 0}, {"signal_pos": 1}, {"signal_pos": 2}]
        self.assertEqual(_veto_mask(rows, features, ("fr__bad",)).tolist(), [True, False, True])

    def test_select_events_keeps_best_unvetoed_per_signal(self):
        features = pd.DataFrame({"fr__bad": [0.0, 1.0, 0.0]})
        rows = [{"signal_pos": 0, "id": "a"}, {"signal_pos": 1, "id": "b"}, {"signal_pos": 2, "id": "c"}]
        selected = _select_events(rows, np.asarray([0.2, 0.9, 0.1]), threshold=0.0, features=features, veto=("fr__bad",))
        self.assertEqual([r["id"] for r in selected], ["a", "c"])

    def test_candidate_vetoes_filters_by_rate(self):
        features = pd.DataFrame({"fr__rare": [0.0, 0.0, 1.0], "fr__always": [1.0, 1.0, 1.0]})
        rows = [{"signal_pos": 0}, {"signal_pos": 1}, {"signal_pos": 2}]
        vetoes = _candidate_vetoes(features, rows, FailureVetoTTECfg("s", "m", "o", "[]", "[]", "[]", min_veto_feature_rate=0.2, max_veto_feature_rate=0.8))
        self.assertIn(("fr__rare",), vetoes)
        self.assertNotIn(("fr__always",), vetoes)


if __name__ == "__main__":
    unittest.main()
