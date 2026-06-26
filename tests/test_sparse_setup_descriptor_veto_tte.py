import unittest

import numpy as np
import pandas as pd

from training.sparse_setup_descriptor_veto_tte import DescriptorVetoTTECfg, _label_rows, _rule_fires, _select_events, _veto_mask


class TestSparseSetupDescriptorVetoTTE(unittest.TestCase):
    def test_label_rows_uses_train_outcome_thresholds(self):
        rows = [
            {"reward": {"utility": 0.5, "mae_pct": 1.0}},
            {"reward": {"utility": -0.5, "mae_pct": 2.0}},
            {"reward": {"utility": 0.1, "mae_pct": 1.0}},
        ]
        good, bad = _label_rows(rows, DescriptorVetoTTECfg("s", "m", "o", "[]", "[]", "[]"))
        self.assertEqual(len(good), 1)
        self.assertEqual(len(bad), 1)

    def test_rule_fires_respects_side_scope_and_direction(self):
        features = pd.DataFrame({"f": [0.1, 0.9]})
        desc = {"scope": "long", "feature": "f", "veto_rule": {"direction": "ge", "threshold": 0.5}}
        self.assertTrue(_rule_fires({"signal_pos": 1, "side": 1}, features, desc))
        self.assertFalse(_rule_fires({"signal_pos": 1, "side": -1}, features, desc))

    def test_veto_mask_and_select_events_block_descriptor_hits(self):
        features = pd.DataFrame({"f": [0.1, 0.9, 0.2]})
        desc = {"scope": "overall", "feature": "f", "veto_rule": {"direction": "ge", "threshold": 0.5}}
        rows = [{"signal_pos": 0, "id": "a"}, {"signal_pos": 1, "id": "b"}, {"signal_pos": 2, "id": "c"}]
        self.assertEqual(_veto_mask(rows, features, (desc,)).tolist(), [True, False, True])
        selected = _select_events(rows, np.asarray([0.3, 1.0, 0.2]), 0.0, features, (desc,))
        self.assertEqual([r["id"] for r in selected], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
