import json
import tempfile
import unittest
from pathlib import Path

from training.sparse_setup_contextual_ranker import ContextualRankerCfg, _fit_ridge, _predict_action, load_records


class TestSparseSetupContextualRanker(unittest.TestCase):
    def test_load_records_orders_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "r.jsonl"
            path.write_text('\n'.join([
                json.dumps({"date": "2024-01-02", "signal_pos": 2, "key": "b"}),
                json.dumps({"date": "2024-01-01", "signal_pos": 1, "key": "a"}),
            ]))
            rows = load_records(path)
        self.assertEqual([r["key"] for r in rows], ["a", "b"])

    def test_ranker_prefers_action_token_seen_with_positive_reward(self):
        rows = [
            {
                "state_tokens": ["regime=trend"],
                "actions": {
                    "LONG_2_base": {"side": "LONG", "hold_bars": 2, "risk_profile": "base", "net_return": 0.01, "mae": 0.001, "utility": 0.01},
                    "LONG_2_sl3": {"side": "LONG", "hold_bars": 2, "risk_profile": "sl3", "net_return": -0.01, "mae": 0.001, "utility": -0.01},
                },
            }
            for _ in range(5)
        ]
        cfg = ContextualRankerCfg("r.jsonl", "s.json", "o.json", l2=0.1, min_pred_utility=-1, min_pred_net=-1)
        vocab, w, fit = _fit_ridge(rows, cfg)
        action = _predict_action(rows[0], vocab, w, cfg)
        self.assertEqual(action["action_key"], "LONG_2_base")
        self.assertEqual(fit["samples"], 10)


if __name__ == "__main__":
    unittest.main()
