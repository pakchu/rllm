import unittest

import pandas as pd

from training.sparse_setup_action_policy import (
    SparseActionPolicyCfg,
    _action_outcomes,
    _actions_for_records,
    fit_action_rules,
    walkforward_action_eval,
)


class TestSparseSetupActionPolicy(unittest.TestCase):
    def test_action_outcomes_include_source_and_opposite_holds(self):
        market = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=8, freq="5min"),
                "open": [100, 101, 102, 103, 104, 105, 106, 107],
                "high": [101, 102, 103, 104, 105, 106, 107, 108],
                "low": [99, 100, 101, 102, 103, 104, 105, 106],
                "close": [101, 102, 103, 104, 105, 106, 107, 108],
                "volume": [1] * 8,
            }
        )
        cfg = SparseActionPolicyCfg("sparse.json", "m.csv", "o.jsonl", "r.json", hold_candidates=(2,))
        actions = _action_outcomes(market, 0, "LONG", cfg)
        self.assertIn("LONG_2", actions)
        self.assertIn("SHORT_2", actions)
        self.assertGreater(actions["LONG_2"]["net_return"], actions["SHORT_2"]["net_return"])

    def test_fit_action_rules_uses_history_stats(self):
        rows = []
        for i in range(10):
            rows.append(
                {
                    "key": "k",
                    "actions": {
                        "LONG_2": {"side": "LONG", "hold_bars": 2, "net_return": 0.01, "mae": 0.005, "utility": 0.005},
                        "SHORT_2": {"side": "SHORT", "hold_bars": 2, "net_return": -0.01, "mae": 0.02, "utility": -0.03},
                    },
                }
            )
        cfg = SparseActionPolicyCfg("sparse.json", "m.csv", "o.jsonl", "r.json", hold_candidates=(2,), min_action_samples=5)
        rules = fit_action_rules(rows, cfg)
        self.assertEqual(rules["k"]["action"]["side"], "LONG")
        self.assertEqual(rules["k"]["action"]["hold_bars"], 2)

    def test_walkforward_action_eval_does_not_fit_on_current_fold(self):
        sparse = {"folds": [{"name": "A", "eval_start": "2024-01-01"}, {"name": "B", "eval_start": "2024-07-01"}]}
        records = [
            {
                "date": "2024-01-01",
                "signal_pos": 0,
                "fold": "A",
                "key": "k",
                "source_action": {"side": "LONG", "hold_bars": 2},
                "actions": {"LONG_2": {"side": "LONG", "hold_bars": 2, "net_return": 0.01, "mae": 0.001, "utility": 0.009}},
            },
            {
                "date": "2024-07-01",
                "signal_pos": 10,
                "fold": "B",
                "key": "k",
                "source_action": {"side": "LONG", "hold_bars": 2},
                "actions": {"LONG_2": {"side": "LONG", "hold_bars": 2, "net_return": 0.01, "mae": 0.001, "utility": 0.009}},
            },
        ]
        cfg = SparseActionPolicyCfg("sparse.json", "m.csv", "o.jsonl", "r.json", hold_candidates=(2,), min_action_samples=1)
        report = walkforward_action_eval(records, sparse, cfg)
        self.assertEqual(report["folds"][0]["rules_count"], 0)
        self.assertEqual(report["folds"][0]["action_reasons"], {"COLD_START_SOURCE_ACTION": 1})
        self.assertEqual(report["folds"][1]["rules_count"], 1)
        self.assertEqual(report["folds"][1]["action_reasons"], {"HISTORY_ACTION_RULE": 1})

    def test_actions_for_records_no_history_can_no_trade(self):
        cfg = SparseActionPolicyCfg("sparse.json", "m.csv", "o.jsonl", "r.json", hold_candidates=(2,), cold_start_source_action=False, fallback_source_action="cold_start")
        row = {"key": "k", "source_action": {"side": "LONG", "hold_bars": 2}, "actions": {"LONG_2": {}}}
        self.assertEqual(_actions_for_records([row], {}, cfg, allow_cold_start=True)[0]["gate"], "NO_TRADE")

    def test_fit_action_rules_can_pool_by_source_side(self):
        rows = []
        for key in ("k1", "k2"):
            for _ in range(3):
                rows.append(
                    {
                        "key": key,
                        "source_action": {"side": "LONG", "hold_bars": 2},
                        "actions": {"LONG_2": {"side": "LONG", "hold_bars": 2, "net_return": 0.01, "mae": 0.001, "utility": 0.009}},
                    }
                )
        candidate_cfg = SparseActionPolicyCfg("sparse.json", "m.csv", "o.jsonl", "r.json", hold_candidates=(2,), min_action_samples=5)
        pooled_cfg = SparseActionPolicyCfg("sparse.json", "m.csv", "o.jsonl", "r.json", hold_candidates=(2,), min_action_samples=5, rule_key_mode="source_side")
        self.assertEqual(fit_action_rules(rows, candidate_cfg), {})
        self.assertIn("source_side=LONG", fit_action_rules(rows, pooled_cfg))

    def test_actions_for_records_can_fallback_to_source_after_cold_start(self):
        cfg = SparseActionPolicyCfg("sparse.json", "m.csv", "o.jsonl", "r.json", hold_candidates=(2,), fallback_source_action="always_when_no_rule")
        row = {"key": "k", "source_action": {"side": "LONG", "hold_bars": 2}, "actions": {"LONG_2": {}}}
        action = _actions_for_records([row], {}, cfg, allow_cold_start=False)[0]
        self.assertEqual(action["gate"], "TRADE")
        self.assertEqual(action["reason"], "COLD_START_SOURCE_ACTION")


if __name__ == "__main__":
    unittest.main()
