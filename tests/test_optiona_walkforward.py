import unittest
from unittest.mock import patch

from training.optiona_walkforward import (
    compute_policy_score,
    parse_folds_spec,
    run_optiona_walkforward,
)


class TestOptionAWalkforward(unittest.TestCase):
    def test_parse_folds_spec(self):
        folds = parse_folds_spec("2025-03-01:2025-03-07:2025-03-08:2025-03-14")
        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0]["val_start"], "2025-03-01")
        self.assertEqual(folds[0]["test_end"], "2025-03-14")

    def test_compute_policy_score_penalizes_hold_all(self):
        active = {
            "cumulative_return_pct": 1.0,
            "min_sharpe": 0.2,
            "max_drawdown_pct": 2.0,
            "action_ratio": {"hold": 0.2},
        }
        hold_all = {
            "cumulative_return_pct": 1.0,
            "min_sharpe": 0.2,
            "max_drawdown_pct": 2.0,
            "action_ratio": {"hold": 1.0},
        }
        self.assertGreater(compute_policy_score(active), compute_policy_score(hold_all))

    def test_run_optiona_walkforward_selects_highest_validation(self):
        def _fake_backtest(**kwargs):
            mode = kwargs.get("decision_mode", "policy")
            start = kwargs.get("start_date")
            if start == "2025-03-01":  # validation phase
                if mode == "policy":
                    return {
                        "cumulative_return_pct": 1.0,
                        "min_sharpe": 0.5,
                        "max_drawdown_pct": 1.0,
                        "action_ratio": {"hold": 0.0},
                    }
                return {
                    "cumulative_return_pct": 0.1,
                    "min_sharpe": 0.0,
                    "max_drawdown_pct": 1.0,
                    "action_ratio": {"hold": 0.0},
                }
            # test phase
            return {
                "cumulative_return_pct": 0.2,
                "min_sharpe": 0.1,
                "max_drawdown_pct": 1.0,
                "action_ratio": {"hold": 0.0},
            }

        folds = [
            {
                "name": "f1",
                "val_start": "2025-03-01",
                "val_end": "2025-03-07",
                "test_start": "2025-03-08",
                "test_end": "2025-03-14",
            }
        ]
        with patch("training.optiona_walkforward.run_backtest", side_effect=_fake_backtest):
            out = run_optiona_walkforward(model_path="/tmp/m.zip", folds=folds)
        self.assertEqual(out["num_folds"], 1)
        sel = out["folds"][0]["selected_policy"]["name"]
        self.assertEqual(sel, "policy_det")


if __name__ == "__main__":
    unittest.main()
