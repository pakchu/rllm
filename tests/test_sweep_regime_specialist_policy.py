import unittest

from training.calibrated_regime_policy import CalibratedPolicyConfig
from training.sweep_regime_specialist_policy import _group_by_router, _precompute_router_action_rows, _precompute_router_stats, _router_key, evaluate_router_specialists, evaluate_router_specialists_precomputed


class TestSweepRegimeSpecialistPolicy(unittest.TestCase):
    def test_router_key_uses_summary_fields(self):
        row = {"summary": {"regime": "UP", "risk_state": "CALM"}}
        self.assertEqual(_router_key(row, ("regime", "risk_state")), "regime=UP|risk_state=CALM")

    def test_group_by_router_splits_records(self):
        rows = [
            {"summary": {"regime": "UP"}},
            {"summary": {"regime": "DOWN"}},
            {"summary": {"regime": "UP"}},
        ]
        groups = _group_by_router(rows, ("regime",))
        self.assertEqual(len(groups["regime=UP"]), 2)
        self.assertEqual(len(groups["regime=DOWN"]), 1)

    def test_evaluate_router_specialists_applies_router_book(self):
        records = [
            {
                "date": "2025-01-01",
                "signal_pos": 10,
                "summary": {"regime": "UP", "location": "PREMIUM"},
                "actions": {"LONG_48": {"side": "LONG", "hold_bars": 48, "net_return": 0.01, "mae": 0.001, "utility": 0.009}},
            }
        ]
        specialists = {
            "regime=UP": {
                "rules": {
                    "location=PREMIUM": {"action": {"side": "LONG", "hold_bars": 48}}
                }
            }
        }
        metrics = evaluate_router_specialists(records, specialists, router_fields=("regime",), specialist_key_fields=("location",))
        self.assertEqual(metrics["trades"], 1)
        self.assertGreater(metrics["compounded_return"], 0)

        precomputed = evaluate_router_specialists_precomputed(
            records_count=len(records),
            action_rows=_precompute_router_action_rows(records, router_fields=("regime",), specialist_key_fields=("location",)),
            specialists=specialists,
        )
        self.assertEqual(precomputed["trades"], metrics["trades"])
        self.assertEqual(precomputed["compounded_return"], metrics["compounded_return"])


    def test_precompute_router_stats_builds_each_router_once(self):
        rows = [
            {
                "date": "2023-01-01",
                "summary": {"regime": "UP", "location": "PREMIUM"},
                "actions": {"LONG_48": {"side": "LONG", "hold_bars": 48, "net_return": 0.01, "mae": 0.001, "utility": 0.009}},
            }
        ]
        stats = _precompute_router_stats(rows, router_fields=("regime",), specialist_key_fields=("location",))
        self.assertEqual(stats["regime=UP"]["train_records"], 1)
        self.assertIn("location=PREMIUM", stats["regime=UP"]["stats"])
