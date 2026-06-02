import tempfile
import unittest

from training.calibrated_regime_policy import CalibratedPolicyConfig
from training.sweep_yearly_stable_policy import _fit_from_stats, _load_or_build_records, _precompute_group_year_stats, _records_cache_path, _rule_signature
from training.yearly_stable_regime_policy import YearlyStableConfig


class TestSweepYearlyStablePolicy(unittest.TestCase):
    def test_fit_from_stats_requires_yearly_thresholds(self):
        stats = {
            "k": {
                "group_samples": 2,
                "actions": [
                    {
                        "overall": {
                            "samples": 2,
                            "side": "LONG",
                            "hold_bars": 48,
                            "mean_net_return": 0.01,
                            "mean_utility": 0.0,
                            "win_rate": 1.0,
                            "mean_mae": 0.01,
                        },
                        "yearly": {
                            "2023": {
                                "samples": 1,
                                "side": "LONG",
                                "hold_bars": 48,
                                "mean_net_return": 0.01,
                                "mean_utility": 0.0,
                                "win_rate": 1.0,
                                "mean_mae": 0.01,
                            },
                            "2024": {
                                "samples": 1,
                                "side": "LONG",
                                "hold_bars": 48,
                                "mean_net_return": -0.01,
                                "mean_utility": -0.02,
                                "win_rate": 0.0,
                                "mean_mae": 0.01,
                            },
                        },
                    }
                ],
            }
        }
        cfg = CalibratedPolicyConfig(
            min_train_samples=2,
            min_train_mean_net=0,
            min_train_win_rate=0.5,
            max_train_mean_mae=0.1,
        )
        stable = YearlyStableConfig(
            min_year_samples=1,
            min_year_mean_net=0,
            min_year_win_rate=0.5,
            max_year_mean_mae=0.1,
        )
        self.assertFalse(_fit_from_stats(stats, cfg, stable))


    def test_records_cache_path_is_stable_for_period_and_stride(self):
        cfg = CalibratedPolicyConfig(hold_candidates=(48, 96), window_size=96)
        with tempfile.TemporaryDirectory() as tmp:
            path = _records_cache_path(
                tmp,
                split="train",
                start_date="2023-01-01",
                end_date="2024-12-31",
                stride_bars=12,
                cfg=cfg,
            )
        self.assertIn("train_2023-01-01_2024-12-31_stride12_w96_h48-96", path.name)
        self.assertTrue(path.name.endswith(".json"))


    def test_rule_signature_ignores_dict_order(self):
        left = {
            "b": {"action": {"side": "SHORT", "hold_bars": 96}},
            "a": {"action": {"side": "LONG", "hold_bars": 48}},
        }
        right = {
            "a": {"action": {"side": "LONG", "hold_bars": 48}},
            "b": {"action": {"side": "SHORT", "hold_bars": 96}},
        }
        self.assertEqual(_rule_signature(left), _rule_signature(right))


    def test_load_or_build_records_reads_cache_without_market(self):
        cfg = CalibratedPolicyConfig(hold_candidates=(48,), window_size=96)
        with tempfile.TemporaryDirectory() as tmp:
            path = _records_cache_path(
                tmp,
                split="train",
                start_date="2023-01-01",
                end_date="2023-01-02",
                stride_bars=12,
                cfg=cfg,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('[{"date":"2023-01-01","key":"k","actions":{}}]')
            records = _load_or_build_records(
                None,
                cfg,
                start_date="2023-01-01",
                end_date="2023-01-02",
                stride_bars=12,
                split="train",
                records_cache_dir=tmp,
            )
        self.assertEqual(records[0]["key"], "k")


    def test_precompute_group_year_stats_keeps_empty_year_action_samples(self):
        records = [
            {
                "date": "2023-01-01",
                "key": "k",
                "actions": {"LONG_48": {"side": "LONG", "hold_bars": 48, "net_return": 0.01, "mae": 0.001, "utility": 0.009}},
            },
            {
                "date": "2024-01-01",
                "key": "k",
                "actions": {"SHORT_48": {"side": "SHORT", "hold_bars": 48, "net_return": 0.02, "mae": 0.002, "utility": 0.018}},
            },
        ]
        stats = _precompute_group_year_stats(records)
        long = next(action for action in stats["k"]["actions"] if action["overall"].get("side") == "LONG")
        self.assertEqual(long["yearly"]["2023"]["samples"], 1)
        self.assertEqual(long["yearly"]["2024"]["samples"], 0)
