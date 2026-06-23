import json
import unittest

import numpy as np
import pandas as pd

from training.dxy_kimchi_regime_policy_dataset import (
    DxyKimchiPolicyDatasetCfg,
    _signal_for_value,
    _target,
    _prompt,
    _kimchi_signal_strength_bucket,
    _threshold_distance_bucket,
)


class TestDxyKimchiRegimePolicyDataset(unittest.TestCase):
    def test_signal_for_value_uses_frozen_rule_sides(self):
        rule = {"low_threshold": -1.0, "high_threshold": 1.0, "high_side": "SHORT", "low_side": "LONG"}
        self.assertEqual(_signal_for_value(2.0, rule), "SHORT")
        self.assertEqual(_signal_for_value(-2.0, rule), "LONG")
        self.assertEqual(_signal_for_value(0.0, rule), "NONE")

    def test_target_rejects_bad_path_and_activates_good_path(self):
        cfg = DxyKimchiPolicyDatasetCfg(market_csv="m", output="o", min_activate_net_pct=0.2, max_activate_mae_pct=6.0)
        good = _target(prior_side="LONG", audit={"net_return_pct": 0.5, "mae_pct": 2.0}, cfg=cfg)
        bad = _target(prior_side="LONG", audit={"net_return_pct": -0.1, "mae_pct": 2.0}, cfg=cfg)
        none = _target(prior_side="NONE", audit=None, cfg=cfg)
        self.assertTrue(good["activate"])
        self.assertEqual(good["action"], "LONG")
        self.assertFalse(bad["activate"])
        self.assertEqual(bad["action"], "NO_TRADE")
        self.assertEqual(none["reason_code"], "no_prior_signal")


    def test_threshold_distance_buckets_hide_raw_thresholds(self):
        self.assertEqual(_threshold_distance_bucket(-1.2, -1.0, direction="below"), "near")
        self.assertEqual(_threshold_distance_bucket(-2.0, -1.0, direction="below"), "deep")
        rule = {"high_side": "LONG", "low_side": "SHORT", "high_threshold": 1.0, "low_threshold": -1.0}
        self.assertEqual(_kimchi_signal_strength_bucket(1.8, "LONG", rule), "deep")
        self.assertEqual(_kimchi_signal_strength_bucket(-1.2, "SHORT", rule), "near")

    def test_prompt_contains_prior_but_not_future_reward(self):
        cfg = DxyKimchiPolicyDatasetCfg(market_csv="m", output="o")
        prompt = _prompt(
            date="2025-01-01",
            tokens={"dxy_zscore_bucket": "down", "kimchi_zscore_bucket": "up", "session_trend": "up"},
            prior_side="LONG",
            dxy_value=-1.2,
            kimchi_value=1.5,
            rule={"high_side": "LONG", "low_side": "SHORT", "high_threshold": 1.0, "low_threshold": -1.0},
            dxy_low_threshold=-0.5,
            cfg=cfg,
        )
        self.assertIn("prior_family: dxy_low_kimchi_zscore", prompt)
        self.assertIn("kimchi_prior_signal: LONG", prompt)
        self.assertIn("dxy_low_depth_bucket: medium", prompt)
        self.assertIn("kimchi_signal_strength_bucket: medium", prompt)
        self.assertIn("prior_side_trend_alignment: aligned", prompt)
        self.assertNotIn("net_return", prompt)
        self.assertNotIn("mae_pct", prompt)


if __name__ == "__main__":
    unittest.main()
