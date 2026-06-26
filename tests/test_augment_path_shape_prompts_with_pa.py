import json
import unittest

import pandas as pd

from training.augment_path_shape_prompts_with_pa import _macro_by_date, augment_row, macro_tokens, micro_path_tokens, price_action_tokens
from training.path_shape_token_policy_tte import tokens_from_row


class TestAugmentPathShapePromptsWithPA(unittest.TestCase):
    def test_price_action_tokens_use_signal_and_past_window(self):
        market = pd.DataFrame({"close": [100, 101, 102, 101], "high": [101, 102, 103, 102], "low": [99, 100, 101, 100], "volume": [1, 2, 3, 4]})
        toks, nums = price_action_tokens(market, 3, [3])
        self.assertTrue(any(t.startswith("pa.w3.range_pos=") for t in toks))
        self.assertIn("pa.w3.return_pct", nums)

    def test_augment_row_adds_tokens_consumed_by_policy(self):
        market = pd.DataFrame({"close": [100, 101, 102, 101], "high": [101, 102, 103, 102], "low": [99, 100, 101, 100], "volume": [1, 2, 3, 4]})
        row = {"signal_pos": 3, "prompt": 'Past-only analyzer summary: {"regime":"UPTREND"}\n\nAnalyzer path-shape output: {}'}
        out = augment_row(row, market, [3])
        summary = json.loads(out["prompt"].split("Past-only analyzer summary: ", 1)[1].split("\n\nAnalyzer", 1)[0])
        self.assertIn("augmented_price_action_tokens", summary)
        self.assertTrue(any(t.startswith("aug.pa.w3") for t in tokens_from_row(out)))


if __name__ == "__main__":
    unittest.main()


class TestMacroAugmentation(unittest.TestCase):
    def test_macro_tokens_bucket_context_values(self):
        toks, nums = macro_tokens({"dxy_zscore": 2.2, "dxy_momentum": 0.004, "kimchi_available": 1})
        self.assertIn("macro.dxy.z=HIGH_EXTREME", toks)
        self.assertIn("macro.dxy.mom=UP", toks)
        self.assertIn("macro.kimchi.available=1", toks)
        self.assertIn("macro.dxy.z", nums)

    def test_macro_by_date_uses_backward_asof(self):
        ctx = pd.DataFrame({"date": ["2025-01-01 00:00:00", "2025-01-01 00:10:00"], "dxy_zscore": [1.0, 2.0]})
        rows = [{"date": "2025-01-01 00:05:00"}]
        out = _macro_by_date(ctx, rows)
        self.assertEqual(float(out["0"]["dxy_zscore"]), 1.0)

    def test_augment_row_adds_macro_tokens_consumed_by_policy(self):
        market = pd.DataFrame({"close": [100, 101, 102, 101], "high": [101, 102, 103, 102], "low": [99, 100, 101, 100], "volume": [1, 2, 3, 4]})
        row = {"signal_pos": 3, "prompt": 'Past-only analyzer summary: {"regime":"UPTREND"}\n\nAnalyzer path-shape output: {}'}
        out = augment_row(row, market, [3], {"dxy_zscore": 2.2})
        self.assertTrue(any(t.startswith("aug.macro.dxy.z=") for t in tokens_from_row(out)))


class TestMicroPathAugmentation(unittest.TestCase):
    def test_micro_path_tokens_capture_recent_trajectory(self):
        market = pd.DataFrame({
            "open": [100, 101, 102, 103, 102, 101, 103, 104],
            "close": [101, 102, 103, 102, 101, 103, 104, 105],
            "high": [102, 103, 104, 104, 103, 104, 105, 106],
            "low": [99, 100, 101, 101, 100, 100, 102, 103],
            "volume": [1, 1, 2, 3, 2, 2, 4, 5],
        })
        toks, nums = micro_path_tokens(market, 7, [6])
        self.assertTrue(any(t.startswith("micro.w6.return=") for t in toks))
        self.assertTrue(any(t.startswith("micro.w6.alternation=") for t in toks))
        self.assertIn("micro.w6.body_efficiency", nums)

    def test_augment_row_adds_micro_tokens_consumed_by_policy(self):
        market = pd.DataFrame({
            "open": [100, 101, 102, 103, 102, 101, 103, 104],
            "close": [101, 102, 103, 102, 101, 103, 104, 105],
            "high": [102, 103, 104, 104, 103, 104, 105, 106],
            "low": [99, 100, 101, 101, 100, 100, 102, 103],
            "volume": [1, 1, 2, 3, 2, 2, 4, 5],
        })
        row = {"signal_pos": 7, "prompt": 'Past-only analyzer summary: {"regime":"UPTREND"}\n\nAnalyzer path-shape output: {}'}
        out = augment_row(row, market, [3], micro_windows=[6])
        self.assertTrue(any(t.startswith("aug.micro.w6") for t in tokens_from_row(out)))
