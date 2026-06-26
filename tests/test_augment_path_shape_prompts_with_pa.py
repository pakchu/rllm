import json
import unittest

import pandas as pd

from training.augment_path_shape_prompts_with_pa import augment_row, price_action_tokens
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
