import json
import unittest

from training.event_action_compressor_ranker_data import convert_row, parse_prompt_sections, state_tokens


PROMPT = """Date: 2026-01-01
Past-only state: {"trend_24":0.01,"trend_96":-0.02,"range_pos":0.8,"bb_z":1.2,"rsi_norm":0.1,"volume_zscore":0.2,"taker_imbalance":-0.03,"dxy_zscore":0.0,"usdkrw_zscore":0.0,"kimchi_premium_zscore":0.9,"kimchi_premium_change":-0.001,"htf_4h_return_4":0.01,"htf_1d_return_1":-0.02,"htf_1w_return_4":0.03,"window_drawdown":0.04}
Action book: [{"family":"f","side":"LONG","strength":0.5}]
Candidate action: {"family":"f","side":"SHORT","hold_bars":144,"strength":0.5}
"""


class TestEventActionCompressorRankerData(unittest.TestCase):
    def test_parse_prompt_sections(self):
        sections = parse_prompt_sections(PROMPT)
        self.assertEqual(sections["Candidate action"]["side"], "SHORT")
        self.assertEqual(len(sections["Action book"]), 1)

    def test_state_tokens_include_side_interactions(self):
        sections = parse_prompt_sections(PROMPT)
        toks = state_tokens(sections["Past-only state"], sections["Candidate action"])
        self.assertEqual(toks["side"], "SHORT")
        self.assertIn("side_trend_24", toks)
        self.assertEqual(toks["kimchi_level"], "up")

    def test_convert_row_keeps_reward_label_only(self):
        row = {
            "date": "2026-01-01",
            "signal_pos": 1,
            "prompt": PROMPT,
            "action_audit": {"rank_utility": 0.02, "net_return": 0.03, "mae": 0.01, "mfe": 0.04},
        }
        out = convert_row(row, "rank_utility")
        self.assertEqual(out["reward"]["net_return_pct"], 0.02)
        self.assertNotIn("target", out["feature_snapshot"])
        self.assertTrue(json.loads(out["llm_compressor_target"])["side"] == "SHORT")
        self.assertTrue(out["leakage_guard"]["reward_is_label_only"])


if __name__ == "__main__":
    unittest.main()
