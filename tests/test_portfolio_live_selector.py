import unittest
import pandas as pd

from execution.portfolio_live import _apply_portfolio_selector_overlay


def _frames():
    enriched = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3, freq="5min")})
    features = pd.DataFrame(
        {
            "htf_1d_return_1": [0.0, 0.0, 0.02],
            "htf_1d_range_pos": [0.5, 0.5, 0.9],
            "dxy_momentum": [0.0, 0.0, 0.0],
            "kimchi_premium_zscore": [0.0, 0.0, 1.0],
        }
    )
    return enriched, features


class TestPortfolioLiveSelector(unittest.TestCase):
    def test_portfolio_selector_blocks_bad_context_new_entries_only(self):
        enriched, features = _frames()
        sleeves = [
            {"name": "bear_rex_short", "active": True, "reasons": []},
            {"name": "already_inactive", "active": False, "reasons": []},
        ]
        overlay = {
            "name": "portfolio_bull_bear_oi_rex_llm_selector_overlay",
            "output_space": ["ALLOW", "BLOCK_RISK"],
            "symbolic_proxy": {
                "context_keys": ["trend_1d", "range_pos_1d", "dxy", "kimchi"],
                "blocked_contexts": [
                    {"context_id": "trend_1d=up|range_pos_1d=high|dxy=flat|kimchi=hot"}
                ],
            },
        }

        record = _apply_portfolio_selector_overlay(sleeves, overlay=overlay, enriched=enriched, features=features)

        self.assertEqual(record["action"], "BLOCK_RISK")
        self.assertFalse(record["allowed"])
        self.assertEqual(record["context_id"], "trend_1d=up|range_pos_1d=high|dxy=flat|kimchi=hot")
        self.assertFalse(sleeves[0]["active"])
        self.assertIn("portfolio_selector_context=", sleeves[0]["reasons"][-1])
        self.assertFalse(sleeves[1]["active"])

    def test_portfolio_selector_allows_unblocked_context(self):
        enriched, features = _frames()
        features.loc[len(features) - 1, "kimchi_premium_zscore"] = 0.0
        sleeves = [{"name": "bear_rex_short", "active": True, "reasons": []}]
        overlay = {
            "name": "portfolio_bull_bear_oi_rex_llm_selector_overlay",
            "symbolic_proxy": {
                "context_keys": ["trend_1d", "range_pos_1d", "dxy", "kimchi"],
                "blocked_contexts": [
                    {"context_id": "trend_1d=up|range_pos_1d=high|dxy=flat|kimchi=hot"}
                ],
            },
        }

        record = _apply_portfolio_selector_overlay(sleeves, overlay=overlay, enriched=enriched, features=features)

        self.assertEqual(record["action"], "ALLOW")
        self.assertTrue(record["allowed"])
        self.assertTrue(sleeves[0]["active"])


if __name__ == "__main__":
    unittest.main()
