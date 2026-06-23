import json
import unittest

import pandas as pd

from training.llm_context_regime_miner import (
    LlmContextRegimeMinerCfg,
    _bucket_value,
    _fit_bucket_edges,
    _prompt,
    _htf_trend_stack,
    _price_action_event_tokens,
    _state_tokens,
    _target,
)


class TestLlmContextRegimeMiner(unittest.TestCase):
    def test_bucket_edges_fit_train_only(self):
        cfg = LlmContextRegimeMinerCfg(market_csv="m", output="o", train_start="2020-01-01", train_end="2020-04-19")
        dates = pd.Series(pd.date_range("2020-01-01", periods=120, freq="D"))
        vals = list(range(10)) * 11 + [10_000] * 10  # eval outliers must not shape train buckets
        features = pd.DataFrame({"x": vals[:120]})
        edges = _fit_bucket_edges(features, dates, cfg, feature_names=("x",))
        self.assertLess(max(edges["x"]), 10_000)
        self.assertEqual(_bucket_value(10_000, edges["x"]), "very_high")

    def test_prompt_hides_future_reward_terms(self):
        cfg = LlmContextRegimeMinerCfg(market_csv="m", output="o")
        prompt = _prompt("2026-01-01", {"trend_alignment": "aligned_up", "risk_state": "normal"}, cfg)
        self.assertIn("causal_state_tokens", prompt)
        self.assertNotIn("net_return", prompt)
        self.assertNotIn("mae_pct", prompt)
        self.assertNotIn("future_label", prompt)

    def test_target_selects_best_side_or_abstains(self):
        cfg = LlmContextRegimeMinerCfg(market_csv="m", output="o", horizon=2, min_trade_net_pct=0.1, min_trade_edge_gap_pct=0.05, max_trade_mae_pct=10.0)
        market = pd.DataFrame({
            "open": [100, 100, 103, 104, 104],
            "high": [101, 101, 104, 105, 105],
            "low": [99, 99, 102, 103, 103],
        })
        target, audit = _target(market, 0, cfg)
        self.assertEqual(target["action"], "LONG")
        self.assertIn("edge_gap_pct", audit)
        bad_cfg = LlmContextRegimeMinerCfg(market_csv="m", output="o", horizon=2, min_trade_net_pct=10.0)
        target2, _ = _target(market, 0, bad_cfg)
        self.assertEqual(target2["action"], "NO_TRADE")

    def test_state_tokens_are_bucket_strings(self):
        cols = {name: [0.0, 1.0] for name in __import__("training.llm_context_regime_miner", fromlist=["FEATURES"]).FEATURES}
        cols["external_any_available"] = [1.0, 1.0]
        cols["binance_aux_any_available"] = [0.0, 0.0]
        features = pd.DataFrame(cols)
        edges = {name: [-1, 0, 1] for name in cols if name not in {"external_any_available", "binance_aux_any_available"}}
        tokens = _state_tokens(features, 1, edges)
        self.assertEqual(tokens["external_availability"], "available")
        self.assertEqual(tokens["binance_aux_availability"], "missing_or_partial")
        self.assertIsInstance(tokens["trend_alignment"], str)
        self.assertIn("pa_event_pressure", tokens)
        self.assertIn("htf_trend_stack", tokens)

    def test_htf_trend_stack_compacts_multitimeframe_direction(self):
        tokens = {
            "htf_4h_return_4": "high",
            "htf_1d_return_4": "mid_high",
            "htf_3d_return_4": "very_high",
            "htf_1w_return_4": "low",
        }
        self.assertEqual(_htf_trend_stack(tokens), "htf_aligned_up")


    def test_price_action_event_tokens_are_compact(self):
        features = pd.DataFrame({
            "pae_w36_break_below": [0.0, 1.0],
            "pae_w576_failed_breakdown_long": [0.0, 1.0],
            "pae_w2016_high_sweep_reject": [0.0, 0.0],
        })
        tokens = _price_action_event_tokens(features, 1)
        self.assertEqual(tokens["pa_event_pressure"], "downside_break_or_reclaim")
        self.assertEqual(tokens["pa_long_window_event"], "long_window_downside_reclaim_candidate")
        self.assertEqual(tokens["pa_downside_reclaim"], "active_w576")


if __name__ == "__main__":
    unittest.main()
