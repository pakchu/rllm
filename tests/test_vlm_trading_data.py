import unittest

import numpy as np
import pandas as pd

from training.vlm_trading_data import (
    action_from_next_return,
    action_from_utilities,
    build_vlm_training_samples,
    completion_text_to_label,
    compute_action_utilities,
    compute_dynamic_risk_weight,
    reward_from_action,
    samples_to_hf_records,
)


def _market_df(n: int = 120) -> pd.DataFrame:
    base = [100.0 + i * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="1min"),
            "open": base,
            "high": [x * 1.01 for x in base],
            "low": [x * 0.99 for x in base],
            "close": [x * 1.001 for x in base],
            "volume": [1.0 for _ in base],
        }
    )


def _oscillating_market_df(n: int = 120) -> pd.DataFrame:
    opens = []
    p = 100.0
    for i in range(n):
        p = p * (1.01 if i % 2 == 0 else 0.99)
        opens.append(p)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="1min"),
            "open": opens,
            "high": [x * 1.01 for x in opens],
            "low": [x * 0.99 for x in opens],
            "close": [x * 1.001 for x in opens],
            "volume": [1.0 for _ in opens],
        }
    )


def _rich_market_df(n: int = 140) -> pd.DataFrame:
    df = _oscillating_market_df(n)
    df["number_of_trades"] = np.linspace(100, 300, n)
    df["taker_buy_base"] = np.where(np.arange(n) % 2 == 0, 0.7, 0.3)
    df["funding_rate"] = np.sin(np.linspace(0, 6, n)) * 0.01
    df["open_interest"] = np.linspace(1000, 1500, n)
    return df


class TestVlmTradingData(unittest.TestCase):
    def test_action_from_return(self):
        self.assertEqual(action_from_next_return(0.01), "BUY")
        self.assertEqual(action_from_next_return(-0.01), "SELL")
        self.assertEqual(action_from_next_return(0.0), "HOLD")
        self.assertEqual(action_from_utilities(0.004, 0.0039, -0.01, hold_margin=0.0002), "HOLD")

    def test_completion_text_to_label(self):
        self.assertEqual(completion_text_to_label("I choose BUY"), "BUY")
        self.assertEqual(
            completion_text_to_label([{"content": [{"text": "sell now"}]}]), "SELL"
        )
        self.assertEqual(
            completion_text_to_label(
                [
                    {
                        "role": "user",
                        "content": "Return exactly one action token: BUY or HOLD or SELL.",
                    },
                    {"role": "assistant", "content": "SELL"},
                ]
            ),
            "SELL",
        )

    def test_reward_sign(self):
        r_ok = reward_from_action("BUY", "BUY", 0.01)
        r_bad = reward_from_action("SELL", "BUY", 0.01)
        self.assertGreater(r_ok, 0)
        self.assertLess(r_bad, 0)

    def test_reward_opposite_is_worse_than_neutral(self):
        r_hold_miss = reward_from_action("HOLD", "BUY", 0.01)
        r_opposite = reward_from_action("SELL", "BUY", 0.01)
        self.assertLess(r_opposite, r_hold_miss)

    def test_reward_weight_scales_target_class_signal(self):
        base = reward_from_action("BUY", "BUY", 0.01, buy_reward_weight=1.0)
        boosted = reward_from_action("BUY", "BUY", 0.01, buy_reward_weight=1.8)
        self.assertGreater(boosted, base)

    def test_reward_utility_mode_prefers_higher_utility_action(self):
        r_buy = reward_from_action(
            "BUY",
            "BUY",
            0.0,
            reward_mode="utility",
            action_utility_buy=0.0020,
            action_utility_hold=0.0,
            action_utility_sell=-0.0020,
            utility_reward_scale=400.0,
            utility_gap_scale=400.0,
        )
        r_sell = reward_from_action(
            "SELL",
            "BUY",
            0.0,
            reward_mode="utility",
            action_utility_buy=0.0020,
            action_utility_hold=0.0,
            action_utility_sell=-0.0020,
            utility_reward_scale=400.0,
            utility_gap_scale=400.0,
        )
        self.assertGreater(r_buy, r_sell)

    def test_compute_action_utilities_cost_and_risk(self):
        low_risk = compute_action_utilities(
            open_t=100.0,
            open_th=101.0,
            horizon_min_low=99.5,
            horizon_max_high=101.5,
            fee_rate=0.0004,
            slippage_rate=0.0001,
            leverage=1.0,
            dynamic_risk_weight=0.0,
        )
        high_risk = compute_action_utilities(
            open_t=100.0,
            open_th=101.0,
            horizon_min_low=99.5,
            horizon_max_high=101.5,
            fee_rate=0.0004,
            slippage_rate=0.0001,
            leverage=1.0,
            dynamic_risk_weight=1.0,
        )
        self.assertGreater(low_risk["BUY"], low_risk["SELL"])
        self.assertLess(high_risk["BUY"], low_risk["BUY"])

    def test_compute_dynamic_risk_weight_increases_under_stress(self):
        calm = compute_dynamic_risk_weight(
            range_volatility_pct=0.005,
            trend_pct=0.01,
            drawdown_pct=0.01,
            base_risk_weight=0.2,
            regime_weight_volatility=0.2,
            regime_weight_downtrend=0.2,
            regime_weight_drawdown=0.2,
        )
        stress = compute_dynamic_risk_weight(
            range_volatility_pct=0.05,
            trend_pct=-0.03,
            drawdown_pct=0.15,
            base_risk_weight=0.2,
            regime_weight_volatility=0.2,
            regime_weight_downtrend=0.2,
            regime_weight_drawdown=0.2,
        )
        self.assertGreater(stress, calm)

    def test_build_samples_and_records(self):
        samples = build_vlm_training_samples(
            market_df=_market_df(),
            timeframe="1m",
            window_size=32,
            resolution=32,
            cache_dir=None,
            max_samples=5,
        )
        self.assertEqual(len(samples), 5)
        self.assertIn(samples[0].target_action, {"BUY", "HOLD", "SELL"})
        records = samples_to_hf_records(samples)
        self.assertEqual(len(records), 5)
        self.assertIn("prompt", records[0])
        self.assertIn("image", records[0])
        self.assertIsInstance(records[0]["prompt"], list)
        self.assertEqual(records[0]["prompt"][0]["role"], "system")
        self.assertEqual(records[0]["prompt"][1]["role"], "user")

    def test_build_samples_symbolic_prompt_style(self):
        samples = build_vlm_training_samples(
            market_df=_oscillating_market_df(),
            timeframe="1m",
            window_size=32,
            resolution=32,
            cache_dir=None,
            max_samples=4,
            sample_mode="random",
            sample_seed=7,
            prompt_style="symbolic",
        )
        self.assertEqual(len(samples), 4)
        self.assertIn("Regime:", samples[0].prompt)
        self.assertIn("Momentum:", samples[0].prompt)
        self.assertNotIn("Position Size (%)", samples[0].prompt)

    def test_build_samples_engineered_prompt_mode(self):
        samples = build_vlm_training_samples(
            market_df=_rich_market_df(),
            timeframe="5m",
            window_size=32,
            resolution=32,
            cache_dir=None,
            max_samples=4,
            sample_mode="uniform",
            prompt_style="hybrid",
            prompt_feature_mode="engineered_v1",
        )
        self.assertEqual(len(samples), 4)
        prompt = samples[0].prompt
        self.assertIn("Close Z-Score (48):", prompt)
        self.assertIn("Trend Alignment:", prompt)
        self.assertIn("Candle Pattern:", prompt)
        self.assertIn("Order Flow:", prompt)
        self.assertIn("Tags:", prompt)

    def test_build_samples_random_mode_is_seeded(self):
        s1 = build_vlm_training_samples(
            market_df=_oscillating_market_df(),
            timeframe="1m",
            window_size=32,
            resolution=32,
            cache_dir=None,
            max_samples=8,
            sample_mode="random",
            sample_seed=3,
        )
        s2 = build_vlm_training_samples(
            market_df=_oscillating_market_df(),
            timeframe="1m",
            window_size=32,
            resolution=32,
            cache_dir=None,
            max_samples=8,
            sample_mode="random",
            sample_seed=3,
        )
        self.assertEqual([x.date for x in s1], [x.date for x in s2])

    def test_build_samples_balanced_mode_uses_multiple_labels(self):
        samples = build_vlm_training_samples(
            market_df=_oscillating_market_df(),
            timeframe="1m",
            window_size=32,
            resolution=32,
            cache_dir=None,
            max_samples=18,
            hold_band=0.0005,
            sample_mode="balanced",
            sample_seed=11,
        )
        labels = {s.target_action for s in samples}
        self.assertEqual(len(samples), 18)
        self.assertGreaterEqual(len(labels), 2)

    def test_target_horizon_changes_available_sample_count(self):
        s_h1 = build_vlm_training_samples(
            market_df=_market_df(80),
            timeframe="1m",
            window_size=16,
            resolution=32,
            cache_dir=None,
            max_samples=None,
            target_horizon=1,
        )
        s_h5 = build_vlm_training_samples(
            market_df=_market_df(80),
            timeframe="1m",
            window_size=16,
            resolution=32,
            cache_dir=None,
            max_samples=None,
            target_horizon=5,
        )
        self.assertGreater(len(s_h1), len(s_h5))

    def test_label_mode_utility_populates_action_utilities(self):
        samples = build_vlm_training_samples(
            market_df=_oscillating_market_df(90),
            timeframe="1m",
            window_size=16,
            resolution=32,
            cache_dir=None,
            max_samples=10,
            target_horizon=3,
            label_mode="utility",
            utility_base_risk_weight=0.2,
            utility_regime_weight_volatility=0.2,
            utility_regime_weight_downtrend=0.2,
            utility_regime_weight_drawdown=0.2,
            sample_mode="random",
            sample_seed=5,
        )
        self.assertEqual(len(samples), 10)
        self.assertTrue(all(np.isfinite(s.action_utility_buy) for s in samples))
        self.assertTrue(all(np.isfinite(s.action_utility_hold) for s in samples))
        self.assertTrue(all(np.isfinite(s.action_utility_sell) for s in samples))
        self.assertTrue(any(s.dynamic_risk_weight > 0.0 for s in samples))


if __name__ == "__main__":
    unittest.main()
