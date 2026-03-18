import unittest

import numpy as np
import pandas as pd

from envs.trading_env import ACTION_BUY, ACTION_HOLD, TradingEnv, TradingEnvConfig


def _make_market_df(opens):
    opens = list(opens)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01 00:00:00", periods=len(opens), freq="1min"),
            "open": opens,
            "high": [x + 1.0 for x in opens],
            "low": [max(x - 1.0, 0.0) for x in opens],
            "close": opens,
            "volume": [1.0 for _ in opens],
        }
    )


class TestTradingEnv(unittest.TestCase):
    def test_buy_then_hold_has_expected_timing(self):
        market_df = _make_market_df([100.0, 100.0, 110.0, 121.0])
        env = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(window_size=2, initial_equity=1000.0, leverage=1.0),
        )

        _, _ = env.reset()
        _, reward1, terminated1, truncated1, info1 = env.step(ACTION_BUY)
        self.assertFalse(terminated1)
        self.assertFalse(truncated1)
        self.assertLess(reward1, 0.0)  # first step has only costs (flat -> long)
        self.assertEqual(info1["position_side"], 1)

        _, reward2, terminated2, truncated2, info2 = env.step(ACTION_HOLD)
        self.assertFalse(terminated2)
        self.assertTrue(truncated2)
        self.assertGreater(reward2, 0.0)  # long position benefits from 110 -> 121
        self.assertEqual(info2["position_side"], 0)  # HOLD defaults to flat target

    def test_hold_action_mode_maintain_keeps_side(self):
        market_df = _make_market_df([100.0, 100.0, 110.0, 121.0])
        env = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(
                window_size=2,
                initial_equity=1000.0,
                leverage=1.0,
                hold_action_mode="maintain",
            ),
        )

        _, _ = env.reset()
        env.step(ACTION_BUY)
        _, _, _, _, info2 = env.step(ACTION_HOLD)
        self.assertEqual(info2["position_side"], 1)

    def test_range_volatility_is_window_bounded(self):
        market_df = pd.DataFrame(
            {
                "date": pd.date_range(
                    "2025-01-01 00:00:00", periods=5, freq="1min"
                ),
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 2000, 3000],  # huge future values
                "low": [99, 100, 101, 1, 1],
                "close": [100, 101, 102, 103, 104],
                "volume": [1, 1, 1, 1, 1],
            }
        )
        env = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(window_size=3, initial_equity=1000.0),
        )
        obs, _ = env.reset()  # current_step = 2, should only look at rows 0..2

        max_high = 103.0
        min_low = 99.0
        expected = (max_high - min_low) / ((max_high + min_low) / 2.0)
        self.assertAlmostEqual(float(obs["scalars"][2]), expected, places=7)

    def test_liquidation_penalty(self):
        market_df = _make_market_df([100.0, 100.0, 100.0, 40.0])
        env = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(
                window_size=2,
                initial_equity=1000.0,
                leverage=10.0,
                liquidation_penalty=-10.0,
            ),
        )

        _, _ = env.reset()
        env.step(ACTION_BUY)  # enter long
        _, reward, terminated, truncated, info = env.step(ACTION_HOLD)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -10.0)
        self.assertTrue(info["liquidated"])

    def test_accepts_window_aligned_image_array(self):
        market_df = _make_market_df([100.0, 101.0, 102.0, 103.0, 104.0])
        # len = len(df)-window+1 = 4 for window_size=2
        images = np.zeros((4, 3, 32, 32), dtype=np.float32)
        env = TradingEnv(
            market_df=market_df,
            images=images,
            config=TradingEnvConfig(window_size=2, image_shape=(3, 32, 32)),
        )
        obs, _ = env.reset()
        self.assertEqual(obs["image"].shape, (3, 32, 32))

    def test_flat_hold_penalty_applies(self):
        market_df = _make_market_df([100.0, 100.0, 100.0, 100.0])
        env = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(window_size=2, flat_hold_penalty=0.01),
        )
        _, _ = env.reset()
        _, reward, _, _, _ = env.step(ACTION_HOLD)
        self.assertLess(reward, 0.0)

    def test_open_position_bonus_applies(self):
        market_df = _make_market_df([100.0, 100.0, 100.0, 100.0])
        env = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(window_size=2, open_position_bonus=0.02),
        )
        _, _ = env.reset()
        _, reward, _, _, _ = env.step(ACTION_BUY)
        # Buy from flat pays costs, but should still include positive bonus component.
        self.assertGreater(reward, -0.02)

    def test_random_reset_start_changes_start_step(self):
        market_df = _make_market_df([100.0 + i for i in range(30)])
        env = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(window_size=5, random_reset_start=True),
        )
        _, info1 = env.reset(seed=7)
        _, info2 = env.reset(seed=8)
        self.assertGreaterEqual(info1["step"], 4)
        self.assertGreaterEqual(info2["step"], 4)
        self.assertLessEqual(info1["step"], 28)
        self.assertLessEqual(info2["step"], 28)

    def test_episode_horizon_truncates_before_dataset_end(self):
        market_df = _make_market_df([100.0 + i for i in range(20)])
        env = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(window_size=5, episode_horizon=2),
        )
        _, info0 = env.reset()
        self.assertEqual(info0["episode_steps"], 0)

        _, _, terminated1, truncated1, info1 = env.step(ACTION_HOLD)
        self.assertFalse(terminated1)
        self.assertFalse(truncated1)
        self.assertEqual(info1["episode_steps"], 1)

        _, _, terminated2, truncated2, info2 = env.step(ACTION_HOLD)
        self.assertFalse(terminated2)
        self.assertTrue(truncated2)
        self.assertEqual(info2["episode_steps"], 2)
        self.assertLess(info2["step"], len(market_df) - 1)

    def test_random_reset_start_respects_episode_horizon(self):
        market_df = _make_market_df([100.0 + i for i in range(30)])
        env = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(
                window_size=5,
                random_reset_start=True,
                episode_horizon=4,
            ),
        )
        _, info = env.reset(seed=7)
        self.assertGreaterEqual(info["step"], 4)
        self.assertLessEqual(info["step"], 25)

    def test_observation_scalars_are_normalized(self):
        market_df = _make_market_df([100.0, 100.0, 105.0, 110.0])
        env = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(window_size=2),
        )
        obs, _ = env.reset()
        self.assertAlmostEqual(float(obs["scalars"][0]), 0.0)
        self.assertAlmostEqual(float(obs["scalars"][1]), 0.0)
        obs, _, _, _, _ = env.step(ACTION_BUY)
        self.assertAlmostEqual(float(obs["scalars"][0]), 1.0)
        self.assertLess(abs(float(obs["scalars"][1])), 0.2)

    def test_extended_scalar_feature_mode_exposes_richer_state(self):
        market_df = _make_market_df([100.0 + i for i in range(120)])
        env = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(
                window_size=96,
                scalar_feature_mode="extended_v1",
                episode_horizon=128,
                hold_action_mode="maintain",
            ),
        )
        obs, _ = env.reset()
        self.assertEqual(obs["scalars"].shape[0], 15)
        self.assertGreaterEqual(float(obs["scalars"][9]), -1.0)
        self.assertLessEqual(float(obs["scalars"][9]), 1.0)
        self.assertGreaterEqual(float(obs["scalars"][10]), -1.0)
        self.assertLessEqual(float(obs["scalars"][10]), 1.0)
        self.assertGreaterEqual(float(obs["scalars"][13]), 0.0)
        self.assertLessEqual(float(obs["scalars"][14]), 1.0)

    def test_directional_bias_penalty_reduces_reward_when_one_sided(self):
        market_df = _make_market_df([100.0, 100.0, 100.0, 100.0])
        env_no_penalty = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(window_size=2, directional_bias_penalty=0.0),
        )
        env_with_penalty = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(window_size=2, directional_bias_penalty=0.01),
        )

        env_no_penalty.reset()
        env_with_penalty.reset()
        _, reward_a, _, _, _ = env_no_penalty.step(ACTION_BUY)
        _, reward_b, _, _, info_b = env_with_penalty.step(ACTION_BUY)
        self.assertLess(reward_b, reward_a)
        self.assertGreaterEqual(float(info_b["directional_abs_mean"]), 0.0)

    def test_directional_symmetry_mirrors_buy_sell_mapping(self):
        market_df = _make_market_df([100.0, 100.0, 99.0, 98.0])
        env = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(window_size=2, directional_symmetry_prob=1.0),
        )
        obs, _ = env.reset(seed=123)
        # mirrored episode: policy-side scalar is mirrored, starts flat
        self.assertAlmostEqual(float(obs["scalars"][0]), 0.0)
        obs, _, _, _, info = env.step(ACTION_BUY)
        # policy BUY should map to market SELL when mirrored.
        self.assertEqual(info["mirror_sign"], -1)
        self.assertEqual(info["position_side"], -1)
        # observation side is mirrored back toward policy frame (+1)
        self.assertAlmostEqual(float(obs["scalars"][0]), 1.0)

    def test_action_balance_penalty_reduces_one_sided_policy_reward(self):
        market_df = _make_market_df([100.0, 100.0, 100.0, 100.0])
        env_base = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(window_size=2, action_balance_penalty=0.0),
        )
        env_penalized = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(window_size=2, action_balance_penalty=0.01),
        )
        env_base.reset()
        env_penalized.reset()
        _, r_base, _, _, _ = env_base.step(ACTION_BUY)
        _, r_pen, _, _, info_pen = env_penalized.step(ACTION_BUY)
        self.assertLess(r_pen, r_base)
        self.assertAlmostEqual(float(info_pen["policy_directional_imbalance"]), 1.0)

    def test_drawdown_penalty_reduces_reward_after_loss(self):
        market_df = _make_market_df([100.0, 100.0, 90.0, 90.0])
        env_base = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(window_size=2, hold_action_mode="maintain"),
        )
        env_pen = TradingEnv(
            market_df=market_df,
            config=TradingEnvConfig(
                window_size=2,
                hold_action_mode="maintain",
                drawdown_penalty=0.5,
            ),
        )
        env_base.reset()
        env_pen.reset()
        env_base.step(ACTION_BUY)
        env_pen.step(ACTION_BUY)
        _, r_base, _, _, info_base = env_base.step(ACTION_HOLD)
        _, r_pen, _, _, info_pen = env_pen.step(ACTION_HOLD)
        self.assertLess(r_pen, r_base)
        self.assertGreater(float(info_base["equity_drawdown_pct"]), 0.0)
        self.assertGreater(float(info_pen["equity_drawdown_pct"]), 0.0)


if __name__ == "__main__":
    unittest.main()
