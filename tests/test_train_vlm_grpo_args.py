import unittest

from training.train_vlm_grpo import parse_args


class TestTrainVlmGrpoArgs(unittest.TestCase):
    def test_parse_defaults_and_flags(self):
        import sys

        original = sys.argv
        try:
            sys.argv = [
                "train_vlm_grpo.py",
                "--max-steps",
                "2",
                "--load-in-4bit",
                "--source",
                "synthetic",
                "--sample-mode",
                "uniform",
                "--sample-seed",
                "7",
                "--action-schema",
                "trade_gate",
                "--prompt-style",
                "hybrid",
                "--prompt-feature-mode",
                "engineered_v1",
                "--target-horizon",
                "3",
                "--label-mode",
                "utility",
                "--buy-reward-weight",
                "1.4",
                "--hold-reward-weight",
                "1.0",
                "--sell-reward-weight",
                "0.9",
                "--reward-mode",
                "utility",
                "--utility-hold-margin",
                "0.0002",
                "--utility-fee-rate",
                "0.0004",
                "--utility-slippage-rate",
                "0.0001",
                "--utility-leverage",
                "2.0",
                "--utility-stop-loss",
                "0.02",
                "--utility-take-profit",
                "0.05",
                "--utility-use-log-return",
                "false",
                "--utility-base-risk-weight",
                "0.2",
                "--utility-regime-weight-volatility",
                "0.1",
                "--utility-regime-weight-downtrend",
                "0.1",
                "--utility-regime-weight-drawdown",
                "0.1",
                "--utility-min-risk-weight",
                "0.05",
                "--utility-max-risk-weight",
                "0.9",
                "--utility-hold-reward-bias",
                "-0.0001",
                "--utility-reward-scale",
                "350",
                "--utility-gap-scale",
                "420",
                "--max-completion-length",
                "5",
                "--min-new-tokens",
                "2",
                "--do-sample",
                "false",
                "--log-completions",
                "--num-completions-to-print",
                "4",
                "--scale-rewards",
                "none",
                "--reward-variance-guard",
                "off",
                "--dry-run",
            ]
            args = parse_args()
            self.assertEqual(args.max_steps, 2)
            self.assertTrue(args.load_in_4bit)
            self.assertEqual(args.source, "synthetic")
            self.assertEqual(args.sample_mode, "uniform")
            self.assertEqual(args.sample_seed, 7)
            self.assertEqual(args.action_schema, "trade_gate")
            self.assertEqual(args.prompt_style, "hybrid")
            self.assertEqual(args.prompt_feature_mode, "engineered_v1")
            self.assertEqual(args.target_horizon, 3)
            self.assertEqual(args.label_mode, "utility")
            self.assertAlmostEqual(args.buy_reward_weight, 1.4)
            self.assertAlmostEqual(args.hold_reward_weight, 1.0)
            self.assertAlmostEqual(args.sell_reward_weight, 0.9)
            self.assertEqual(args.reward_mode, "utility")
            self.assertAlmostEqual(args.utility_hold_margin, 0.0002)
            self.assertAlmostEqual(args.utility_fee_rate, 0.0004)
            self.assertAlmostEqual(args.utility_slippage_rate, 0.0001)
            self.assertAlmostEqual(args.utility_leverage, 2.0)
            self.assertAlmostEqual(args.utility_stop_loss, 0.02)
            self.assertAlmostEqual(args.utility_take_profit, 0.05)
            self.assertEqual(args.utility_use_log_return, "false")
            self.assertAlmostEqual(args.utility_base_risk_weight, 0.2)
            self.assertAlmostEqual(args.utility_regime_weight_volatility, 0.1)
            self.assertAlmostEqual(args.utility_regime_weight_downtrend, 0.1)
            self.assertAlmostEqual(args.utility_regime_weight_drawdown, 0.1)
            self.assertAlmostEqual(args.utility_min_risk_weight, 0.05)
            self.assertAlmostEqual(args.utility_max_risk_weight, 0.9)
            self.assertAlmostEqual(args.utility_hold_reward_bias, -0.0001)
            self.assertAlmostEqual(args.utility_reward_scale, 350)
            self.assertAlmostEqual(args.utility_gap_scale, 420)
            self.assertEqual(args.max_completion_length, 5)
            self.assertEqual(args.min_new_tokens, 2)
            self.assertEqual(args.do_sample, "false")
            self.assertTrue(args.log_completions)
            self.assertEqual(args.num_completions_to_print, 4)
            self.assertEqual(args.scale_rewards, "none")
            self.assertEqual(args.reward_variance_guard, "off")
            self.assertTrue(args.dry_run)
        finally:
            sys.argv = original


if __name__ == "__main__":
    unittest.main()
