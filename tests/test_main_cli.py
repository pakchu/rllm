import unittest

from main import build_parser


class TestMainCLI(unittest.TestCase):
    def test_parse_prepare_dataset(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "prepare-dataset",
                "--window-size",
                "64",
                "--resolution",
                "128",
                "--image-render-backend",
                "fast",
            ]
        )
        self.assertEqual(args.cmd, "prepare-dataset")
        self.assertEqual(args.window_size, 64)
        self.assertEqual(args.resolution, 128)
        self.assertEqual(args.image_render_backend, "fast")

    def test_parse_train_smoke(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "train-smoke",
                "--timesteps",
                "512",
                "--n-envs",
                "4",
                "--vec-env",
                "dummy",
                "--open-position-bonus",
                "0.002",
                "--directional-bias-penalty",
                "0.0005",
                "--directional-symmetry-prob",
                "0.5",
                "--action-balance-penalty",
                "0.0009",
                "--drawdown-penalty",
                "0.0011",
                "--random-reset-start",
                "--episode-horizon",
                "1024",
                "--scalar-feature-mode",
                "extended_v1",
                "--hold-action-mode",
                "maintain",
                "--target-kl",
                "0.03",
                "--seed",
                "7",
                "--synthetic-regime-amplitude",
                "0.0007",
                "--synthetic-regime-period",
                "360",
                "--mirror-augmentation",
                "--visual-backbone",
                "dinov2_reg_s",
                "--visual-pooling",
                "cls_patch_mean",
                "--image-render-backend",
                "fast",
            ]
        )
        self.assertEqual(args.cmd, "train-smoke")
        self.assertEqual(args.timesteps, 512)
        self.assertEqual(args.n_envs, 4)
        self.assertAlmostEqual(args.open_position_bonus, 0.002)
        self.assertAlmostEqual(args.directional_bias_penalty, 0.0005)
        self.assertAlmostEqual(args.directional_symmetry_prob, 0.5)
        self.assertAlmostEqual(args.action_balance_penalty, 0.0009)
        self.assertAlmostEqual(args.drawdown_penalty, 0.0011)
        self.assertTrue(args.random_reset_start)
        self.assertEqual(args.episode_horizon, 1024)
        self.assertEqual(args.scalar_feature_mode, "extended_v1")
        self.assertEqual(args.hold_action_mode, "maintain")
        self.assertAlmostEqual(args.target_kl, 0.03)
        self.assertEqual(args.seed, 7)
        self.assertAlmostEqual(args.synthetic_regime_amplitude, 0.0007)
        self.assertEqual(args.synthetic_regime_period, 360)
        self.assertTrue(args.mirror_augmentation)
        self.assertEqual(args.visual_backbone, "dinov2_reg_s")
        self.assertEqual(args.visual_pooling, "cls_patch_mean")
        self.assertEqual(args.image_render_backend, "fast")
        self.assertFalse(args.unfreeze_visual_backbone)

    def test_parse_train_vlm_action_schema(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "train-vlm-smoke",
                "--action-schema",
                "trade_gate",
                "--prompt-style",
                "symbolic",
            ]
        )
        self.assertEqual(args.cmd, "train-vlm-smoke")
        self.assertEqual(args.action_schema, "trade_gate")
        self.assertEqual(args.prompt_style, "symbolic")

    def test_parse_eval_vlm_action_schema(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "eval-vlm",
                "--action-schema",
                "trade_side",
                "--decision-mode",
                "likelihood",
            ]
        )
        self.assertEqual(args.cmd, "eval-vlm")
        self.assertEqual(args.action_schema, "trade_side")
        self.assertEqual(args.decision_mode, "likelihood")

    def test_parse_backtest(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "backtest",
                "--model-path",
                "abc.zip",
                "--source",
                "csv",
                "--input-csv",
                "m.csv",
                "--use-images",
                "true",
                "--image-render-backend",
                "fast",
                "--scalar-feature-mode",
                "extended_v1",
                "--flat-hold-penalty",
                "0.001",
                "--hold-action-mode",
                "flat",
                "--deterministic",
                "false",
                "--decision-mode",
                "score_band",
                "--flat-start-policy",
                "prefer_entry",
                "--debiased-action",
                "mirror_scalar",
                "--score-entry-threshold",
                "0.03",
                "--score-centering",
                "ema",
                "--score-center-alpha",
                "0.11",
                "--trend-guard",
                "hard",
                "--score-exit-threshold",
                "0.007",
                "--volatility-gate",
                "hard",
                "--volatility-threshold",
                "0.09",
                "--eval-seeds",
                "1,2,3",
            ]
        )
        self.assertEqual(args.cmd, "backtest")
        self.assertEqual(args.model_path, "abc.zip")
        self.assertEqual(args.source, "csv")
        self.assertEqual(args.input_csv, "m.csv")
        self.assertEqual(args.use_images, "true")
        self.assertEqual(args.image_render_backend, "fast")
        self.assertEqual(args.scalar_feature_mode, "extended_v1")
        self.assertAlmostEqual(args.flat_hold_penalty, 0.001)
        self.assertEqual(args.hold_action_mode, "flat")
        self.assertEqual(args.deterministic, "false")
        self.assertEqual(args.decision_mode, "score_band")
        self.assertEqual(args.flat_start_policy, "prefer_entry")
        self.assertEqual(args.debiased_action, "mirror_scalar")
        self.assertAlmostEqual(args.score_entry_threshold, 0.03)
        self.assertAlmostEqual(args.score_exit_threshold, 0.007)
        self.assertEqual(args.score_centering, "ema")
        self.assertAlmostEqual(args.score_center_alpha, 0.11)
        self.assertEqual(args.trend_guard, "hard")
        self.assertEqual(args.volatility_gate, "hard")
        self.assertAlmostEqual(args.volatility_threshold, 0.09)
        self.assertEqual(args.eval_seeds, "1,2,3")

    def test_parse_backtest_regime_switch(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "backtest",
                "--model-path",
                "base.zip",
                "--decision-mode",
                "regime_switch",
                "--regime-model-up",
                "up.zip",
                "--regime-model-down",
                "down.zip",
                "--regime-neutral-policy",
                "hold",
                "--regime-transition-mode",
                "none",
                "--regime-score-mode",
                "raw",
                "--regime-enter-threshold",
                "0.8",
                "--regime-confirm-bars",
                "5",
            ]
        )
        self.assertEqual(args.cmd, "backtest")
        self.assertEqual(args.decision_mode, "regime_switch")
        self.assertEqual(args.regime_model_up, "up.zip")
        self.assertEqual(args.regime_model_down, "down.zip")
        self.assertEqual(args.regime_neutral_policy, "hold")
        self.assertEqual(args.regime_transition_mode, "none")
        self.assertEqual(args.regime_score_mode, "raw")
        self.assertAlmostEqual(args.regime_enter_threshold, 0.8)
        self.assertEqual(args.regime_confirm_bars, 5)

    def test_parse_backtest_trend_guard_long_flat(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "backtest",
                "--model-path",
                "base.zip",
                "--trend-guard",
                "long_flat",
                "--trend-threshold",
                "0.015",
            ]
        )
        self.assertEqual(args.cmd, "backtest")
        self.assertEqual(args.trend_guard, "long_flat")
        self.assertAlmostEqual(args.trend_threshold, 0.015)

    def test_parse_backtest_blend_score_band(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "backtest",
                "--model-path",
                "base.zip",
                "--decision-mode",
                "blend_score_band",
                "--blend-model-a",
                "a.zip",
                "--blend-model-b",
                "b.zip",
                "--blend-weight-mode",
                "trend",
                "--blend-weight-a",
                "0.7",
                "--blend-weight-up",
                "0.95",
                "--blend-weight-down",
                "0.3",
                "--blend-trend-threshold",
                "0.004",
            ]
        )
        self.assertEqual(args.cmd, "backtest")
        self.assertEqual(args.decision_mode, "blend_score_band")
        self.assertEqual(args.blend_model_a, "a.zip")
        self.assertEqual(args.blend_model_b, "b.zip")
        self.assertEqual(args.blend_weight_mode, "trend")
        self.assertAlmostEqual(args.blend_weight_a, 0.7)
        self.assertAlmostEqual(args.blend_weight_up, 0.95)
        self.assertAlmostEqual(args.blend_weight_down, 0.3)
        self.assertAlmostEqual(args.blend_trend_threshold, 0.004)

    def test_parse_train_real(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "train-real",
                "--start-date",
                "2025-01-01",
                "--end-date",
                "2025-01-02",
                "--n-envs",
                "2",
                "--directional-bias-penalty",
                "0.0007",
                "--directional-symmetry-prob",
                "0.25",
                "--action-balance-penalty",
                "0.0004",
                "--drawdown-penalty",
                "0.0007",
                "--episode-horizon",
                "2048",
                "--scalar-feature-mode",
                "extended_v1",
                "--visual-backbone",
                "hf_custom",
                "--visual-backbone-model",
                "facebook/dinov2-with-registers-base",
                "--visual-pooling",
                "mean",
                "--unfreeze-visual-backbone",
            ]
        )
        self.assertEqual(args.cmd, "train-real")
        self.assertEqual(args.start_date, "2025-01-01")
        self.assertEqual(args.end_date, "2025-01-02")
        self.assertEqual(args.n_envs, 2)
        self.assertAlmostEqual(args.directional_bias_penalty, 0.0007)
        self.assertAlmostEqual(args.directional_symmetry_prob, 0.25)
        self.assertAlmostEqual(args.action_balance_penalty, 0.0004)
        self.assertAlmostEqual(args.drawdown_penalty, 0.0007)
        self.assertEqual(args.episode_horizon, 2048)
        self.assertEqual(args.scalar_feature_mode, "extended_v1")
        self.assertEqual(args.visual_backbone, "hf_custom")
        self.assertEqual(args.visual_backbone_model, "facebook/dinov2-with-registers-base")
        self.assertEqual(args.visual_pooling, "mean")
        self.assertTrue(args.unfreeze_visual_backbone)

    def test_parse_train_vlm_smoke(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "train-vlm-smoke",
                "--source",
                "synthetic",
                "--max-steps",
                "3",
                "--max-samples",
                "16",
                "--sample-mode",
                "uniform",
                "--sample-seed",
                "9",
                "--prompt-style",
                "symbolic",
                "--prompt-feature-mode",
                "engineered_v1",
                "--target-horizon",
                "3",
                "--label-mode",
                "utility",
                "--buy-reward-weight",
                "1.5",
                "--reward-mode",
                "utility",
                "--utility-stop-loss",
                "0.02",
                "--utility-use-log-return",
                "false",
                "--temperature",
                "1.2",
                "--max-completion-length",
                "6",
                "--min-new-tokens",
                "2",
                "--do-sample",
                "false",
                "--log-completions",
                "--num-completions-to-print",
                "3",
                "--scale-rewards",
                "none",
                "--reward-variance-guard",
                "off",
                "--synthetic-regime-amplitude",
                "0.0006",
                "--dry-run",
            ]
        )
        self.assertEqual(args.cmd, "train-vlm-smoke")
        self.assertEqual(args.max_steps, 3)
        self.assertEqual(args.max_samples, 16)
        self.assertEqual(args.sample_mode, "uniform")
        self.assertEqual(args.sample_seed, 9)
        self.assertEqual(args.prompt_style, "symbolic")
        self.assertEqual(args.prompt_feature_mode, "engineered_v1")
        self.assertEqual(args.target_horizon, 3)
        self.assertEqual(args.label_mode, "utility")
        self.assertAlmostEqual(args.buy_reward_weight, 1.5)
        self.assertEqual(args.reward_mode, "utility")
        self.assertAlmostEqual(args.utility_stop_loss, 0.02)
        self.assertEqual(args.utility_use_log_return, "false")
        self.assertAlmostEqual(args.temperature, 1.2)
        self.assertEqual(args.max_completion_length, 6)
        self.assertEqual(args.min_new_tokens, 2)
        self.assertEqual(args.do_sample, "false")
        self.assertTrue(args.log_completions)
        self.assertEqual(args.num_completions_to_print, 3)
        self.assertEqual(args.scale_rewards, "none")
        self.assertEqual(args.reward_variance_guard, "off")
        self.assertAlmostEqual(args.synthetic_regime_amplitude, 0.0006)
        self.assertTrue(args.dry_run)

    def test_parse_eval_vlm(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "eval-vlm",
                "--model-name",
                "auto",
                "--adapter-dir",
                "/tmp/a",
                "--source",
                "csv",
                "--input-csv",
                "m.csv",
                "--max-samples",
                "30",
                "--target-horizon",
                "5",
                "--label-mode",
                "utility",
                "--utility-stop-loss",
                "0.03",
                "--sample-mode",
                "uniform",
                "--prompt-style",
                "hybrid",
                "--prompt-feature-mode",
                "engineered_v1",
                "--do-sample",
                "true",
                "--temperature",
                "1.5",
                "--decision-mode",
                "likelihood",
                "--action-bias-buy",
                "0.2",
                "--store-action-scores",
                "true",
            ]
        )
        self.assertEqual(args.cmd, "eval-vlm")
        self.assertEqual(args.model_name, "auto")
        self.assertEqual(args.adapter_dir, "/tmp/a")
        self.assertEqual(args.source, "csv")
        self.assertEqual(args.input_csv, "m.csv")
        self.assertEqual(args.max_samples, 30)
        self.assertEqual(args.target_horizon, 5)
        self.assertEqual(args.label_mode, "utility")
        self.assertAlmostEqual(args.utility_stop_loss, 0.03)
        self.assertEqual(args.sample_mode, "uniform")
        self.assertEqual(args.prompt_style, "hybrid")
        self.assertEqual(args.prompt_feature_mode, "engineered_v1")
        self.assertEqual(args.do_sample, "true")
        self.assertAlmostEqual(args.temperature, 1.5)
        self.assertEqual(args.decision_mode, "likelihood")
        self.assertAlmostEqual(args.action_bias_buy, 0.2)
        self.assertEqual(args.store_action_scores, "true")

    def test_parse_select_vlm_checkpoint(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "select-vlm-checkpoint",
                "--candidate-steps",
                "10,20",
                "--source",
                "csv",
                "--input-csv",
                "m.csv",
                "--prompt-style",
                "hybrid",
                "--prompt-feature-mode",
                "engineered_v1",
                "--target-horizon",
                "4",
                "--label-mode",
                "utility",
                "--buy-reward-weight",
                "1.2",
                "--reward-mode",
                "utility",
                "--utility-stop-loss",
                "0.02",
                "--sample-mode",
                "uniform",
                "--eval-max-samples",
                "60",
                "--eval-sample-mode",
                "uniform",
                "--eval-decision-mode",
                "likelihood",
                "--eval-action-bias-buy",
                "0.1",
            ]
        )
        self.assertEqual(args.cmd, "select-vlm-checkpoint")
        self.assertEqual(args.candidate_steps, "10,20")
        self.assertEqual(args.source, "csv")
        self.assertEqual(args.input_csv, "m.csv")
        self.assertEqual(args.prompt_style, "hybrid")
        self.assertEqual(args.prompt_feature_mode, "engineered_v1")
        self.assertEqual(args.target_horizon, 4)
        self.assertEqual(args.label_mode, "utility")
        self.assertAlmostEqual(args.buy_reward_weight, 1.2)
        self.assertEqual(args.reward_mode, "utility")
        self.assertAlmostEqual(args.utility_stop_loss, 0.02)
        self.assertEqual(args.sample_mode, "uniform")
        self.assertEqual(args.eval_max_samples, 60)
        self.assertEqual(args.eval_sample_mode, "uniform")
        self.assertEqual(args.eval_decision_mode, "likelihood")
        self.assertAlmostEqual(args.eval_action_bias_buy, 0.1)

    def test_parse_calibrate_vlm_bias(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "calibrate-vlm-bias",
                "--input-report",
                "a.json",
                "--input-report",
                "b.json",
                "--buy-min",
                "-0.7",
                "--buy-max",
                "0.9",
                "--buy-step",
                "0.2",
                "--weight-directional-gap",
                "0.2",
            ]
        )
        self.assertEqual(args.cmd, "calibrate-vlm-bias")
        self.assertEqual(args.input_report, ["a.json", "b.json"])
        self.assertAlmostEqual(args.buy_min, -0.7)
        self.assertAlmostEqual(args.buy_max, 0.9)
        self.assertAlmostEqual(args.buy_step, 0.2)
        self.assertAlmostEqual(args.weight_directional_gap, 0.2)

    def test_parse_cnn_dryrun(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "cnn-dryrun",
                "--chunks",
                "3",
                "--regime-amplitude",
                "0.0012",
                "--directional-bias-penalty",
                "0.0002",
            ]
        )
        self.assertEqual(args.cmd, "cnn-dryrun")
        self.assertEqual(args.chunks, 3)
        self.assertAlmostEqual(args.regime_amplitude, 0.0012)
        self.assertAlmostEqual(args.directional_bias_penalty, 0.0002)

    def test_parse_optiona_walkforward(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "optiona-walkforward",
                "--model-path",
                "m.zip",
                "--symbol",
                "ETHUSDT",
                "--hold-action-mode",
                "flat",
                "--use-images",
                "false",
                "--folds",
                "2025-03-01:2025-03-07:2025-03-08:2025-03-14",
            ]
        )
        self.assertEqual(args.cmd, "optiona-walkforward")
        self.assertEqual(args.model_path, "m.zip")
        self.assertEqual(args.symbol, "ETHUSDT")
        self.assertEqual(args.hold_action_mode, "flat")
        self.assertEqual(args.use_images, "false")
        self.assertEqual(
            args.folds,
            "2025-03-01:2025-03-07:2025-03-08:2025-03-14",
        )

    def test_parse_robust_sweep(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "robust-sweep",
                "--timesteps",
                "1024",
                "--eval-seeds",
                "10,20",
                "--synthetic-regime-amplitude",
                "0.0009",
                "--max-candidates",
                "2",
            ]
        )
        self.assertEqual(args.cmd, "robust-sweep")
        self.assertEqual(args.timesteps, 1024)
        self.assertEqual(args.eval_seeds, "10,20")
        self.assertAlmostEqual(args.synthetic_regime_amplitude, 0.0009)
        self.assertEqual(args.max_candidates, 2)


if __name__ == "__main__":
    unittest.main()
