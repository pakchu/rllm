import unittest

from training.train_sb3 import parse_args


class TestTrainSb3Args(unittest.TestCase):
    def test_parse_custom_args(self):
        # parse_args() reads sys.argv, so simulate via monkeypatching not needed by
        # directly importing argparse parser is unavailable; keep smoke check simple.
        import sys

        original = sys.argv
        try:
            sys.argv = [
                "train_sb3.py",
                "--source",
                "synthetic",
                "--synthetic-regime-amplitude",
                "0.0008",
                "--synthetic-regime-period",
                "480",
                "--n-envs",
                "4",
                "--vec-env",
                "dummy",
                "--timesteps",
                "512",
                "--use-images",
                "--image-resolution",
                "32",
                "--image-render-backend",
                "fast",
                "--flat-hold-penalty",
                "0.001",
                "--open-position-bonus",
                "0.002",
                "--directional-bias-penalty",
                "0.0006",
                "--directional-symmetry-prob",
                "0.4",
                "--action-balance-penalty",
                "0.0008",
                "--drawdown-penalty",
                "0.0012",
                "--random-reset-start",
                "--episode-horizon",
                "1024",
                "--scalar-feature-mode",
                "extended_v1",
                "--hold-action-mode",
                "maintain",
                "--learning-rate",
                "0.0003",
                "--ent-coef",
                "0.02",
                "--n-steps",
                "512",
                "--batch-size",
                "128",
                "--n-epochs",
                "5",
                "--target-kl",
                "0.03",
                "--mirror-augmentation",
                "--visual-backbone",
                "dinov2_reg_s",
                "--visual-pooling",
                "cls",
            ]
            args = parse_args()
            self.assertEqual(args.source, "synthetic")
            self.assertAlmostEqual(args.synthetic_regime_amplitude, 0.0008)
            self.assertEqual(args.synthetic_regime_period, 480)
            self.assertEqual(args.n_envs, 4)
            self.assertEqual(args.vec_env, "dummy")
            self.assertEqual(args.timesteps, 512)
            self.assertTrue(args.use_images)
            self.assertEqual(args.image_resolution, 32)
            self.assertEqual(args.image_render_backend, "fast")
            self.assertAlmostEqual(args.open_position_bonus, 0.002)
            self.assertAlmostEqual(args.directional_bias_penalty, 0.0006)
            self.assertAlmostEqual(args.directional_symmetry_prob, 0.4)
            self.assertAlmostEqual(args.action_balance_penalty, 0.0008)
            self.assertAlmostEqual(args.drawdown_penalty, 0.0012)
            self.assertTrue(args.random_reset_start)
            self.assertEqual(args.episode_horizon, 1024)
            self.assertEqual(args.scalar_feature_mode, "extended_v1")
            self.assertEqual(args.hold_action_mode, "maintain")
            self.assertAlmostEqual(args.learning_rate, 0.0003)
            self.assertAlmostEqual(args.ent_coef, 0.02)
            self.assertEqual(args.n_steps, 512)
            self.assertEqual(args.batch_size, 128)
            self.assertEqual(args.n_epochs, 5)
            self.assertAlmostEqual(args.target_kl, 0.03)
            self.assertTrue(args.mirror_augmentation)
            self.assertEqual(args.visual_backbone, "dinov2_reg_s")
            self.assertEqual(args.visual_pooling, "cls")
            self.assertFalse(args.unfreeze_visual_backbone)
        finally:
            sys.argv = original


if __name__ == "__main__":
    unittest.main()
