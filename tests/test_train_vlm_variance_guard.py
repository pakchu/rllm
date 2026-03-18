import unittest

from training.train_vlm_grpo import _apply_reward_variance_guard


class TestVlmVarianceGuard(unittest.TestCase):
    def test_auto_guard_raises_batch_for_batch_scaling(self):
        batch, gens, notes = _apply_reward_variance_guard(
            per_device_train_batch_size=1,
            num_generations=2,
            scale_rewards="batch",
            dataset_size=64,
            label_counts={"BUY": 20, "HOLD": 24, "SELL": 20},
            reward_variance_guard="auto",
        )
        self.assertEqual(batch, 4)
        self.assertEqual(gens, 2)
        self.assertTrue(any("Raised per_device_train_batch_size" in n for n in notes))

    def test_off_guard_keeps_inputs(self):
        batch, gens, notes = _apply_reward_variance_guard(
            per_device_train_batch_size=1,
            num_generations=1,
            scale_rewards="batch",
            dataset_size=64,
            label_counts={"BUY": 20, "HOLD": 24, "SELL": 20},
            reward_variance_guard="off",
        )
        self.assertEqual(batch, 1)
        self.assertEqual(gens, 1)
        self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main()
