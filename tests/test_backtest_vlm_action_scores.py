import unittest

from training.backtest_vlm_action_scores import _action_to_hold_bars, _action_to_signal


class TestBacktestVlmActionScores(unittest.TestCase):
    def test_multi_horizon_labels_encode_side_and_hold(self):
        self.assertEqual(_action_to_signal("LONG_36"), 1)
        self.assertEqual(_action_to_signal("SHORT_144"), -1)
        self.assertEqual(_action_to_signal("NO_TRADE"), 0)
        self.assertEqual(_action_to_hold_bars("LONG_36", 144), 36)
        self.assertEqual(_action_to_hold_bars("SHORT_72", 144), 72)
        self.assertEqual(_action_to_hold_bars("BUY", 144), 144)


if __name__ == "__main__":
    unittest.main()
