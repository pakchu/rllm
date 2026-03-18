import unittest

from training.select_vlm_checkpoint import (
    compute_selection_score,
    parse_step_list,
)


class TestSelectVlmCheckpoint(unittest.TestCase):
    def test_parse_step_list(self):
        self.assertEqual(parse_step_list("10, 20,30"), [10, 20, 30])

    def test_compute_selection_score_directional_bonus(self):
        base = {
            "accuracy": 0.5,
            "per_class": {
                "BUY": {"recall": 0.0},
                "HOLD": {"recall": 1.0},
                "SELL": {"recall": 0.0},
            },
        }
        directional = {
            "accuracy": 0.5,
            "per_class": {
                "BUY": {"recall": 0.4},
                "HOLD": {"recall": 0.2},
                "SELL": {"recall": 0.4},
            },
        }
        self.assertGreater(
            compute_selection_score(directional),
            compute_selection_score(base),
        )

    def test_compute_selection_score_penalizes_buy_sell_gap(self):
        balanced_directional = {
            "accuracy": 0.45,
            "per_class": {
                "BUY": {"recall": 0.6},
                "HOLD": {"recall": 0.2},
                "SELL": {"recall": 0.6},
            },
        }
        imbalanced_directional = {
            "accuracy": 0.45,
            "per_class": {
                "BUY": {"recall": 0.8},
                "HOLD": {"recall": 0.2},
                "SELL": {"recall": 0.4},
            },
        }
        self.assertGreater(
            compute_selection_score(balanced_directional),
            compute_selection_score(imbalanced_directional),
        )


if __name__ == "__main__":
    unittest.main()
