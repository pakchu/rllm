import unittest

from training.apply_event_side_map_text_predictions import _apply


class TestApplyEventSideMapTextPredictions(unittest.TestCase):
    def test_apply_unreliable_blocks(self):
        row = {"prediction": {"gate": "TRADE", "side": "LONG"}}
        self.assertEqual(_apply(row, "UNRELIABLE")["prediction"]["gate"], "NO_TRADE")

    def test_apply_inverse_flips(self):
        row = {"prediction": {"gate": "TRADE", "side": "LONG"}}
        self.assertEqual(_apply(row, "INVERSE")["prediction"]["side"], "SHORT")


if __name__ == "__main__":
    unittest.main()
