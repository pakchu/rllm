import unittest

from training.event_side_map_reliability_dataset import _label, _score_tokens, _split, _trade_side


class TestEventSideMapReliabilityDataset(unittest.TestCase):
    def test_label(self):
        self.assertEqual(_label({"actual_long_pct": 1, "actual_short_pct": -1}, "LONG", 0)[0], "normal")
        self.assertEqual(_label({"actual_long_pct": -1, "actual_short_pct": 1}, "LONG", 0)[0], "inverse")
        self.assertEqual(_label({"actual_long_pct": -1, "actual_short_pct": -0.5}, "LONG", 0)[0], "unreliable")

    def test_trade_side(self):
        self.assertEqual(_trade_side({"prediction": {"gate": "TRADE", "side": "SHORT"}}), "SHORT")
        self.assertEqual(_trade_side({"prediction": {"gate": "NO_TRADE"}}), "NONE")

    def test_split_and_score_tokens(self):
        self.assertEqual(_split("2025-01", "2024-12", "2025-12"), "val")
        toks = _score_tokens({"score_long": 0.2, "score_short": 0.0, "score_wait": 0.01})
        self.assertIn("score_side_gap", toks)


if __name__ == "__main__":
    unittest.main()
