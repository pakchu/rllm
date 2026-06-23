import unittest

from training.nested_score_geometry_transform_selection import _apply_transform, _resolve_action, _side_gap


class TestNestedScoreGeometryTransformSelection(unittest.TestCase):
    def test_resolve_gap_actions(self):
        self.assertEqual(_resolve_action("invert_low_gap", 0.01, 0.05), "invert")
        self.assertEqual(_resolve_action("invert_low_gap", 0.10, 0.05), "pass")
        self.assertEqual(_resolve_action("block_high_gap", 0.10, 0.05), "block")

    def test_apply_transform_inverts_below_score(self):
        rows = [{"date": "2026-01-01", "score_long": 0.1, "score_short": 0.0, "prediction": {"gate": "TRADE", "side": "LONG"}}]
        out, counts = _apply_transform(rows, {"2026-01": -501.0}, -500.0, "invert", "pass", 0.0)
        self.assertEqual(out[0]["prediction"]["side"], "SHORT")
        self.assertEqual(counts["inverted_trades"], 1)

    def test_side_gap_abs_long_short(self):
        self.assertAlmostEqual(_side_gap({"score_long": 0.1, "score_short": -0.2}), 0.3)


if __name__ == "__main__":
    unittest.main()
