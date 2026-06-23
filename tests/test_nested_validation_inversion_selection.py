import unittest

from training.nested_validation_inversion_selection import _apply_validation_action_gate, _invert_prediction


class TestNestedValidationInversionSelection(unittest.TestCase):
    def test_invert_prediction_flips_trade_side_only(self):
        self.assertEqual(_invert_prediction({"gate": "TRADE", "side": "LONG"})["side"], "SHORT")
        self.assertEqual(_invert_prediction({"gate": "TRADE", "side": "SHORT"})["side"], "LONG")
        self.assertEqual(_invert_prediction({"gate": "NO_TRADE", "side": "NONE"}).get("side"), "NONE")

    def test_validation_action_gate_uses_threshold(self):
        rows = [
            {"date": "2026-01-01T00:00:00", "prediction": {"gate": "TRADE", "side": "LONG"}},
            {"date": "2026-02-01T00:00:00", "prediction": {"gate": "TRADE", "side": "SHORT"}},
        ]
        out, counts = _apply_validation_action_gate(rows, {"2026-01": -1.0, "2026-02": 1.0}, 0.0, "invert", "pass")
        self.assertEqual(out[0]["prediction"]["side"], "SHORT")
        self.assertEqual(out[1]["prediction"]["side"], "SHORT")
        self.assertEqual(counts["inverted_trades"], 1)
        self.assertEqual(counts["passed_trades"], 1)


if __name__ == "__main__":
    unittest.main()
