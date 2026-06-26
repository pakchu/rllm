import unittest

from training.path_shape_val_token_veto_tte import _prediction_rows


class TestPathShapeValTokenVetoTTE(unittest.TestCase):
    def test_prediction_rows_veto_matching_tokens(self):
        rows = [{"date": "d", "signal_pos": 1, "prompt": 'Past-only analyzer summary: {"regime":"UPTREND"}'}]
        preds = [{"label": "LONG", "prob": 0.9, "margin": 0.5}]
        out = _prediction_rows(rows, preds, prob_th=0.0, margin_th=0.0, side_mode="normal", veto_tokens={"regime=UPTREND"}, hold_bars=144)
        self.assertEqual(out[0]["prediction"]["gate"], "NO_TRADE")
        self.assertTrue(out[0]["vetoed"])


if __name__ == "__main__":
    unittest.main()
