import unittest

from training.path_shape_val_token_veto_tte import ValTokenVetoTTECfg, _prediction_rows, _semantic_unit, _veto_units


class TestPathShapeValTokenVetoTTE(unittest.TestCase):
    def test_prediction_rows_veto_matching_tokens(self):
        cfg = ValTokenVetoTTECfg("", "", "", "", "", "")
        rows = [{"date": "d", "signal_pos": 1, "prompt": 'Past-only analyzer summary: {"regime":"UPTREND"}'}]
        preds = [{"label": "LONG", "prob": 0.9, "margin": 0.5}]
        out = _prediction_rows(rows, preds, prob_th=0.0, margin_th=0.0, side_mode="normal", veto_tokens={"regime=UPTREND"}, hold_bars=144, cfg=cfg)
        self.assertEqual(out[0]["prediction"]["gate"], "NO_TRADE")
        self.assertTrue(out[0]["vetoed"])

    def test_semantic_unit_drops_window_but_keeps_bucket(self):
        self.assertEqual(_semantic_unit("aug.micro.w12.return=-2..-0.75pct"), "aug.micro.return=-2..-0.75pct")
        self.assertEqual(_semantic_unit("aug.pa.w576.to_high=NEAR"), "aug.pa.to_high=NEAR")

    def test_veto_units_can_exclude_recent_tokens(self):
        cfg = ValTokenVetoTTECfg("", "", "", "", "", "", veto_unit_mode="semantic", exclude_veto_regex="^recent=")
        row = {"prompt": 'Past-only analyzer summary: {"recent_bar_sequence":["UP|NORMAL|NORMAL|BALANCED"],"augmented_micro_path_tokens":["micro.w12.return=flat"]}'}
        units = _veto_units(row, cfg)
        self.assertIn("aug.micro.return=flat", units)
        self.assertFalse(any(x.startswith("recent=") for x in units))


if __name__ == "__main__":
    unittest.main()
