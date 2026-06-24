import json
import tempfile
import unittest
from pathlib import Path

from training.event_candidate_pairwise_walkforward import EventCandidatePairwiseWalkForwardCfg, _feature_filter_values, _max_feature_names, _side_scales_from_validation, _side_trade_stats, _allowed_sides_from_validation, _no_trade_predictions, make_folds


class TestEventCandidatePairwiseWalkForward(unittest.TestCase):
    def test_make_folds_uses_half_open_calendar_windows(self):
        cfg = EventCandidatePairwiseWalkForwardCfg(input_jsonl="x", market_csv="m", output="o", fit_months=12, val_months=6, test_months=6, step_months=6)
        folds = make_folds("2022-01-15 00:00:00", "2024-02-01 00:00:00", cfg)
        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[0].fit_start, "2022-01-01 00:00:00")
        self.assertEqual(folds[0].fit_end, "2023-01-01 00:00:00")
        self.assertEqual(folds[0].val_start, "2023-01-01 00:00:00")
        self.assertEqual(folds[0].test_start, "2023-07-01 00:00:00")
        self.assertEqual(folds[1].fit_start, "2022-07-01 00:00:00")

    def test_no_trade_predictions_emit_one_row_per_signal(self):
        rows = [
            {"date": "2024-01-01 00:00:00", "signal_pos": 1, "side": "LONG"},
            {"date": "2024-01-01 00:00:00", "signal_pos": 1, "side": "SHORT"},
            {"date": "2024-01-02 00:00:00", "signal_pos": 2, "side": "LONG"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pred.jsonl"
            summary = _no_trade_predictions(rows, str(out), reason="unit")
            loaded = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(summary["rows"], 2)
        self.assertEqual([r["prediction"]["gate"] for r in loaded], ["NO_TRADE", "NO_TRADE"])
        self.assertEqual([r["signal_pos"] for r in loaded], [1, 2])

    def test_side_allowlist_uses_validation_side_stats(self):
        stats = _side_trade_stats([
            {"side": "LONG", "trade_ret_pct": 1.0},
            {"side": "LONG", "trade_ret_pct": -0.2},
            {"side": "SHORT", "trade_ret_pct": -1.0},
        ])
        cfg = EventCandidatePairwiseWalkForwardCfg(
            input_jsonl="x", market_csv="m", output="o", side_min_val_trades=2, side_min_val_mean_ret_pct=0.0
        )
        self.assertEqual(_allowed_sides_from_validation(stats, cfg), {"LONG"})

    def test_side_scales_use_validation_mean_strength(self):
        stats = {"LONG": {"n": 3.0, "mean_trade_ret_pct": 0.5}, "SHORT": {"n": 3.0, "mean_trade_ret_pct": -0.1}}
        cfg = EventCandidatePairwiseWalkForwardCfg(
            input_jsonl="x", market_csv="m", output="o", side_min_val_trades=2, side_scale_val_mean_ret_pct=1.0
        )
        self.assertEqual(_side_scales_from_validation(stats, cfg), {"LONG": 0.5, "SHORT": 0.0})

    def test_max_feature_names_prefers_multi_arg(self):
        cfg = EventCandidatePairwiseWalkForwardCfg(input_jsonl="x", market_csv="m", output="o", max_feature_name="old", max_feature_names="a,b,a")
        self.assertEqual(_max_feature_names(cfg), ["a", "b"])

    def test_feature_filter_values_uses_selected_quantiles(self):
        import numpy as np
        values = {"a": np.asarray([1.0, 3.0, 5.0]), "b": np.asarray([2.0, 4.0, 6.0])}
        self.assertEqual(_feature_filter_values(values, {"a": 0.5, "b": None}), {"a": 3.0})


if __name__ == "__main__":
    unittest.main()
