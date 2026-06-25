import numpy as np

from training.prediction_trade_feature_audit import _corr, _feature_row


def test_corr_handles_constant_arrays():
    assert _corr(np.ones(3), np.arange(3, dtype=float)) == 0.0


def test_feature_row_reports_quantile_spread():
    x = np.arange(8, dtype=float)
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    side = np.ones(8, dtype=float)
    row = _feature_row("x", x, y, side)
    assert row["feature"] == "x"
    assert row["high_minus_low_mean_trade_ret_pct"] < 0
    assert row["n"] == 8
