import numpy as np
import pandas as pd

from training import build_pposm_pre2024_intrahour_premium_action_source as build


def _premium_block(end, minutes=1440):
    dates = pd.date_range(pd.Timestamp(end, tz="UTC") - pd.Timedelta(minutes=minutes), periods=minutes, freq="1min")
    close = np.linspace(-0.001, 0.001, minutes)
    return pd.DataFrame({"date": dates, "open": close - 1e-6, "high": close + 2e-6, "low": close - 2e-6, "close": close})


def test_feature_row_uses_strictly_prior_exact_grid():
    end = "2023-01-01T00:00:00Z"
    p = _premium_block(end)
    row = build.feature_row(pd.Timestamp(end), p)
    assert row["source_valid"] is True
    assert row["p60m_close_last_minus_first"] > 0
    assert p["date"].max() == pd.Timestamp(end) - pd.Timedelta(minutes=1)


def test_feature_row_rejects_missing_minute():
    end = "2023-01-01T00:00:00Z"
    p = _premium_block(end).iloc[:-1]
    row = build.feature_row(pd.Timestamp(end), p)
    assert row["source_valid"] is False
    assert "grid" in row["invalid_reason"]


def test_auc_handles_ties_and_perfect_order():
    assert build.auc_score(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert build.auc_score(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5


def test_expanding_eval_does_not_use_future_rows():
    n = 140
    years = [2020] * 90 + [2021] * 50
    times = [pd.Timestamp(f"{y}-06-01T00:00:00Z") + pd.Timedelta(hours=i) for i, y in enumerate(years)]
    x = np.r_[np.linspace(-2, 2, 90), np.linspace(-1, 1, 50)]
    frame = pd.DataFrame({"decision_time": times, "source_valid": True, "SKIP": (x > 0).astype(int), "TP12": (x > 0).astype(int), "p60m_close_mean": x})
    old = build.prereg.VALIDATION_YEARS
    build.prereg.VALIDATION_YEARS = [2021]
    try:
        ev = build.evaluate_candidate(frame, "SKIP", ["p60m_close_mean"])
    finally:
        build.prereg.VALIDATION_YEARS = old
    assert ev["folds"][0]["train_signals"] == 90
    assert ev["folds"][0]["validation_signals"] == 50
    assert ev["pooled"]["auc"] > 0.9
