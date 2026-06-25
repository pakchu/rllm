from training.monthly_selector_inversion_audit import _corr, _month_row


def test_corr_returns_negative_for_inverse_series():
    assert _corr([1, 2, 3], [3, 2, 1]) < -0.99


def test_month_row_extracts_validation_and_eval_metrics():
    row = _month_row(
        "r.json",
        {
            "month": "2026-04",
            "status": "TRADED",
            "selected": {"target": "t", "threshold": 1, "validation_passed": True, "backtest": {"sim": {"cagr_pct": 10, "strict_mdd_pct": 5, "cagr_to_strict_mdd": 2, "trade_entries": 3}, "trade_stats": {"p_value_mean_ret_approx": 0.1, "t_stat_like": 2}}},
            "eval": {"backtest": {"sim": {"cagr_pct": -5, "strict_mdd_pct": 4, "cagr_to_strict_mdd": -1, "trade_entries": 2}, "trade_stats": {"p_value_mean_ret_approx": 0.2}}},
        },
    )
    assert row["val_cagr_pct"] == 10
    assert row["eval_cagr_pct"] == -5
