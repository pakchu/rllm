from training.economic_fold_stability_sweep import aggregate_score, score_fold_result, split_rows


def test_aggregate_score_counts_positive_folds():
    agg = aggregate_score([
        {"trades": 60, "ratio": 1.0, "cagr": 2.0, "mean_trade_ret_pct": 0.1, "positive": 1, "strong": 0},
        {"trades": 60, "ratio": -1.0, "cagr": -2.0, "mean_trade_ret_pct": -0.1, "positive": 0, "strong": 0},
    ])
    assert agg["positive_folds"] == 1
    assert agg["min_ratio"] == -1.0


def test_score_fold_requires_min_trades():
    result = {"sim": {"trade_entries": 10, "cagr_to_strict_mdd": 5.0, "cagr_pct": 10.0}, "trade_stats": {"mean_trade_ret_pct": 0.2}}
    assert score_fold_result(result, min_trades=50)["positive"] == 0


def test_split_rows_uses_chronological_boundaries():
    rows = [{"date": "2024-01-01 00:00:00"}, {"date": "2024-02-01 00:00:00"}]
    train, test = split_rows(rows, train_end="2024-01-15 00:00:00", test_start="2024-02-01 00:00:00", test_end="2024-02-28 00:00:00")
    assert len(train) == 1
    assert len(test) == 1
