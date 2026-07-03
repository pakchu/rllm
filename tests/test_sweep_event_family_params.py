from training.sweep_event_family_params import _rank_metric, _sort_dedup_rows


def test_rank_metric_rejects_eval_like_unprofitable_or_small_samples():
    assert _rank_metric({"trade_entries": 3, "cagr_pct": 100, "cagr_to_strict_mdd": 10}, min_trades=10) < -1e8
    assert _rank_metric({"trade_entries": 30, "cagr_pct": -1, "cagr_to_strict_mdd": 10}, min_trades=10) < -1e8
    assert _rank_metric({"trade_entries": 30, "cagr_pct": 12, "cagr_to_strict_mdd": 2, "p_value_mean_ret_approx": 0.1}, min_trades=10) > 2


def test_sort_dedup_rows_keeps_highest_priority_same_signal_side():
    rows = [
        {"signal_date": "2025-01-01T00:00:00", "side": "LONG", "score_mean": 1.0, "family": "a"},
        {"signal_date": "2025-01-01T00:00:00", "side": "LONG", "score_mean": 2.0, "family": "b"},
        {"signal_date": "2025-01-01T00:00:00", "side": "SHORT", "score_mean": 1.5, "family": "c"},
    ]
    out = _sort_dedup_rows(rows)
    assert len(out) == 2
    assert any(row["family"] == "b" for row in out)
    assert any(row["family"] == "c" for row in out)
