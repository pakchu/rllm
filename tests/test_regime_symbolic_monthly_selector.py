from pathlib import Path

from training.regime_symbolic_monthly_selector import (
    _history_before,
    _month_starts,
    _slice_rows,
    _write_no_trade_month,
    _compact_policy_result,
    _safe_unlink,
    _executed_month_stats,
)


def test_month_starts_half_open_range():
    months = _month_starts("2026-01-15", "2026-04-01")
    assert [str(m.date()) for m in months] == ["2026-01-01", "2026-02-01", "2026-03-01"]


def test_slice_rows_uses_half_open_dates():
    rows = [
        {"date": "2025-12-31 23:55:00"},
        {"date": "2026-01-01 00:00:00"},
        {"date": "2026-02-01 00:00:00"},
    ]
    assert _history_before(rows, "2026-01-01") == [rows[0]]
    assert _slice_rows(rows, "2026-01-01", "2026-02-01") == [rows[1]]


def test_write_no_trade_month_deduplicates_signal_rows(tmp_path: Path):
    path = tmp_path / "no_trade.jsonl"
    summary = _write_no_trade_month(
        [
            {"date": "2026-01-01 00:00:00", "signal_pos": 1},
            {"date": "2026-01-01 00:00:00", "signal_pos": 1},
            {"date": "2026-01-02 00:00:00", "signal_pos": 2},
        ],
        path,
        "failed",
    )
    assert summary["rows"] == 2
    assert summary["trade_signals"] == 0
    assert path.read_text().count("NO_TRADE") == 2


def test_compact_policy_result_drops_large_summary_and_keeps_evidence():
    row = {
        "target": "tail_risk",
        "threshold": -0.1,
        "summary": {"large": "drop"},
        "backtest": {"sim": {"trade_entries": 1}},
        "predictions": "pred.jsonl",
        "selection_score": 1.2,
        "validation_passed": False,
        "validation_reject_reasons": ["bad"],
    }
    compact = _compact_policy_result(row)
    assert "summary" not in compact
    assert compact["backtest"] == row["backtest"]
    assert compact["validation_reject_reasons"] == ["bad"]


def test_safe_unlink_is_idempotent(tmp_path: Path):
    path = tmp_path / "x.txt"
    path.write_text("x")
    _safe_unlink(path)
    _safe_unlink(path)
    assert not path.exists()


def test_executed_month_stats_summarizes_trade_returns():
    stats = _executed_month_stats([
        {"date": "2026-01-01", "trade_ret_pct": 1.0},
        {"date": "2026-01-02", "trade_ret_pct": -0.25},
        {"date": "2026-02-01", "trade_ret_pct": -2.0},
    ])
    assert stats["positive_months"] == 1
    assert stats["worst_month_ret_pct"] == -2.0
    assert stats["months"] == [
        {"month": "2026-01", "trades": 2, "sum_trade_ret_pct": 0.75},
        {"month": "2026-02", "trades": 1, "sum_trade_ret_pct": -2.0},
    ]
