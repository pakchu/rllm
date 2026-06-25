from pathlib import Path

from training.regime_symbolic_monthly_selector import (
    _history_before,
    _month_starts,
    _slice_rows,
    _write_no_trade_month,
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
