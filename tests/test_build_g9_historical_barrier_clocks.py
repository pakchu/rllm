from __future__ import annotations

import json

import pandas as pd
import pytest

from training.build_g9_historical_barrier_clocks import (
    _exit_kind,
    _exit_price,
    _market_coverage,
    _trade_records,
    _write_jsonl,
)
from training.search_inventory_purge_reclaim_alpha import Trade


def _market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="5min"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "high": [101.0] * 8,
            "low": [99.0] * 8,
            "close": [100.5] * 8,
        }
    )


def _trade(*, side: int = 1, gross_return: float = 0.04, exit_position: int = 3) -> Trade:
    return Trade(
        signal_position=0,
        entry_position=1,
        exit_position=exit_position,
        side=side,
        gross_return=gross_return,
        price_factor=1.0 + 0.5 * gross_return,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=1.02,
        adverse_price_factor=0.99,
        entry_date="2024-01-01 00:05:00",
    )


def test_market_coverage_uses_last_bar_plus_one_5m_as_end_exclusive() -> None:
    coverage = _market_coverage(_market())
    assert coverage["first_bar"] == "2024-01-01 00:00:00"
    assert coverage["last_bar"] == "2024-01-01 00:35:00"
    assert coverage["data_end_exclusive"] == "2024-01-01 00:40:00"
    assert coverage["complete_5m_grid"] is True


def test_exit_kind_and_exit_price_distinguish_open_cap_take_and_stop() -> None:
    bars = _market()
    cap = _trade(gross_return=104.0 / 101.0 - 1.0, exit_position=4)
    assert _exit_kind(cap, 3) == "open"
    assert _exit_price(bars, cap, hold_bars=3, take_bps=400, stop_bps=250) == pytest.approx(104.0)
    long_take = _trade(side=1, gross_return=0.04, exit_position=2)
    assert _exit_kind(long_take, 3) == "barrier"
    assert _exit_price(bars, long_take, hold_bars=3, take_bps=400, stop_bps=250) == pytest.approx(101.0 * 1.04)
    short_stop = _trade(side=-1, gross_return=-0.025, exit_position=2)
    assert _exit_price(bars, short_stop, hold_bars=3, take_bps=400, stop_bps=250) == pytest.approx(101.0 * 1.025)


def test_trade_records_keep_numeric_side_and_metadata() -> None:
    rows = _trade_records(
        _market(),
        [_trade(side=-1, gross_return=0.04, exit_position=2)],
        sleeve="x",
        window="w",
        hold_bars=3,
        take_bps=400,
        stop_bps=250,
    )
    assert rows[0]["side"] == -1
    assert rows[0]["side_label"] == "short"
    assert rows[0]["exit_kind"] == "barrier"
    assert rows[0]["hold_bars"] == 3
    assert rows[0]["take_bps"] == 400
    assert rows[0]["stop_bps"] == 250


def test_write_jsonl_hashes_exact_rows(tmp_path) -> None:
    artifact = _write_jsonl(tmp_path / "x.jsonl", [{"b": 2, "a": 1}, {"a": 3}])
    assert artifact["rows"] == 2
    lines = (tmp_path / "x.jsonl").read_text().splitlines()
    assert json.loads(lines[0]) == {"a": 1, "b": 2}
    assert len(artifact["sha256"]) == 64
