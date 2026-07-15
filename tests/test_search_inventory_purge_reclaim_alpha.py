from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.search_inventory_purge_reclaim_alpha import (
    Config,
    ExecutionEngine,
    Trade,
    _apply_gate,
    _context_value,
    _state_admits,
    equity_stats,
)


def _cfg(**overrides: object) -> Config:
    values: dict[str, object] = {
        "input_csv": "market.csv",
        "metrics_csv": "metrics.csv",
        "funding_csv": "funding.csv",
        "output": "output.json",
        "manifest_output": "manifest.json",
    }
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


def _trade(signal: int, side: int, *, favorable: float = 1.0, adverse: float = 1.0) -> Trade:
    return Trade(
        signal_position=signal,
        entry_position=signal + 1,
        exit_position=signal + 2,
        side=side,
        gross_return=0.0,
        price_factor=1.0,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=favorable,
        adverse_price_factor=adverse,
        entry_date="2023-01-01 00:05:00",
    )


def test_strict_mdd_uses_favorable_then_adverse_path() -> None:
    stats = equity_stats(
        [_trade(0, 1, favorable=1.10, adverse=0.90)],
        start="2023-01-01",
        end="2024-01-01",
        cfg=_cfg(fee_rate=0.0, slippage_rate=0.0),
    )
    assert stats["absolute_return_pct"] == pytest.approx(0.0)
    assert stats["strict_mdd_pct"] == pytest.approx((1.0 - 0.90 / 1.10) * 100.0)


def test_short_context_is_side_aligned() -> None:
    raw = {"smart_retail_align": np.asarray([0.0, -0.8, 0.8])}
    threshold = {"0.6": 0.25}
    supporting_short = _trade(1, -1)
    opposing_short = _trade(2, -1)
    assert _context_value("smart_retail_align", raw, supporting_short) == pytest.approx(0.8)
    assert _state_admits("smart_retail_align", "high60", threshold, raw, supporting_short)
    assert not _state_admits("smart_retail_align", "high60", threshold, raw, opposing_short)


def test_gate_only_removes_short_trades_and_never_reschedules() -> None:
    long_trade = _trade(0, 1)
    accepted_short = _trade(1, -1)
    rejected_short = _trade(2, -1)
    schedules = {"fit": [long_trade, accepted_short, rejected_short]}
    raw = {"smart_retail_align": np.asarray([0.0, -0.8, 0.8])}
    gated = _apply_gate(
        schedules,
        raw,
        {"states": ["smart_retail_align:high60"], "target": "short"},
        {"smart_retail_align": {"0.6": 0.25}},
    )
    assert gated["fit"] == [long_trade, accepted_short]
    assert all(trade in schedules["fit"] for trade in gated["fit"])
    assert [(trade.entry_position, trade.exit_position) for trade in gated["fit"]] == [(1, 2), (2, 3)]


def test_next_open_entry_and_same_bar_stop_precedes_take() -> None:
    dates = pd.date_range("2023-01-01", periods=6, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 6,
            "high": [100.0, 103.0, 100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 98.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    funding = pd.DataFrame({"date": pd.to_datetime([]), "funding_rate": np.asarray([], dtype=float)})
    engine = ExecutionEngine(market, funding, _cfg(leverage=0.5))
    trade = engine.trade_at(0, 1, 2, 200, 100)
    assert trade is not None
    assert trade.entry_position == 1
    assert trade.exit_position == 1
    assert trade.gross_return == pytest.approx(-0.01)
    assert trade.adverse_price_factor == pytest.approx(1.0 - 0.5 * 0.01)
    assert trade.favorable_price_factor == pytest.approx(1.0 + 0.5 * 0.03)


def test_frozen_manifest_has_not_opened_oos() -> None:
    manifest = json.loads(Path("results/inventory_purge_reclaim_manifest_2026-07-15.json").read_text())
    assert manifest["protocol"]["oos_opened"] is False
    assert manifest["protocol"]["selection_cutoff"] == "2024-01-01"
    assert manifest["base_champion"]["horizon_bars"] == 48
    assert manifest["base_champion"]["reclaim_bars"] == 12
    assert manifest["gate_champion"]["states"] == ["smart_retail_align:high60"]
    assert manifest["gate_champion"]["target"] == "short"
    assert manifest["search_counts"] == {
        "base_tested": 3760,
        "base_stable_positive": 1,
        "gate_tested": 1248,
        "gate_stable_positive": 131,
        "gate_eligible": 1,
    }
