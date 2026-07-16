from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import training.build_causal_residual_expert_switcher_2026_support as builder


def candidate(signal: str) -> dict[str, object]:
    return {
        "signal_time": pd.Timestamp(signal),
        "residual_horizon_hours": 12,
        "long_symbol": "ADAUSDT",
        "short_symbol": "ETHUSDT",
        "long_weight": 0.4,
        "short_weight_abs": 0.6,
        "long_beta": 1.5,
        "short_beta": 1.0,
        "loser_residual_z": -2.0,
        "winner_residual_z": 2.0,
        "loser_flow_z": 0.0,
        "winner_flow_z": 0.0,
        "exhaustion_score": 2.0,
        "eligible": True,
    }


def test_reserve_base_clock_is_nonoverlapping_and_canonical_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "CONFIRMATION_START", pd.Timestamp("2026-01-01"))
    monkeypatch.setattr(builder, "END", pd.Timestamp("2026-02-01"))
    monkeypatch.setattr(builder, "RISK_LOOKBACK_BARS", 2)
    candidates = pd.DataFrame(
        [
            candidate("2026-01-01 01:00"),
            candidate("2026-01-01 02:00"),
            candidate("2026-01-01 14:00"),
        ]
    )
    quality_index = pd.date_range("2025-12-31", "2026-02-01", freq="1h")
    quality = pd.Series(True, index=quality_index)
    five_dates = pd.date_range("2025-12-31", "2026-02-01", freq="5min", inclusive="left")
    ranges = {symbol: np.full(len(five_dates), 0.01) for symbol in builder.SYMBOLS}
    clock = builder.reserve_base_clock(candidates, quality, ranges, five_dates)
    assert len(clock) == 2
    assert clock.loc[0, "continuation_long_symbol"] == "ETHUSDT"
    assert clock.loc[0, "continuation_short_symbol"] == "ADAUSDT"
    assert clock.loc[0, "continuation_long_weight_gross1"] == pytest.approx(0.6)
    assert clock.loc[0, "continuation_short_weight_abs_gross1"] == pytest.approx(0.4)
    builder.assert_clock_contract(clock)


def test_clock_contract_rejects_outcome_like_column() -> None:
    clock = pd.DataFrame(columns=[*builder.CLOCK_COLUMNS, "future_return"])
    clock.loc[0] = [None] * len(clock.columns)
    with pytest.raises(RuntimeError, match="outcome-like"):
        builder.assert_clock_contract(clock)


def test_support_loader_rejects_source_hash_before_market_access(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"manifest_hash":"drift"}')
    with pytest.raises(RuntimeError, match="source manifest hash changed"):
        builder._load_source_manifest(str(path))


def test_range_risk_excludes_signal_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder, "RISK_LOOKBACK_BARS", 2)
    dates = pd.date_range("2026-01-01", periods=5, freq="5min")
    ranges = {
        symbol: np.array([0.01, 0.01, 9.0, 9.0, 9.0])
        for symbol in builder.SYMBOLS
    }
    value = builder._range_risk(
        ranges, dates, pd.Timestamp("2026-01-01 00:10"), "ETHUSDT", "ADAUSDT"
    )
    assert value == pytest.approx(0.01)
