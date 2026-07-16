from __future__ import annotations

import pandas as pd
import pytest

import training.export_causal_residual_expert_switcher_2026_sources as exporter


def market_frame(start: str, periods: int, symbol: str = "ETHUSDT") -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="5min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "quote_asset_volume": 100.0,
            "number_of_trades": 1,
            "taker_buy_base": 0.5,
            "taker_buy_quote": 50.0,
            "tic": symbol,
            "day": dates.dayofweek,
        }
    )


def test_combine_market_prefix_deduplicates_boundary_and_requires_exact_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exporter, "START", pd.Timestamp("2026-06-01 00:00"))
    monkeypatch.setattr(exporter, "END", pd.Timestamp("2026-06-01 00:15"))
    base = market_frame("2026-06-01 00:00", 2)
    june = market_frame("2026-06-01 00:05", 2)
    combined = exporter.combine_market_prefix(base, june, "ETHUSDT")
    assert list(combined["date"]) == list(pd.date_range("2026-06-01", periods=3, freq="5min"))


def test_combine_market_prefix_rejects_missing_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exporter, "START", pd.Timestamp("2026-06-01 00:00"))
    monkeypatch.setattr(exporter, "END", pd.Timestamp("2026-06-01 00:15"))
    with pytest.raises(ValueError, match="market grid mismatch"):
        exporter.combine_market_prefix(
            market_frame("2026-06-01 00:00", 1),
            market_frame("2026-06-01 00:10", 1),
            "ETHUSDT",
        )


def test_run_stops_before_sources_when_protocol_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exporter, "canonical_hash", lambda value: "drift")
    monkeypatch.setattr(
        exporter,
        "_market_inputs",
        lambda *args: pytest.fail("source boundary must not be reached"),
    )
    with pytest.raises(RuntimeError, match="protocol drifted"):
        exporter.run()
