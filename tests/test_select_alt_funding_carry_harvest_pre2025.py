from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import training.select_alt_funding_carry_harvest_pre2025 as selector
from training.select_alt_funding_carry_harvest_pre2025 import MarketBundle, simulate


def synthetic_bundle(
    long_values: list[float],
    short_values: list[float],
    funding: dict[str, list[tuple[pd.Timestamp, float]]] | None = None,
) -> MarketBundle:
    dates = pd.date_range("2023-01-01", periods=len(long_values), freq="5min")
    market = {}
    for symbol, values in (("ADAUSDT", long_values), ("ETHUSDT", short_values)):
        array = np.asarray(values, dtype=float)
        market[symbol] = {
            "open": array.copy(),
            "high": array * 1.01,
            "low": array * 0.99,
            "close": array.copy(),
        }
    # Other symbols are required by aggregate net-quantity marking.
    for symbol in selector.SYMBOLS:
        if symbol not in market:
            array = np.full(len(long_values), 100.0)
            market[symbol] = {"open": array.copy(), "high": array.copy(), "low": array.copy(), "close": array.copy()}
    funding_frames = {symbol: pd.DataFrame(columns=["event_time", "funding_rate"]) for symbol in selector.SYMBOLS}
    if funding:
        for symbol, rows in funding.items():
            funding_frames[symbol] = pd.DataFrame(rows, columns=["event_time", "funding_rate"])
    return MarketBundle(dates, market, funding_frames)


def clock(
    entry: str = "2023-01-01 00:05",
    exit_time: str = "2023-01-01 00:20",
    sleeve_gross: float = 0.25,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "policy_id": "AFCH01",
        "signal_time": pd.Timestamp(entry) - pd.Timedelta(minutes=5),
        "feature_available_time": pd.Timestamp(entry) - pd.Timedelta(minutes=5),
        "entry_time": pd.Timestamp(entry),
        "exit_time": pd.Timestamp(exit_time),
        "hold_days": 28,
        "sleeve_gross": sleeve_gross,
        "long_symbol": "ADAUSDT",
        "short_symbol": "ETHUSDT",
        "long_weight_norm": 0.5,
        "short_weight_norm": 0.5,
        "long_beta": 1.0,
        "short_beta": 1.0,
        "long_trailing_funding_sum": -0.01,
        "short_trailing_funding_sum": 0.01,
        "projected_28d_carry": 0.01,
    }])


def test_pair_price_profit_is_scaled_by_sleeve_gross() -> None:
    bundle = synthetic_bundle([100, 100, 102, 104, 104], [100, 100, 99, 98, 98])
    stats = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=0)
    assert stats["absolute_return_pct"] == pytest.approx(0.75)
    assert stats["price_pnl_pct_initial"] == pytest.approx(0.75)


def test_exact_funding_cash_is_the_direct_payoff() -> None:
    funding = {
        "ADAUSDT": [(pd.Timestamp("2023-01-01 00:10"), -0.01)],
        "ETHUSDT": [(pd.Timestamp("2023-01-01 00:10"), 0.01)],
    }
    bundle = synthetic_bundle([100] * 5, [100] * 5, funding)
    for symbol in ("ADAUSDT", "ETHUSDT"):
        bundle.market[symbol]["high"][:] = 100.0
        bundle.market[symbol]["low"][:] = 100.0
    stats = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=0)
    assert stats["funding_cash_pct_initial"] == pytest.approx(0.25)
    assert stats["absolute_return_pct"] == pytest.approx(0.25)
    removed = simulate(
        bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=0, include_funding=False
    )
    assert removed["absolute_return_pct"] == pytest.approx(0.0)


def test_funding_at_entry_is_excluded_and_at_exit_is_included() -> None:
    funding = {
        "ETHUSDT": [
            (pd.Timestamp("2023-01-01 00:05"), 0.01),
            (pd.Timestamp("2023-01-01 00:20"), 0.01),
        ]
    }
    bundle = synthetic_bundle([100] * 5, [100] * 5, funding)
    stats = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=0)
    assert stats["funding_cash_pct_initial"] == pytest.approx(0.125)


def test_strict_mdd_uses_aggregate_favorable_before_adverse_path() -> None:
    bundle = synthetic_bundle([100] * 5, [100] * 5)
    stats = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=6)
    assert stats["strict_mdd_pct"] > stats["close_mdd_pct"]
    assert stats["strict_mdd_pct"] > 0.25


def test_overlapping_sleeves_are_aggregated_in_one_cash_ledger() -> None:
    bundle = synthetic_bundle([100] * 8, [100] * 8)
    clocks = pd.concat([
        clock(entry="2023-01-01 00:05", exit_time="2023-01-01 00:25"),
        clock(entry="2023-01-01 00:10", exit_time="2023-01-01 00:30"),
    ], ignore_index=True)
    stats = simulate(bundle, clocks, start="2023-01-01", end="2023-01-02", cost_bp=6)
    assert stats["sleeves"] == 2
    assert stats["transaction_cost_pct_initial"] > 0.05


def test_simulation_refuses_2025_access() -> None:
    bundle = synthetic_bundle([100] * 5, [100] * 5)
    with pytest.raises(ValueError, match="escaped 2023-2024"):
        simulate(bundle, clock(), start="2023-01-01", end="2025-01-02", cost_bp=0)


def test_run_refuses_bad_support_before_attestation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        selector,
        "_manifest",
        lambda *args, **kwargs: {"clock_sha256": "wrong", "support": {"passes_support": False}},
    )
    monkeypatch.setattr(selector, "_git_attestation", lambda: pytest.fail("must stop before outcomes"))
    with pytest.raises(RuntimeError, match="support freeze is not approved"):
        selector.run()


def test_frozen_clock_is_live_and_contains_no_2026_exit() -> None:
    frozen = selector.load_clock()
    assert len(frozen) == 127
    assert frozen["exit_time"].max() < pd.Timestamp("2026-01-01")
