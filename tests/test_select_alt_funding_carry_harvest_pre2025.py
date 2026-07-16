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


def test_pre2025_source_composition_uses_recent_rows_from_handoff() -> None:
    old_market = pd.DataFrame({
        "date": ["2023-06-01", "2024-06-01"],
        "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0],
        "tic": ["ADAUSDT", "ADAUSDT"],
    })
    recent_market = pd.DataFrame({
        "date": ["2024-06-01", "2025-06-01"],
        "open": [3.0, 4.0], "high": [3.0, 4.0], "low": [3.0, 4.0], "close": [3.0, 4.0],
        "tic": ["ADAUSDT", "ADAUSDT"],
    })
    market = selector.compose_pre2025_market(old_market, recent_market, "ADAUSDT")
    assert market["date"].tolist() == [pd.Timestamp("2023-06-01"), pd.Timestamp("2024-06-01")]
    assert market["open"].tolist() == [1.0, 3.0]

    def funding_frame(dates: list[str], rates: list[float]) -> pd.DataFrame:
        times = pd.to_datetime(dates)
        return pd.DataFrame({
            "symbol": ["ADAUSDT"] * len(times),
            "funding_time": (times.astype("int64") // 1_000_000).astype("int64"),
            "funding_rate": rates,
            "mark_price": [100.0] * len(times),
        })

    old_funding = funding_frame(["2023-06-01", "2024-06-01"], [0.001, 0.002])
    recent_funding = funding_frame(["2024-06-01", "2025-06-01"], [0.003, 0.004])
    funding = selector.compose_pre2025_funding(old_funding, recent_funding, "ADAUSDT")
    assert funding["event_time"].tolist() == [pd.Timestamp("2023-06-01"), pd.Timestamp("2024-06-01")]
    assert funding["funding_rate"].tolist() == [0.001, 0.003]


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


def test_recorded_funding_mark_is_used_before_causal_close_proxy() -> None:
    funding = {
        "ADAUSDT": [(pd.Timestamp("2023-01-01 00:10"), -0.01)],
        "ETHUSDT": [(pd.Timestamp("2023-01-01 00:10"), 0.01)],
    }
    bundle = synthetic_bundle([100] * 5, [100] * 5, funding)
    for symbol in ("ADAUSDT", "ETHUSDT"):
        bundle.funding[symbol]["mark_price"] = 200.0
        bundle.market[symbol]["high"][:] = 100.0
        bundle.market[symbol]["low"][:] = 100.0
    stats = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=0)
    assert stats["funding_cash_pct_initial"] == pytest.approx(0.5)
    assert stats["exact_mark_funding_applications"] == 2
    assert stats["proxy_mark_funding_applications"] == 0


def test_missing_funding_mark_uses_last_completed_close() -> None:
    funding = {"ETHUSDT": [(pd.Timestamp("2023-01-01 00:10"), 0.01)]}
    bundle = synthetic_bundle([100] * 5, [100, 200, 300, 400, 500], funding)
    bundle.market["ETHUSDT"]["open"][:] = 100.0
    bundle.market["ETHUSDT"]["high"][:] = np.maximum(
        bundle.market["ETHUSDT"]["open"], bundle.market["ETHUSDT"]["close"]
    )
    bundle.market["ETHUSDT"]["low"][:] = np.minimum(
        bundle.market["ETHUSDT"]["open"], bundle.market["ETHUSDT"]["close"]
    )
    stats = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=0)
    # At 00:10, only the 00:05 bar close is complete, so the proxy mark is 200.
    assert stats["funding_cash_pct_initial"] == pytest.approx(0.25)
    assert stats["exact_mark_funding_applications"] == 0
    assert stats["proxy_mark_funding_applications"] == 1


def test_frozen_causal_proxy_mark_is_used_and_reported_as_proxy() -> None:
    funding = {"ETHUSDT": [(pd.Timestamp("2023-01-01 00:10"), 0.01)]}
    bundle = synthetic_bundle([100] * 5, [100] * 5, funding)
    bundle.funding["ETHUSDT"]["mark_price"] = 200.0
    bundle.funding["ETHUSDT"]["mark_is_recorded"] = False
    stats = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=0)
    assert stats["funding_cash_pct_initial"] == pytest.approx(0.25)
    assert stats["exact_mark_funding_applications"] == 0
    assert stats["proxy_mark_funding_applications"] == 1


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


def test_strict_mdd_exact_favorable_before_adverse_path() -> None:
    bundle = synthetic_bundle([100] * 5, [100] * 5)
    for symbol in selector.SYMBOLS:
        for field in ("open", "high", "low", "close"):
            bundle.market[symbol][field][:] = 100.0
    bundle.market["ADAUSDT"]["high"][2] = 110.0
    bundle.market["ADAUSDT"]["low"][2] = 90.0
    bundle.market["ETHUSDT"]["high"][2] = 110.0
    bundle.market["ETHUSDT"]["low"][2] = 90.0
    stats = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=0)
    expected = (1.0 - 0.975 / 1.025) * 100.0
    assert stats["strict_mdd_pct"] == pytest.approx(expected)
    assert stats["close_mdd_pct"] == pytest.approx(0.0)


def test_strict_path_nets_opposite_symbol_quantities() -> None:
    bundle = synthetic_bundle([100] * 5, [100] * 5)
    for symbol in selector.SYMBOLS:
        for field in ("open", "high", "low", "close"):
            bundle.market[symbol][field][:] = 100.0
    bundle.market["ADAUSDT"]["high"][2] = 120.0
    bundle.market["ADAUSDT"]["low"][2] = 80.0
    bundle.market["ETHUSDT"]["high"][2] = 120.0
    bundle.market["ETHUSDT"]["low"][2] = 80.0
    opposite = clock()
    opposite[["long_symbol", "short_symbol"]] = opposite[["short_symbol", "long_symbol"]].to_numpy()
    clocks = pd.concat([clock(), opposite], ignore_index=True)
    stats = simulate(bundle, clocks, start="2023-01-01", end="2023-01-02", cost_bp=0)
    assert stats["strict_mdd_pct"] == pytest.approx(0.0)


def test_pre_entry_hwm_is_refreshed_before_overlapping_entry_cost() -> None:
    bundle = synthetic_bundle([100, 100, 100, 200, 200, 200, 200, 200], [100] * 8)
    for symbol in selector.SYMBOLS:
        bundle.market[symbol]["high"][:] = bundle.market[symbol]["open"]
        bundle.market[symbol]["low"][:] = bundle.market[symbol]["open"]
        bundle.market[symbol]["close"][:] = bundle.market[symbol]["open"]
    clocks = pd.concat([
        clock(entry="2023-01-01 00:05", exit_time="2023-01-01 00:30", sleeve_gross=0.01),
        clock(entry="2023-01-01 00:15", exit_time="2023-01-01 00:35", sleeve_gross=0.25),
    ], ignore_index=True)
    stats = simulate(bundle, clocks, start="2023-01-01", end="2023-01-02", cost_bp=6)
    assert stats["strict_mdd_pct"] == pytest.approx(0.03, abs=0.001)


def test_active_positive_funding_cannot_raise_hwm_at_partial_exit() -> None:
    funding = {
        "ADAUSDT": [(pd.Timestamp("2023-01-01 00:15"), -0.01)],
        "ETHUSDT": [(pd.Timestamp("2023-01-01 00:15"), 0.01)],
    }
    bundle = synthetic_bundle([100] * 6, [100] * 6, funding)
    for symbol in ("ADAUSDT", "ETHUSDT"):
        bundle.funding[symbol]["mark_price"] = 100.0
    for symbol in selector.SYMBOLS:
        for field in ("open", "high", "low", "close"):
            bundle.market[symbol][field][:] = 100.0
    bundle.market["ADAUSDT"]["low"][4] = 99.6
    bundle.market["ETHUSDT"]["high"][4] = 100.4
    clocks = pd.concat([
        clock(entry="2023-01-01 00:05", exit_time="2023-01-01 00:20"),
        clock(entry="2023-01-01 00:10", exit_time="2023-01-01 00:25"),
    ], ignore_index=True)
    stats = simulate(bundle, clocks, start="2023-01-01", end="2023-01-02", cost_bp=0)
    assert stats["funding_cash_pct_initial"] == pytest.approx(0.5)
    assert stats["strict_mdd_pct"] == pytest.approx(0.0)


def test_overlapping_sleeves_are_aggregated_in_one_cash_ledger() -> None:
    bundle = synthetic_bundle([100] * 8, [100] * 8)
    clocks = pd.concat([
        clock(entry="2023-01-01 00:05", exit_time="2023-01-01 00:25"),
        clock(entry="2023-01-01 00:10", exit_time="2023-01-01 00:30"),
    ], ignore_index=True)
    stats = simulate(bundle, clocks, start="2023-01-01", end="2023-01-02", cost_bp=6)
    assert stats["sleeves"] == 2
    assert stats["transaction_cost_pct_initial"] > 0.05


def test_entry_exit_cost_and_liquidation_are_exact() -> None:
    bundle = synthetic_bundle([100] * 5, [100] * 5)
    for symbol in selector.SYMBOLS:
        for field in ("open", "high", "low", "close"):
            bundle.market[symbol][field][:] = 100.0
    base = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=6)
    stress = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=10)
    assert base["transaction_cost_pct_initial"] == pytest.approx(0.03)
    assert base["absolute_return_pct"] == pytest.approx(-0.03)
    assert base["strict_mdd_pct"] == pytest.approx(0.03)
    assert stress["transaction_cost_pct_initial"] == pytest.approx(0.05)
    assert stress["absolute_return_pct"] == pytest.approx(-0.05)
    assert stress["strict_mdd_pct"] == pytest.approx(0.05)


def test_cagr_uses_full_declared_calendar_not_trade_span() -> None:
    bundle = synthetic_bundle([100, 100, 102, 104, 104], [100, 100, 99, 98, 98])
    stats = simulate(bundle, clock(), start="2023-01-01", end="2024-01-01", cost_bp=0)
    years = 365.0 / 365.25
    expected = (1.0075 ** (1.0 / years) - 1.0) * 100.0
    assert stats["absolute_return_pct"] == pytest.approx(0.75)
    assert stats["calendar_years"] == pytest.approx(years)
    assert stats["cagr_pct"] == pytest.approx(expected)


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


def test_run_attests_clean_code_before_loading_outcomes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    calls: list[str] = []
    def fake_manifest(path: str, expected: str) -> dict[str, object]:
        if path == selector.SUPPORT_MANIFEST:
            return {
                "clock_sha256": selector.EXPECTED_CLOCK_HASH,
                "support": {"passes_support": True},
            }
        if path == selector.FUNDING_MARK_MANIFEST:
            return {
                "outcomes_opened": False,
                "maximum_proxy_funding_cash_error_bp_notional": 0.1,
                "records": [
                    {
                        "symbol": symbol,
                        "events": 1095,
                        "maximum_proxy_funding_cash_error_bp_notional": 0.01,
                    }
                    for symbol in selector.SYMBOLS
                ],
            }
        raise AssertionError(path)

    monkeypatch.setattr(
        selector,
        "_manifest",
        fake_manifest,
    )
    monkeypatch.setattr(
        selector, "_git_attestation", lambda: calls.append("attestation") or {"head": "frozen"}
    )
    monkeypatch.setattr(selector, "load_bundle", lambda: calls.append("bundle") or object())
    monkeypatch.setattr(selector, "load_clock", lambda: calls.append("clock") or pd.DataFrame())
    monkeypatch.setattr(
        selector,
        "evaluate",
        lambda *args: calls.append("evaluate") or {"passes_2023_2024_selection": False},
    )
    monkeypatch.setattr(selector, "_markdown", lambda result: "safe\n")
    selector.run(str(tmp_path / "result.json"), str(tmp_path / "result.md"))
    assert calls == ["attestation", "bundle", "clock", "evaluate"]


def test_evaluate_never_opens_clock_rows_entering_2025(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = synthetic_bundle([100] * 1000, [100] * 1000)
    future = clock(entry="2025-01-01 00:05", exit_time="2025-01-01 00:20")
    clocks = pd.concat([clock(), future], ignore_index=True)
    monkeypatch.setattr(
        selector,
        "weekly_cluster_signflip",
        lambda *args, **kwargs: {"raw_p_value": 1.0, "weekly_clusters": 1},
    )
    evaluation = selector.evaluate(bundle, clocks)
    assert evaluation["stats"]["combined_2023_2024"]["sleeves"] == 1
    assert evaluation["stats"]["test_2024"]["sleeves"] == 0


def test_selection_gate_thresholds_are_mapped_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selector, "_transform_clock", lambda frozen, *args, **kwargs: frozen)

    window_lookup = {window: name for name, window in selector.WINDOWS.items()}

    def evaluate_case(failure: str | None) -> dict[str, object]:
        def fake_simulate(bundle: object, frozen: pd.DataFrame, *, start: str, end: str, **kwargs: object) -> dict[str, object]:
            name = window_lookup[(start, end)]
            stats: dict[str, object] = {
                "absolute_return_pct": 1.0,
                "cagr_pct": 3.0,
                "strict_mdd_pct": 1.0,
                "cagr_to_strict_mdd": 3.0,
                "sleeves": 40,
                "funding_cash_pct_initial": 1.0,
                "transaction_cost_pct_initial": 1.0,
                "funding_to_cost_ratio": 1.0,
                "sleeve_rows": [{"signal_time": "2023-01-02", "net_cash_initial": 0.01}],
            }
            if name in {"fit_2023", "test_2024"}:
                stats["cagr_to_strict_mdd"] = 1.5
            if name == "2024_h2":
                stats["absolute_return_pct"] = 0.0
            if kwargs.get("cost_bp") == selector.STRESS_COST_BP:
                stats["absolute_return_pct"] = 0.0001
            if failure == "each_year_absolute_return_positive" and name == "fit_2023":
                stats["absolute_return_pct"] = 0.0
            elif failure == "each_year_cagr_to_strict_mdd_at_least_1_5" and name == "test_2024":
                stats["cagr_to_strict_mdd"] = 1.4999
            elif failure == "positive_half_years_at_least_3" and name in {"2023_h1", "2023_h2"}:
                stats["absolute_return_pct"] = 0.0
            elif failure == "combined_cagr_to_strict_mdd_at_least_3" and name == "combined_2023_2024":
                stats["cagr_to_strict_mdd"] = 2.9999
            elif failure == "combined_strict_mdd_at_most_15" and name == "combined_2023_2024":
                stats["strict_mdd_pct"] = 15.0001
            elif failure == "ten_bp_cost_stress_absolute_return_positive" and kwargs.get("cost_bp") == selector.STRESS_COST_BP:
                stats["absolute_return_pct"] = 0.0
            elif failure == "realized_funding_cash_positive" and name == "combined_2023_2024":
                stats["funding_cash_pct_initial"] = 0.0
            elif failure == "realized_funding_cash_at_least_transaction_cost" and name == "combined_2023_2024":
                stats["funding_to_cost_ratio"] = 0.9999
            return stats

        monkeypatch.setattr(selector, "simulate", fake_simulate)
        p_value = 0.1001 if failure == "weekly_cluster_signflip_p_at_most_0_10" else 0.10
        monkeypatch.setattr(
            selector,
            "weekly_cluster_signflip",
            lambda *args, **kwargs: {"raw_p_value": p_value, "weekly_clusters": 10},
        )
        return selector.evaluate(object(), pd.DataFrame())

    accepted = evaluate_case(None)
    assert accepted["passes_2023_2024_selection"] is True
    assert all(accepted["selection_gates"].values())
    for gate in accepted["selection_gates"]:
        rejected = evaluate_case(gate)
        assert rejected["selection_gates"][gate] is False
        assert rejected["passes_2023_2024_selection"] is False


def test_frozen_clock_is_live_and_contains_no_2026_exit() -> None:
    frozen = selector.load_clock()
    assert len(frozen) == 127
    assert frozen["exit_time"].max() < pd.Timestamp("2026-01-01")


def test_frozen_bundle_has_complete_causal_funding_marks() -> None:
    bundle = selector.load_bundle()
    for funding in bundle.funding.values():
        assert funding["mark_price"].notna().all()
        assert funding["mark_price"].gt(0).all()
        assert int((~funding["mark_is_recorded"]).sum()) == 910
