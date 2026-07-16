from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import training.select_leave_one_out_residual_exhaustion_pre2025 as selector
from training.select_leave_one_out_residual_exhaustion_pre2025 import (
    MarketBundle,
    _transform_same_clock,
    simulate,
    weekly_cluster_signflip,
)


def synthetic_bundle(long_closes: list[float], short_closes: list[float], funding=None) -> MarketBundle:
    dates = pd.date_range("2023-01-01", periods=len(long_closes), freq="5min")
    market = {}
    for symbol, closes in (("ADAUSDT", long_closes), ("XRPUSDT", short_closes)):
        values = np.asarray(closes, dtype=float)
        market[symbol] = {
            "open": values.copy(),
            "high": values * 1.01,
            "low": values * 0.99,
            "close": values.copy(),
        }
    funding_frames = {
        "ADAUSDT": pd.DataFrame(columns=["event_time", "funding_rate"]),
        "XRPUSDT": pd.DataFrame(columns=["event_time", "funding_rate"]),
    }
    if funding:
        for symbol, rows in funding.items():
            funding_frames[symbol] = pd.DataFrame(rows, columns=["event_time", "funding_rate"])
    return MarketBundle(dates, market, funding_frames, {})


def clock(entry="2023-01-01 00:05", exit_time="2023-01-01 00:20") -> pd.DataFrame:
    return pd.DataFrame([{
        "policy_id": "L01",
        "signal_time": pd.Timestamp(entry) - pd.Timedelta(minutes=5),
        "feature_available_time": pd.Timestamp(entry) - pd.Timedelta(minutes=5),
        "entry_time": pd.Timestamp(entry),
        "exit_time": pd.Timestamp(exit_time),
        "residual_horizon_hours": 6,
        "hold_hours": 0,
        "long_symbol": "ADAUSDT",
        "short_symbol": "XRPUSDT",
        "long_weight": 0.5,
        "short_weight_abs": 0.5,
        "long_beta": 1.0,
        "short_beta": 1.0,
        "loser_residual_z": -2.0,
        "winner_residual_z": 2.0,
        "loser_flow_z": 0.0,
        "winner_flow_z": 0.0,
        "exhaustion_score": 2.0,
    }])


def test_pair_profit_and_cost_accounting() -> None:
    bundle = synthetic_bundle([100, 100, 102, 104, 104], [100, 100, 99, 98, 98])
    stats = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=0)
    assert stats["trades"] == 1
    assert stats["absolute_return_pct"] == pytest.approx(3.0)
    costed = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=6)
    assert costed["absolute_return_pct"] < stats["absolute_return_pct"]
    assert costed["transaction_cost_pct_initial"] > 0


def test_strict_mdd_uses_favorable_before_adverse_and_liquidation_cost() -> None:
    bundle = synthetic_bundle([100, 100, 100, 100, 100], [100, 100, 100, 100, 100])
    stats = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=6)
    assert stats["strict_mdd_pct"] > stats["close_mdd_pct"]
    assert stats["strict_mdd_pct"] > 1.0


def test_funding_excludes_entry_time_and_includes_later_event() -> None:
    rows = {
        "ADAUSDT": [
            (pd.Timestamp("2023-01-01 00:05"), 0.01),
            (pd.Timestamp("2023-01-01 00:10"), 0.01),
        ]
    }
    bundle = synthetic_bundle([100, 100, 100, 100, 100], [100, 100, 100, 100, 100], rows)
    stats = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=0)
    assert stats["funding_cash_pct_initial"] == pytest.approx(-0.5)


def test_funding_at_scheduled_exit_is_included_without_exit_bar_ohlc() -> None:
    rows = {
        "ADAUSDT": [(pd.Timestamp("2023-01-01 00:20"), 0.01)],
    }
    bundle = synthetic_bundle([100, 100, 100, 100, 100], [100, 100, 100, 100, 100], rows)
    bundle.market["ADAUSDT"]["high"][-1] = 1_000.0
    bundle.market["ADAUSDT"]["low"][-1] = 1.0
    stats = simulate(bundle, clock(), start="2023-01-01", end="2023-01-02", cost_bp=0)
    assert stats["funding_cash_pct_initial"] == pytest.approx(-0.5)
    assert stats["strict_mdd_pct"] < 3.0


def test_funding_credits_do_not_raise_intratrade_peak() -> None:
    rows = {
        "ADAUSDT": [
            (pd.Timestamp("2023-01-01 00:10"), 0.02),
            (pd.Timestamp("2023-01-01 00:15"), -0.02),
        ],
    }
    bundle = synthetic_bundle([100] * 6, [100] * 6, rows)
    for symbol in bundle.market:
        bundle.market[symbol]["high"][:] = 100.0
        bundle.market[symbol]["low"][:] = 100.0
    bundle.market["ADAUSDT"]["high"][4] = 120.0
    stats = simulate(
        bundle,
        clock(exit_time="2023-01-01 00:25"),
        start="2023-01-01",
        end="2023-01-02",
        cost_bp=0,
    )
    assert stats["funding_cash_pct_initial"] == pytest.approx(0.0)
    assert stats["strict_mdd_pct"] == pytest.approx((1.0 - 1.0 / 1.09) * 100.0)


def test_direction_flip_swaps_symbols_weights_and_betas() -> None:
    base = clock()
    base.loc[0, ["long_weight", "short_weight_abs", "long_beta", "short_beta"]] = [0.4, 0.6, 1.5, 1.0]
    flipped = _transform_same_clock(base, "direction_flip")
    assert flipped.loc[0, "long_symbol"] == "XRPUSDT"
    assert flipped.loc[0, "short_symbol"] == "ADAUSDT"
    assert flipped.loc[0, "long_weight"] == pytest.approx(0.6)
    assert flipped.loc[0, "short_weight_abs"] == pytest.approx(0.4)


def test_weekly_signflip_is_deterministic() -> None:
    rows = [
        {"signal_time": f"2023-01-{day:02d}", "net_log_return": 0.01}
        for day in range(1, 15)
    ]
    one = weekly_cluster_signflip(rows, seed=7, samples=1000)
    two = weekly_cluster_signflip(rows, seed=7, samples=1000)
    assert one == two
    assert one["weekly_clusters"] >= 2


def test_simulation_rejects_overlap() -> None:
    bundle = synthetic_bundle([100] * 10, [100] * 10)
    clocks = pd.concat([clock(), clock(entry="2023-01-01 00:15", exit_time="2023-01-01 00:30")], ignore_index=True)
    with pytest.raises(RuntimeError, match="overlaps"):
        simulate(bundle, clocks, start="2023-01-01", end="2023-01-02", cost_bp=0)


def test_git_attestation_rejects_dirty_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selector.subprocess, "check_output", lambda *args, **kwargs: "?? opened-result.json\n")
    with pytest.raises(RuntimeError, match="must be clean"):
        selector._git_attestation()


def test_git_attestation_records_committed_selector_and_test(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_output(args, **kwargs):
        if args[:2] == ["git", "status"]:
            return ""
        if args[:2] == ["git", "rev-parse"]:
            return "frozen-head\n"
        raise AssertionError(args)

    def fake_call(args, **kwargs):
        calls.append(tuple(args))
        return 0

    monkeypatch.setattr(selector.subprocess, "check_output", fake_output)
    monkeypatch.setattr(selector.subprocess, "check_call", fake_call)
    monkeypatch.setattr(selector, "_file_hash", lambda path: f"hash:{path}")
    result = selector._git_attestation()
    assert result == {
        "head": "frozen-head",
        "selector_sha256": f"hash:{selector.SELECTOR_PATH}",
        "test_sha256": f"hash:{selector.TEST_PATH}",
    }
    assert len(calls) == 2
    assert all(call[:3] == ("git", "ls-files", "--error-unmatch") for call in calls)


def test_run_refuses_unapproved_support_before_git_or_outcome_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        selector,
        "_load_json_with_body_hash",
        lambda *args, **kwargs: {"clock_sha256": "wrong", "all_policies_pass_support": False},
    )
    monkeypatch.setattr(
        selector,
        "_git_attestation",
        lambda: pytest.fail("git/outcome boundary must not be reached"),
    )
    with pytest.raises(RuntimeError, match="support freeze is not approved"):
        selector.run()


def test_load_bundle_refuses_source_hash_drift_before_reading_market(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {
        "records": [
            {
                "symbol": symbol,
                "output_market_sha256": "expected-market",
                "output_funding_sha256": "expected-funding",
            }
            for symbol in selector.SYMBOLS
        ]
    }
    monkeypatch.setattr(selector, "_load_json_with_body_hash", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(selector, "_file_hash", lambda path: "tampered")
    with pytest.raises(RuntimeError, match="market source changed"):
        selector.load_bundle(source_dir="unused")


def test_frozen_clock_hash_and_temporal_contract_are_live() -> None:
    frozen = selector.load_clock()
    assert not frozen.empty
    assert (frozen["entry_time"] == frozen["signal_time"] + pd.Timedelta(minutes=5)).all()
    assert (frozen["exit_time"] < pd.Timestamp("2025-01-01")).all()
