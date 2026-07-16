from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import training.evaluate_leave_one_out_residual_continuation_2025 as evaluator
from training.evaluate_leave_one_out_residual_continuation_2025 import MarketBundle, simulate


def synthetic_bundle(long_values: list[float], short_values: list[float], funding=None) -> MarketBundle:
    dates = pd.date_range("2025-01-01", periods=len(long_values), freq="5min")
    market = {}
    for symbol, values in (("ETHUSDT", long_values), ("ADAUSDT", short_values)):
        array = np.asarray(values, dtype=float)
        market[symbol] = {
            "open": array.copy(),
            "high": array * 1.01,
            "low": array * 0.99,
            "close": array.copy(),
        }
    funding_frames = {
        "ETHUSDT": pd.DataFrame(columns=["event_time", "funding_rate"]),
        "ADAUSDT": pd.DataFrame(columns=["event_time", "funding_rate"]),
    }
    if funding:
        for symbol, rows in funding.items():
            funding_frames[symbol] = pd.DataFrame(rows, columns=["event_time", "funding_rate"])
    return MarketBundle(dates, market, funding_frames, {})


def clock(entry="2025-01-01 00:05", exit_time="2025-01-01 00:20") -> pd.DataFrame:
    return pd.DataFrame([{
        "policy_id": "LORC01",
        "signal_time": pd.Timestamp(entry) - pd.Timedelta(minutes=5),
        "feature_available_time": pd.Timestamp(entry) - pd.Timedelta(minutes=5),
        "entry_time": pd.Timestamp(entry),
        "exit_time": pd.Timestamp(exit_time),
        "residual_horizon_hours": 12,
        "hold_hours": 12,
        "long_symbol": "ETHUSDT",
        "short_symbol": "ADAUSDT",
        "long_weight": 0.5,
        "short_weight_abs": 0.5,
        "long_beta": 1.0,
        "short_beta": 1.0,
        "loser_residual_z": -2.0,
        "winner_residual_z": 2.0,
        "loser_flow_z": 0.0,
        "winner_flow_z": 0.0,
        "continuation_score": 2.0,
    }])


def test_pair_profit_cost_and_full_calendar_cagr() -> None:
    bundle = synthetic_bundle([100, 100, 102, 104, 104], [100, 100, 99, 98, 98])
    stats = simulate(bundle, clock(), start="2025-01-01", end="2025-01-02", cost_bp=0)
    assert stats["absolute_return_pct"] == pytest.approx(3.0)
    assert stats["cagr_pct"] > stats["absolute_return_pct"]
    costed = simulate(bundle, clock(), start="2025-01-01", end="2025-01-02", cost_bp=6)
    assert costed["absolute_return_pct"] < stats["absolute_return_pct"]


def test_strict_mdd_is_worse_than_close_mdd() -> None:
    bundle = synthetic_bundle([100] * 5, [100] * 5)
    stats = simulate(bundle, clock(), start="2025-01-01", end="2025-01-02", cost_bp=6)
    assert stats["strict_mdd_pct"] > stats["close_mdd_pct"]
    assert stats["strict_mdd_pct"] > 1.0


def test_funding_excludes_entry_and_includes_exact_exit() -> None:
    rows = {
        "ETHUSDT": [
            (pd.Timestamp("2025-01-01 00:05"), 0.01),
            (pd.Timestamp("2025-01-01 00:20"), 0.01),
        ]
    }
    bundle = synthetic_bundle([100] * 5, [100] * 5, rows)
    stats = simulate(bundle, clock(), start="2025-01-01", end="2025-01-02", cost_bp=0)
    assert stats["funding_cash_pct_initial"] == pytest.approx(-0.5)


def test_simulation_refuses_any_2026_access() -> None:
    bundle = synthetic_bundle([100] * 5, [100] * 5)
    with pytest.raises(ValueError, match="escaped calendar 2025"):
        simulate(bundle, clock(), start="2025-01-01", end="2026-01-02", cost_bp=0)


def test_git_attestation_rejects_dirty_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluator.subprocess, "check_output", lambda *args, **kwargs: "?? result.json\n")
    with pytest.raises(RuntimeError, match="must be clean"):
        evaluator._git_attestation()


def test_run_refuses_bad_support_before_attestation_or_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluator,
        "_load_manifest",
        lambda *args, **kwargs: {"clock_sha256": "wrong", "support": {"passes_support": False}},
    )
    monkeypatch.setattr(evaluator, "_git_attestation", lambda: pytest.fail("must stop before attestation"))
    with pytest.raises(RuntimeError, match="support freeze is not approved"):
        evaluator.run()


def test_frozen_clock_hash_and_calendar_boundary_are_live() -> None:
    frozen = evaluator.load_clock()
    assert len(frozen) == 99
    assert (frozen["entry_time"] == frozen["signal_time"] + pd.Timedelta(minutes=5)).all()
    assert frozen["exit_time"].max() < pd.Timestamp("2026-01-01")
