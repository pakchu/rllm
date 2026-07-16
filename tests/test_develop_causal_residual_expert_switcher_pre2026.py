from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import training.develop_causal_residual_expert_switcher_pre2026 as cres
from training.develop_causal_residual_expert_switcher_pre2026 import (
    Segment,
    _completed_range_risk,
    online_ridge_choices,
    selected_clock,
    simulate_segments,
)
from training.select_leave_one_out_residual_exhaustion_pre2025 import MarketBundle


def synthetic_bundle(periods: int = 12) -> MarketBundle:
    dates = pd.date_range("2024-01-01", periods=periods, freq="5min")
    market = {}
    for symbol in ("ETHUSDT", "ADAUSDT"):
        values = np.full(periods, 100.0)
        market[symbol] = {
            "open": values.copy(),
            "high": values.copy(),
            "low": values.copy(),
            "close": values.copy(),
        }
    funding = {
        symbol: pd.DataFrame(columns=["event_time", "funding_rate"])
        for symbol in market
    }
    return MarketBundle(dates, market, funding, {})


def base_clock(signal: str = "2024-01-01 00:00", entry: str = "2024-01-01 00:05", exit_time: str = "2024-01-01 00:20") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy_id": "LORC01",
                "signal_time": pd.Timestamp(signal),
                "feature_available_time": pd.Timestamp(signal),
                "entry_time": pd.Timestamp(entry),
                "exit_time": pd.Timestamp(exit_time),
                "long_symbol": "ETHUSDT",
                "short_symbol": "ADAUSDT",
                "long_weight": 0.4,
                "short_weight_abs": 0.6,
                "long_beta": 1.5,
                "short_beta": 1.0,
                "choice": "continuation",
                "gross_scale": 0.5,
                "predicted_edge": 0.01,
                "confidence_threshold": 0.005,
            }
        ]
    )


def event_frame(rows: int = 54) -> pd.DataFrame:
    signal = pd.date_range("2023-01-01", periods=rows, freq="13h")
    frame = pd.DataFrame(
        {
            "signal_time": signal,
            "entry_time": signal + pd.Timedelta(minutes=5),
            "exit_time": signal + pd.Timedelta(hours=12, minutes=5),
            "edge": np.linspace(-0.02, 0.02, rows),
            "gross_scale": 1.0,
        }
    )
    for number, feature in enumerate(cres.MODEL_FEATURES, start=1):
        frame[feature] = np.linspace(-1.0, 1.0, rows) ** number
    return frame


def test_online_ridge_uses_only_published_exited_events() -> None:
    frame = event_frame()
    # Make row 52's otherwise-prior outcome unavailable until after row 53's signal.
    frame.loc[52, "exit_time"] = frame.loc[53, "signal_time"]
    decisions = online_ridge_choices(frame)
    assert decisions.loc[52, "training_rows"] == 52
    assert decisions.loc[53, "training_rows"] == 52


def test_online_ridge_does_not_turn_zero_prediction_into_a_trade() -> None:
    frame = event_frame()
    frame["edge"] = 0.01
    decisions = online_ridge_choices(frame)
    assert decisions.iloc[-1]["choice"] == "flat"
    assert decisions.iloc[-1]["predicted_edge"] == pytest.approx(0.0)


def test_selected_clock_reverses_then_scales_both_legs() -> None:
    clock = base_clock().drop(
        columns=["choice", "gross_scale", "predicted_edge", "confidence_threshold"]
    )
    decisions = pd.DataFrame(
        [
            {
                "signal_time": clock.loc[0, "signal_time"],
                "choice": "reversion",
                "gross_scale": 0.5,
                "predicted_edge": -0.01,
                "confidence_threshold": 0.005,
                "training_rows": 52,
            }
        ]
    )
    selected = selected_clock(clock, decisions)
    assert selected.loc[0, "long_symbol"] == "ADAUSDT"
    assert selected.loc[0, "short_symbol"] == "ETHUSDT"
    assert selected.loc[0, "long_weight"] == pytest.approx(0.3)
    assert selected.loc[0, "short_weight_abs"] == pytest.approx(0.2)
    exposure = selected.loc[0, "long_weight"] * selected.loc[0, "long_beta"]
    exposure -= selected.loc[0, "short_weight_abs"] * selected.loc[0, "short_beta"]
    assert exposure == pytest.approx(0.0)


def test_range_risk_never_reads_the_signal_or_future_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cres, "RISK_LOOKBACK_BARS", 2)
    bundle = synthetic_bundle(6)
    bundle.market["ETHUSDT"]["high"][2:] = 10_000.0
    bundle.market["ETHUSDT"]["low"][2:] = 1.0
    clock = base_clock(signal="2024-01-01 00:10")
    risk = _completed_range_risk(bundle, clock)
    assert risk.iloc[0] == pytest.approx(0.0)


def test_scaled_strict_simulator_uses_full_calendar_and_costs() -> None:
    bundle = synthetic_bundle(8)
    bundle.market["ETHUSDT"]["open"][4:] = 110.0
    bundle.market["ETHUSDT"]["close"][4:] = 110.0
    bundle.market["ETHUSDT"]["high"][4:] = 110.0
    bundle.market["ETHUSDT"]["low"][4:] = 110.0
    clock = base_clock()
    clock.loc[0, ["long_weight", "short_weight_abs"]] = [0.2, 0.3]
    segment = Segment("test", bundle, clock, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
    free = simulate_segments(
        [segment], calendar_start="2024-01-01", calendar_end="2024-01-02", cost_bp=0
    )
    costed = simulate_segments(
        [segment], calendar_start="2024-01-01", calendar_end="2024-01-02", cost_bp=6
    )
    # 0.2 of equity is allocated to the long leg, which rises 10%.
    assert free["absolute_return_pct"] == pytest.approx(2.0)
    assert free["cagr_pct"] > free["absolute_return_pct"]
    assert costed["absolute_return_pct"] < free["absolute_return_pct"]
    assert costed["transaction_cost_pct_initial"] > 0.0


def test_funding_credit_can_raise_the_strict_high_water_mark() -> None:
    bundle = synthetic_bundle(8)
    bundle.funding["ETHUSDT"] = pd.DataFrame(
        [(pd.Timestamp("2024-01-01 00:10"), -0.10)],
        columns=["event_time", "funding_rate"],
    )
    # Give back the funding credit by exit without a price move.
    bundle.funding["ETHUSDT"] = pd.concat(
        [
            bundle.funding["ETHUSDT"],
            pd.DataFrame(
                [(pd.Timestamp("2024-01-01 00:15"), 0.10)],
                columns=["event_time", "funding_rate"],
            ),
        ],
        ignore_index=True,
    )
    segment = Segment(
        "test", bundle, base_clock(), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    )
    result = simulate_segments(
        [segment], calendar_start="2024-01-01", calendar_end="2024-01-02", cost_bp=0
    )
    assert result["absolute_return_pct"] == pytest.approx(0.0)
    assert result["strict_mdd_pct"] > 1.0


def test_git_attestation_refuses_dirty_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cres.subprocess, "check_output", lambda *args, **kwargs: "?? result.json\n")
    with pytest.raises(RuntimeError, match="clean repository"):
        cres._git_attestation()
