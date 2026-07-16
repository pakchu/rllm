from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import search_cross_sectional_fragmentation_absorption_alpha as xfa


def feature_frame(*, flow_z: float = 3.0, size_z: float = -1.0) -> pd.DataFrame:
    signal = pd.Timestamp("2025-03-01 12:00:00")
    rows = []
    for symbol in xfa.SYMBOLS:
        rows.append(
            {
                "signal_time": signal,
                "feature_available_time": signal,
                "symbol": symbol,
                "beta": 1.0 if symbol == xfa.HEDGE_SYMBOL else 1.5,
                "factor_return": 0.0,
                "residual_return": 0.0,
                "residual_z": 0.1,
                "flow": 0.1,
                "flow_z": flow_z if symbol == "SOLUSDT" else 0.0,
                "average_trade_size": 100.0,
                "average_trade_size_z": size_z if symbol == "SOLUSDT" else 0.0,
                "range_risk": 0.01,
            }
        )
    return pd.DataFrame(rows)


def test_prior_zscore_does_not_put_current_value_in_its_reference() -> None:
    values = pd.Series(np.arange(20, dtype=float))
    baseline = xfa._prior_zscore(values, window=10, minimum=5)
    changed = values.copy()
    changed.iloc[-1] = 1_000_000.0
    perturbed = xfa._prior_zscore(changed, window=10, minimum=5)
    prior = values.iloc[-11:-1]
    expected = (changed.iloc[-1] - prior.mean()) / prior.std(ddof=1)
    assert perturbed.iloc[-1] == pytest.approx(expected)
    assert baseline.iloc[:-1].equals(perturbed.iloc[:-1])


def test_absorbed_buy_shorts_signal_and_preserves_beta_neutrality() -> None:
    policy = xfa.Policy("TEST", 2.0, 0.5, -0.5, 3)
    clock = xfa.build_clock(
        feature_frame(),
        policy,
        start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2026-01-01"),
    )
    assert len(clock) == 1
    row = clock.iloc[0]
    assert row["long_symbol"] == "ETHUSDT"
    assert row["short_symbol"] == "SOLUSDT"
    assert row["long_weight"] == pytest.approx(0.6)
    assert row["short_weight_abs"] == pytest.approx(0.4)
    assert row["entry_time"] == row["feature_available_time"] + pd.Timedelta(minutes=5)
    assert row["exit_time"] == row["entry_time"] + pd.Timedelta(hours=3)
    exposure = row["long_weight"] * row["long_beta"]
    exposure -= row["short_weight_abs"] * row["short_beta"]
    assert exposure == pytest.approx(0.0)


def test_absorbed_sell_longs_signal() -> None:
    policy = xfa.Policy("TEST", 2.0, 0.5, -0.5, 3)
    clock = xfa.build_clock(
        feature_frame(flow_z=-3.0),
        policy,
        start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2026-01-01"),
    )
    assert clock.loc[0, "long_symbol"] == "SOLUSDT"
    assert clock.loc[0, "short_symbol"] == "ETHUSDT"


def test_fragmentation_gate_fails_closed_but_control_can_materialize() -> None:
    policy = xfa.Policy("TEST", 2.0, 0.5, -0.5, 3)
    primary = xfa.build_clock(
        feature_frame(size_z=0.5),
        policy,
        start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2026-01-01"),
    )
    control = xfa.build_clock(
        feature_frame(size_z=0.5),
        policy,
        start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2026-01-01"),
        require_fragmentation=False,
    )
    assert primary.empty
    assert len(control) == 1


def test_policy_surface_is_small_and_pre2026_only() -> None:
    assert len(xfa.POLICIES) == 8
    assert all(policy.hold_hours in {3, 6} for policy in xfa.POLICIES)
    source = xfa.Path(xfa.__file__).read_text()
    assert "2026-01-01" in source
    assert "2026-07-01" not in source
