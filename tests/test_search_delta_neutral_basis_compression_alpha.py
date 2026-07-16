from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.search_delta_neutral_basis_compression_alpha import (
    BasisPolicy,
    Config,
    basis_actions,
    basis_feature,
    delay_actions,
    policy_grid,
    weekly_rademacher,
)
from training.search_delta_neutral_funding_carry_alpha import Sources


def sources_from_basis(values: list[float], proxies: list[bool] | None = None) -> Sources:
    dates = pd.date_range("2023-01-01", periods=len(values), freq="1min")
    spot = np.full(len(values), 100.0)
    perp = spot * np.exp(np.asarray(values, dtype=float))
    market = pd.DataFrame(
        {
            "date": dates,
            "spot_open": spot,
            "spot_high": spot,
            "spot_low": spot,
            "spot_close": spot,
            "perp_open": perp,
            "perp_high": perp,
            "perp_low": perp,
            "perp_close": perp,
            "spot_proxy": proxies or [False] * len(values),
            "spot_observations": 4,
        }
    )
    funding = pd.DataFrame(
        columns=[
            "date",
            "funding_rate",
            "reported_mark_price",
            "exec_index",
            "settlement_index",
            "settlement_mark",
            "settlement_mark_is_reported",
            "fallback_index",
        ]
    )
    return Sources(market, funding, {}, {})


def test_grid_is_the_preregistered_twelve_variants() -> None:
    grid = policy_grid()
    assert len(grid) == 12
    assert {row.lookback_minutes for row in grid} == {10_080, 43_200}
    assert {row.entry_z for row in grid} == {2.0, 2.5, 3.0}
    assert {row.max_hold_minutes for row in grid} == {360, 1_440}


def test_basis_z_uses_current_completed_basis_against_strictly_prior_moments() -> None:
    feature = basis_feature(sources_from_basis([0.0, 0.01, 0.03]), 2)
    assert feature.loc[2, "prior_mean"] == pytest.approx(0.005)
    assert feature.loc[2, "prior_std"] == pytest.approx(0.005)
    assert feature.loc[2, "z"] == pytest.approx(5.0)


def test_signal_from_prior_minute_enters_only_at_next_five_minute_open() -> None:
    values = [0.0, 0.0, 0.0, 0.001, 0.001] + [0.001] * 15
    cfg = Config(minimum_expected_compression_bps=0.0)
    actions, trace = basis_actions(
        sources_from_basis(values),
        BasisPolicy(lookback_minutes=2, entry_z=1.0, max_hold_minutes=10),
        cfg,
    )
    assert actions[5] is True
    assert trace[0]["signal_index"] == 4
    assert trace[0]["execution_index"] == 5


def test_max_hold_exit_uses_next_five_minute_execution_grid() -> None:
    values = [0.0, 0.0, 0.0, 0.001, 0.001] + [0.002 + 0.001 * i for i in range(15)]
    actions, trace = basis_actions(
        sources_from_basis(values),
        BasisPolicy(lookback_minutes=2, entry_z=1.0, max_hold_minutes=10),
        Config(minimum_expected_compression_bps=0.0, adverse_stop_bps=1_000.0),
    )
    assert actions[15] is False
    assert next(row for row in trace if not row["target_active"])["reason"] == "max_hold"


def test_proxy_in_rolling_window_fails_closed_for_entry() -> None:
    values = [0.0, 0.0, 0.0, 0.001, 0.001] + [0.001] * 5
    proxies = [False, False, False, True, False] + [False] * 5
    actions, _ = basis_actions(
        sources_from_basis(values, proxies),
        BasisPolicy(lookback_minutes=2, entry_z=1.0, max_hold_minutes=10),
        Config(minimum_expected_compression_bps=0.0),
    )
    assert 5 not in actions


def test_proxy_value_cannot_trigger_an_early_exit() -> None:
    values = [0.0, 0.0, 0.0, 0.001, 0.001] + [0.002 + 0.001 * i for i in range(15)]
    proxies = [False] * len(values)
    proxies[9] = True
    actions, trace = basis_actions(
        sources_from_basis(values, proxies),
        BasisPolicy(lookback_minutes=2, entry_z=1.0, max_hold_minutes=10),
        Config(minimum_expected_compression_bps=0.0, adverse_stop_bps=1_000.0),
    )
    assert actions[5] is True
    assert 10 not in actions
    assert actions[15] is False
    assert next(row for row in trace if not row["target_active"])["reason"].startswith(
        "max_hold"
    )


def test_operational_delay_preserves_targets_and_moves_them_one_full_bar() -> None:
    assert delay_actions({5: True, 15: False}, 5, 30) == {10: True, 20: False}


def test_weekly_rademacher_is_deterministic() -> None:
    daily = pd.Series(
        np.full(28, 0.001), index=pd.date_range("2023-01-01", periods=28, freq="1D")
    )
    cfg = Config(bootstrap_samples=500, bootstrap_seed=7)
    assert weekly_rademacher(daily, cfg) == weekly_rademacher(daily, cfg)
    assert weekly_rademacher(daily, cfg)["one_sided_p"] < 0.2
