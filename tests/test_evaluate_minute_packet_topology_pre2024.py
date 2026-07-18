from __future__ import annotations

from dataclasses import replace

import pandas as pd

from training import evaluate_minute_packet_topology_pre2024 as evaluator
from training.search_inventory_purge_reclaim_alpha import Trade


def _trade(signal: int, side: int = 1) -> Trade:
    return Trade(
        signal_position=signal,
        entry_position=signal + 1,
        exit_position=signal + 3,
        side=side,
        gross_return=0.01,
        price_factor=1.005,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=1.01,
        adverse_price_factor=0.995,
        entry_date="2023-01-01 00:05:00",
    )


def test_window_filter_requires_signal_entry_and_exit_inside_split() -> None:
    dates = pd.Series(pd.date_range("2022-12-31 23:45:00", periods=10, freq="5min"))
    crossing = _trade(2)
    contained = _trade(4)
    selected = evaluator._window_trades(
        [crossing, contained], dates, "select_2023"
    )
    assert selected == [contained]


def test_net_return_charges_both_sides_and_funding() -> None:
    cfg = evaluator.EvaluationConfig()
    trade = replace(_trade(1), price_factor=1.01, funding_factor=0.999)
    observed = evaluator._net_trade_returns([trade], cfg)[0]
    execution = 1.0 - cfg.leverage * (cfg.fee_rate + cfg.slippage_rate)
    assert observed == execution * 1.01 * 0.999 * execution - 1.0


def test_train_gate_requires_ratio_mdd_return_and_count() -> None:
    passing = {
        "absolute_return_pct": 10.0,
        "cagr_to_strict_mdd": 1.5,
        "strict_mdd_pct": 20.0,
        "trades": 100,
    }
    assert all(evaluator._train_coarse_gates(passing).values())
    failing = dict(passing, strict_mdd_pct=20.01)
    assert not evaluator._train_coarse_gates(failing)["strict_mdd_at_most_20"]


def test_selection_gate_requires_both_halves_and_stress() -> None:
    full = {
        "absolute_return_pct": 10.0,
        "cagr_to_strict_mdd": 3.0,
        "strict_mdd_pct": 15.0,
        "weekly_cluster_sign_flip": {"p_value_one_sided": 0.09},
    }
    half = {"absolute_return_pct": 1.0, "trades": 20}
    stress = {"absolute_return_pct": 0.1}
    assert all(evaluator._selection_gates(full, half, half, stress).values())
    bad = dict(half, absolute_return_pct=-0.01)
    assert not evaluator._selection_gates(full, half, bad, stress)[
        "each_half_absolute_return_positive"
    ]
