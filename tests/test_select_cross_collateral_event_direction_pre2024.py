from __future__ import annotations

import numpy as np
import pandas as pd

from training import select_cross_collateral_event_direction_pre2024 as selector


def test_grid_is_small_and_predeclared() -> None:
    grid = selector.candidate_grid()
    assert len(grid) == 48
    assert {row["feature_group"] for row in grid} == set(selector.FEATURE_GROUPS)
    assert {row["c_value"] for row in grid} == set(selector.MODEL_CS)
    assert {row["confidence_margin"] for row in grid} == set(selector.CONFIDENCE_MARGINS)


def test_feature_groups_are_weak_signal_combinations() -> None:
    assert len(selector.FEATURE_GROUPS["trend_state"]) == 10
    assert "funding_rate_lag1" in selector.FEATURE_GROUPS["trend_carry"]
    assert "taker_48" in selector.FEATURE_GROUPS["trend_flow"]
    assert len(selector.FEATURE_GROUPS["weak_combo"]) == 18


def test_exact_factor_includes_two_sided_cost_and_realized_funding() -> None:
    cfg = selector.ExecutionConfig(
        input_csv="",
        metrics_csv="",
        funding_csv="",
        output="",
        manifest_output="",
        leverage=0.5,
        fee_rate=0.0005,
        slippage_rate=0.0001,
    )
    trade = selector.Trade(
        signal_position=0,
        entry_position=1,
        exit_position=2,
        side=1,
        gross_return=0.01,
        price_factor=1.005,
        funding_factor=0.9999,
        funding_debit_factor=0.9999,
        favorable_price_factor=1.005,
        adverse_price_factor=0.995,
        entry_date="2023-01-01",
    )
    cost = 1.0 - 0.5 * 0.0006
    assert selector.exact_factor(trade, cfg) == cost * 1.005 * 0.9999 * cost


def test_selector_never_declares_a_post_2023_fold() -> None:
    for _, (_, fit_end, start, end) in selector.GENERIC_FOLDS.items():
        assert pd.Timestamp(fit_end) <= pd.Timestamp("2023-01-01")
        assert pd.Timestamp(start) < pd.Timestamp(end) <= pd.Timestamp("2023-01-01")
    assert max(pd.Timestamp(end) for _, end in selector.EVENT_WINDOWS.values()) == pd.Timestamp(
        "2024-01-01"
    )
