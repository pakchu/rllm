from __future__ import annotations

import numpy as np

from training.search_absorbed_basis_recoil_pullback_alpha import (
    build_recoil_trigger,
    grid_specs,
    selection_passes,
)


def _stats(*, ratio: float = 3.1, mdd: float = 10.0) -> dict[str, dict[str, float]]:
    names = (
        "train",
        "train_2020h2",
        "train_2021",
        "train_2022",
        "select_2023",
        "select_2023_h1",
        "select_2023_h2",
        "pre_2024",
    )
    out = {
        name: {
            "absolute_return_pct": 1.0,
            "cagr_to_strict_mdd": ratio,
            "strict_mdd_pct": mdd,
            "trades": 5,
        }
        for name in names
    }
    out["train"]["trades"] = 50
    out["select_2023"]["trades"] = 10
    return out


def test_grid_is_the_pre_registered_twelve_combinations() -> None:
    specs = grid_specs()
    assert len(specs) == 12
    assert {spec["lag_hours"] for spec in specs} == {1, 3, 6}
    assert {spec["sell_quantile"] for spec in specs} == {0.30, 0.40}
    assert {spec["participation_quantile"] for spec in specs} == {0.50, 0.70}
    assert len({spec["name"] for spec in specs}) == 12


def test_trigger_requires_a_strictly_earlier_setup_and_premium_recoupling() -> None:
    rows = 40
    setup = np.zeros(rows, dtype=bool)
    decisions = np.zeros(rows, dtype=bool)
    decisions[[0, 12, 24, 36]] = True
    setup[12] = True
    premium = np.zeros(rows)
    premium[12] = -2.0
    premium[24] = -1.0
    available = np.ones(rows, dtype=bool)
    taker = np.full(rows, -0.5)
    price = np.full(rows, 0.1)
    participation = np.full(rows, 2.0)

    active = build_recoil_trigger(
        setup,
        decisions,
        premium=premium,
        premium_available=available,
        taker_imbalance_12=taker,
        price_return_12=price,
        participation=participation,
        lag_hours=1,
        taker_threshold=-0.25,
        price_threshold=0.0,
        participation_threshold=1.0,
    )

    assert not active[12]  # Same-row setup cannot trigger itself.
    assert active[24]
    assert active.sum() == 1

    premium[24] = -3.0
    no_recoupling = build_recoil_trigger(
        setup,
        decisions,
        premium=premium,
        premium_available=available,
        taker_imbalance_12=taker,
        price_return_12=price,
        participation=participation,
        lag_hours=1,
        taker_threshold=-0.25,
        price_threshold=0.0,
        participation_threshold=1.0,
    )
    assert not no_recoupling.any()


def test_trigger_requires_fresh_premium_on_anchor_and_trigger() -> None:
    rows = 25
    setup = np.zeros(rows, dtype=bool)
    decisions = np.zeros(rows, dtype=bool)
    setup[0] = True
    decisions[[0, 12, 24]] = True
    kwargs = {
        "premium": np.linspace(-2.0, 0.0, rows),
        "taker_imbalance_12": np.full(rows, -1.0),
        "price_return_12": np.ones(rows),
        "participation": np.ones(rows),
        "lag_hours": 1,
        "taker_threshold": 0.0,
        "price_threshold": 0.0,
        "participation_threshold": 0.0,
    }

    unavailable_anchor = np.ones(rows, dtype=bool)
    unavailable_anchor[0] = False
    assert not build_recoil_trigger(
        setup,
        decisions,
        premium_available=unavailable_anchor,
        **kwargs,
    ).any()

    unavailable_trigger = np.ones(rows, dtype=bool)
    unavailable_trigger[12] = False
    assert not build_recoil_trigger(
        setup,
        decisions,
        premium_available=unavailable_trigger,
        **kwargs,
    ).any()


def test_selection_gate_enforces_counts_target_and_mdd() -> None:
    assert selection_passes(_stats())

    low_ratio = _stats()
    low_ratio["pre_2024"]["cagr_to_strict_mdd"] = 2.99
    assert not selection_passes(low_ratio)

    high_mdd = _stats()
    high_mdd["select_2023"]["strict_mdd_pct"] = 15.01
    assert not selection_passes(high_mdd)

    sparse = _stats()
    sparse["select_2023_h2"]["trades"] = 3
    assert not selection_passes(sparse)
