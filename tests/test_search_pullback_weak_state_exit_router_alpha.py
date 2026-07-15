from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.search_inventory_purge_reclaim_alpha import Trade
from training.search_pullback_weak_state_exit_router_alpha import (
    FROZEN_SPEC,
    Config,
    _fit_abs_threshold,
    _freeze_hash,
    _frozen_execution_config,
    _implementation_hash,
    _spec_hash,
    _validate_manifest,
    candidate_specs,
    oos_passes,
    schedule_window,
    shift_completed_to_boundary,
    state_labels,
)


def test_completed_bar_is_shifted_to_next_boundary() -> None:
    values = np.array([np.nan, 3.5, np.nan, -2.0])
    shifted = shift_completed_to_boundary(values, np.nan)
    assert np.isnan(shifted[0])
    assert np.isnan(shifted[1])
    assert shifted[2] == 3.5
    assert np.isnan(shifted[3])


def test_fit_threshold_ignores_2023_and_later_values() -> None:
    dates = pd.Series(pd.date_range("2020-07-01", periods=1_000, freq="2D"))
    active = np.ones(len(dates), dtype=bool)
    values = np.linspace(-2.0, 2.0, len(dates))
    baseline = _fit_abs_threshold(values, dates, active, 0.70)
    mutated = values.copy()
    mutated[dates >= pd.Timestamp("2023-01-01")] = 1_000_000.0
    assert _fit_abs_threshold(mutated, dates, active, 0.70) == baseline


def test_one_plus_nonoppose_routes_support_and_adverse() -> None:
    labels = state_labels(
        np.array([2.0, -2.0, 2.0, -2.0, 0.0]),
        np.array([2.0, 2.0, 2.0, 2.0, 0.0]),
        np.array([1, -1, -1, 1, 0]),
        np.array([0, 0, 0, 1, 2]),
        np.array([0, 0, 0, -1, 1]),
        wasserstein_threshold=1.0,
        cone_threshold=1.0,
        mode="one_plus_nonoppose",
        priority="adverse",
    )
    assert labels.tolist() == [1, -1, 0, -1, 1]


def test_adverse_priority_resolves_conflict() -> None:
    adverse = state_labels(
        np.array([2.0]),
        np.array([2.0]),
        np.array([1]),
        np.array([1]),
        np.array([-1]),
        wasserstein_threshold=1.0,
        cone_threshold=1.0,
        mode="consensus",
        priority="adverse",
    )
    support = state_labels(
        np.array([2.0]),
        np.array([2.0]),
        np.array([1]),
        np.array([1]),
        np.array([-1]),
        wasserstein_threshold=1.0,
        cone_threshold=1.0,
        mode="consensus",
        priority="support",
    )
    assert adverse.tolist() == [-1]
    assert support.tolist() == [1]


def test_grid_size_and_frozen_spec_are_stable() -> None:
    specs = candidate_specs()
    assert len(specs) == 1_296
    assert FROZEN_SPEC in specs


class _FakeEngine:
    def __init__(self) -> None:
        self.market = pd.DataFrame(
            {"date": pd.date_range("2023-01-01", periods=400, freq="5min")}
        )
        self.calls: list[tuple[int, int, int, int, int]] = []

    def trade_at(self, signal: int, side: int, hold: int, take: int, stop: int) -> Trade:
        self.calls.append((signal, side, hold, take, stop))
        return Trade(
            signal_position=signal,
            entry_position=signal + 1,
            exit_position=signal + 1 + hold,
            side=side,
            gross_return=0.0,
            price_factor=1.0,
            funding_factor=1.0,
            funding_debit_factor=1.0,
            favorable_price_factor=1.0,
            adverse_price_factor=1.0,
            entry_date=str(self.market.loc[signal + 1, "date"]),
        )


def test_skipped_adverse_signal_does_not_consume_clock() -> None:
    engine = _FakeEngine()
    active = np.zeros(len(engine.market), dtype=bool)
    labels = np.zeros(len(engine.market), dtype=np.int8)
    active[[1, 2]] = True
    labels[1] = -1
    spec = {**FROZEN_SPEC, "adverse": "skip", "neutral": "time24"}
    trades = schedule_window(
        engine,
        active,
        labels,
        spec,
        start="2023-01-01",
        end="2023-02-01",
    )
    assert [trade.signal_position for trade in trades] == [2]
    assert [call[0] for call in engine.calls] == [2]


def _oos_stats(ratio: float = 3.1) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "absolute_return_pct": 1.0,
            "cagr_to_strict_mdd": ratio,
            "strict_mdd_pct": 10.0,
            "trades": 30,
        }
        for name in ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026")
    }


def test_oos_gate_requires_each_long_window() -> None:
    assert oos_passes(_oos_stats())
    failed = _oos_stats()
    failed["eval_2025"]["cagr_to_strict_mdd"] = 2.99
    assert not oos_passes(failed)


def _valid_manifest(cfg: Config) -> dict[str, object]:
    payload: dict[str, object] = {
        "phase": "pre_oos_frozen",
        "oos_opened": False,
        "selection_end": "2024-01-01",
        "spec": FROZEN_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": _implementation_hash(),
        "frozen_execution_config": _frozen_execution_config(cfg),
        "source_prefix_hashes": {},
        "base_activation_hash": "base",
        "weak_feature_hash": "feature",
        "state_label_hash": "labels",
        "weak_thresholds": {},
        "flow_tail": 0.2,
        "selection_passed": True,
        "selection_stats": {"train": {"trades": 60}},
        "selection_schedule_hashes": {},
        "selection_grid_hash": "grid",
        "selection_grid_cells": 1_296,
    }
    payload["freeze_hash"] = _freeze_hash(payload)
    return payload


def test_freeze_hash_changes_with_selection_stats() -> None:
    cfg = Config(
        input_csv="/tmp/market.csv.gz",
        funding_csv="/tmp/funding.csv.gz",
        premium_csv="/tmp/premium.csv.gz",
    )
    payload = _valid_manifest(cfg)
    baseline = _freeze_hash(payload)
    payload["selection_stats"]["train"]["trades"] = 61
    assert _freeze_hash(payload) != baseline


@pytest.mark.parametrize("extra_key", ["oos_stats", "oos_opened_at", "unexpected"])
def test_manifest_rejects_every_extra_key(extra_key: str) -> None:
    cfg = Config(
        input_csv="/tmp/market.csv.gz",
        funding_csv="/tmp/funding.csv.gz",
        premium_csv="/tmp/premium.csv.gz",
    )
    payload = _valid_manifest(cfg)
    _validate_manifest(cfg, payload)
    payload[extra_key] = {"leak": True}
    with pytest.raises(RuntimeError, match="manifest key contract changed"):
        _validate_manifest(cfg, payload)
