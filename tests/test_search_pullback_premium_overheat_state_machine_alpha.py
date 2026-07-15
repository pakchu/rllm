from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from training.search_pullback_premium_overheat_state_machine_alpha import (
    Config,
    FEATURE_QUANTILES,
    FROZEN_CHAMPION,
    SPEC,
    _freeze_hash,
    _frozen_execution_config,
    _implementation_hash,
    _spec_hash,
    _threshold_key,
    _validate_manifest,
    build_state_masks,
    fit_state_thresholds,
    schedule_window,
    selection_passes,
)


def _feature_frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame({name: np.arange(rows, dtype=float) for name in FEATURE_QUANTILES})


def test_state_thresholds_ignore_inactive_and_post_fit_rows() -> None:
    dates = pd.Series(
        pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2023-05-01", "2025-01-01"])
    )
    features = _feature_frame(5)
    for column in features:
        features[column] = [0.0, 10.0, -10_000.0, 20_000.0, 90_000.0]
    active = np.array([True, True, False, True, True])

    thresholds = fit_state_thresholds(features, dates, active, minimum_events=2)

    assert thresholds["fit_active_events"] == 2
    assert thresholds[_threshold_key("htf_1w_return_1", 0.50)] == pytest.approx(5.0)
    assert thresholds[_threshold_key("quote_vol_z_1d", 0.20)] == pytest.approx(2.0)
    assert thresholds[_threshold_key("premium_index_change", 0.67)] == pytest.approx(6.7)


def test_state_masks_use_capitulation_or_and_premium_overheat_and() -> None:
    features = pd.DataFrame(
        {
            "htf_1w_return_1": [-2.0, -2.0, 1.0, 1.0],
            "rex_576_range_width_pct": [3.0, 1.0, 1.0, 1.0],
            "quote_vol_z_1d": [0.0, -3.0, 0.0, 0.0],
            "rex_576_range_pos": [0.0, 0.0, 3.0, 1.0],
            "bb_z": [0.0] * 4,
            "premium_index_change": [0.0, 0.0, 3.0, 3.0],
            "htf_3d_return_1": [0.0] * 4,
        }
    )
    thresholds = {
        _threshold_key("htf_1w_return_1", 0.50): 0.0,
        _threshold_key("rex_576_range_width_pct", 0.50): 2.0,
        _threshold_key("quote_vol_z_1d", 0.20): -2.0,
        _threshold_key("premium_index_change", 0.67): 2.0,
        _threshold_key("rex_576_range_pos", 0.67): 2.0,
    }

    capitulation, overheat = build_state_masks(
        features, thresholds, FROZEN_CHAMPION["overheat"]
    )

    assert capitulation.tolist() == [True, True, False, False]
    assert overheat.tolist() == [False, False, True, False]


def test_scheduler_prioritizes_capitulation_then_skips_overheat() -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.market = pd.DataFrame(
                {"date": pd.date_range("2023-01-01", periods=12, freq="5min")}
            )
            self.calls: list[tuple[int, int, int, int, int]] = []

        def trade_at(self, signal: int, side: int, hold: int, tp: int, stop: int):
            self.calls.append((signal, side, hold, tp, stop))
            return SimpleNamespace(exit_position=signal + 1)

    engine = FakeEngine()
    active = np.zeros(12, dtype=bool)
    active[[1, 4, 7]] = True
    capitulation = np.zeros(12, dtype=bool)
    capitulation[1] = True
    overheat = np.zeros(12, dtype=bool)
    overheat[[1, 4]] = True

    trades = schedule_window(
        engine,
        active,
        capitulation,
        overheat,
        overheat_action="skip",
        start="2023-01-01",
        end="2023-01-02",
    )

    assert len(trades) == 2
    assert engine.calls == [
        (1, 1, 576, 400, 1_000_000),
        (7, 1, 576, 1_200, 1_000_000),
    ]


def _passing_stats() -> dict[str, dict[str, float | int]]:
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
    result = {
        name: {
            "absolute_return_pct": 10.0,
            "cagr_to_strict_mdd": 3.1,
            "strict_mdd_pct": 10.0,
            "trades": 20,
        }
        for name in names
    }
    result["train"]["trades"] = 60
    result["select_2023"]["trades"] = 12
    return result


def test_selection_gate_rejects_unstable_or_subtarget_candidate() -> None:
    stats = _passing_stats()
    assert selection_passes(stats)

    failed = deepcopy(stats)
    failed["train_2022"]["absolute_return_pct"] = -0.1
    assert not selection_passes(failed)

    failed = deepcopy(stats)
    failed["pre_2024"]["cagr_to_strict_mdd"] = 2.99
    assert not selection_passes(failed)


def test_opened_or_modified_manifest_is_rejected() -> None:
    cfg = Config()
    manifest = {
        "oos_opened": False,
        "selection_end": "2024-01-01",
        "spec": SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": _implementation_hash(),
        "frozen_execution_config": _frozen_execution_config(cfg),
        "source_prefix_hashes": {"market": "m", "funding": "f", "premium": "p"},
        "feature_prefix_hash": "features",
        "base_thresholds": {"base": 1.0},
        "state_thresholds": {"state": 1.0},
        "activation_hash": "active",
        "capitulation_hash": "cap",
        "overheat_hash": "hot",
        "selection_passed": True,
        "selection_stats": _passing_stats(),
        "selection_schedule_hashes": {"train": "schedule"},
        "selection_grid": [],
    }
    manifest["freeze_hash"] = _freeze_hash(manifest)
    _validate_manifest(cfg, manifest)

    with pytest.raises(RuntimeError, match="pre-OOS"):
        _validate_manifest(cfg, {**manifest, "oos_opened": True})

    modified = deepcopy(manifest)
    modified["state_thresholds"]["state"] = 2.0
    with pytest.raises(RuntimeError, match="freeze hash"):
        _validate_manifest(cfg, modified)
