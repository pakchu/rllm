from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from training.search_pullback_capitulation_routed_take_alpha import (
    Config,
    SPEC,
    _freeze_hash,
    _frozen_execution_config,
    _implementation_hash,
    _spec_hash,
    _validate_manifest,
    _write_manifest_once,
    build_stress_mask,
    fit_route_thresholds,
    schedule_window,
    selection_passes,
)


def test_stress_route_uses_week_and_range_or_activity_interaction() -> None:
    features = pd.DataFrame(
        {
            "htf_1w_return_1": [-2.0, -2.0, -2.0, 1.0, -2.0, np.nan],
            "rex_576_range_width_pct": [3.0, 1.0, 1.0, 3.0, 3.0, 3.0],
            "quote_vol_z_1d": [0.0, -3.0, 0.0, -3.0, np.nan, -3.0],
        }
    )
    thresholds = {"week_low": 0.0, "range_wide": 2.0, "quote_activity_dry": -2.0}

    actual = build_stress_mask(features, thresholds)

    assert actual.tolist() == [True, True, False, False, False, False]


def test_route_thresholds_fit_only_active_pre_selection_events() -> None:
    dates = pd.Series(
        pd.to_datetime(
            [
                "2021-01-01",
                "2021-01-02",
                "2021-01-03",
                "2023-06-01",
                "2025-01-01",
            ]
        )
    )
    features = pd.DataFrame(
        {
            "htf_1w_return_1": [0.0, 2.0, -1_000.0, 9_000.0, 99_000.0],
            "rex_576_range_width_pct": [10.0, 20.0, -1_000.0, 9_000.0, 99_000.0],
            "quote_vol_z_1d": [0.0, 10.0, -1_000.0, 9_000.0, 99_000.0],
        }
    )
    active = np.array([True, True, False, True, True])

    thresholds = fit_route_thresholds(features, dates, active, minimum_events=2)

    assert thresholds["week_low"] == pytest.approx(1.0)
    assert thresholds["range_wide"] == pytest.approx(15.0)
    assert thresholds["quote_activity_dry"] == pytest.approx(2.0)
    assert thresholds["fit_active_events"] == 2


def test_scheduler_routes_stress_and_normal_take_profits() -> None:
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
    active[[1, 4]] = True
    stress = np.zeros(12, dtype=bool)
    stress[1] = True

    trades = schedule_window(
        engine,
        active,
        stress,
        start="2023-01-01",
        end="2023-01-02",
    )

    assert len(trades) == 2
    assert engine.calls == [
        (1, 1, 576, 400, 1_000_000),
        (4, 1, 576, 1_200, 1_000_000),
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


def test_selection_gate_requires_target_support_and_stable_segments() -> None:
    stats = _passing_stats()
    assert selection_passes(stats)

    failed = deepcopy(stats)
    failed["select_2023_h2"]["absolute_return_pct"] = -0.01
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
        "route_thresholds": {"route": 1.0},
        "activation_hash": "active",
        "route_hash": "route",
        "selection_passed": True,
        "selection_stats": _passing_stats(),
        "selection_schedule_hashes": {"train": "schedule"},
        "operating_sweep": [],
    }
    manifest["freeze_hash"] = _freeze_hash(manifest)
    _validate_manifest(cfg, manifest)

    opened = {**manifest, "oos_opened": True}
    with pytest.raises(RuntimeError, match="pre-OOS"):
        _validate_manifest(cfg, opened)

    modified = deepcopy(manifest)
    modified["route_thresholds"]["route"] = 2.0
    with pytest.raises(RuntimeError, match="freeze hash"):
        _validate_manifest(cfg, modified)


def test_frozen_manifest_cannot_be_replaced(tmp_path) -> None:
    cfg = Config(manifest_output=str(tmp_path / "manifest.json"))
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
        "route_thresholds": {"route": 1.0},
        "activation_hash": "active",
        "route_hash": "route",
        "selection_passed": True,
        "selection_stats": _passing_stats(),
        "selection_schedule_hashes": {"train": "schedule"},
        "operating_sweep": [],
    }
    manifest["freeze_hash"] = _freeze_hash(manifest)
    path = tmp_path / "manifest.json"
    assert _write_manifest_once(path, manifest, cfg) == manifest

    changed = deepcopy(manifest)
    changed["feature_prefix_hash"] = "different"
    changed["freeze_hash"] = _freeze_hash(changed)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        _write_manifest_once(path, changed, cfg)
