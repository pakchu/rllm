from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from training.freeze_pullback_inventory_shadow_handoff_alpha import (
    MANIFEST_PAYLOAD_FIELDS,
    Trade,
    _exclude_entry_boundary_funding,
    _liquidation_cost_mdd_stats,
    _manifest_hash,
    _passes_pre_oos_gate,
    _write_manifest_once,
    combine_shadow_schedules,
)


def _trade(signal: int, exit_position: int, side: int = 1) -> Trade:
    return Trade(
        signal_position=signal,
        entry_position=signal + 1,
        exit_position=exit_position,
        side=side,
        gross_return=0.0,
        price_factor=1.0,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=1.0,
        adverse_price_factor=1.0,
        entry_date="2023-01-01 00:05:00",
    )


def _stats(ratio: float = 3.2, trades: int = 40, return_pct: float = 5.0, mdd: float = 10.0) -> dict:
    return {
        "absolute_return_pct": return_pct,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": mdd,
        "trades": trades,
    }


def test_shadow_handoff_is_chronological_nonoverlap() -> None:
    pullback = [_trade(0, 10), _trade(11, 14)]
    inventory = [_trade(5, 7, -1), _trade(15, 18, -1)]
    selected, sources, diagnostics = combine_shadow_schedules(pullback, inventory)
    assert [trade.signal_position for trade in selected] == [0, 11, 15]
    assert sources == ["pullback", "pullback", "inventory"]
    assert diagnostics == {"collisions": 1, "same_signal_ties": 0}


def test_shadow_handoff_uses_frozen_priority_only_for_exact_ties() -> None:
    pullback = [_trade(3, 6, 1)]
    inventory = [_trade(3, 5, -1)]
    selected, sources, diagnostics = combine_shadow_schedules(pullback, inventory)
    assert selected == pullback
    assert sources == ["pullback"]
    assert diagnostics == {"collisions": 1, "same_signal_ties": 1}


def test_pre_oos_gate_requires_primary_ratio_risk_support_stability_and_stress() -> None:
    names = (
        "fit",
        "fit_2020q4",
        "fit_2021",
        "fit_2022",
        "select_2023",
        "select_2023_h1",
        "select_2023_h2",
        "pre_2024",
    )
    base = {name: _stats() for name in names}
    base["fit"] = _stats(trades=160)
    stress = {name: _stats() for name in names}
    assert _passes_pre_oos_gate(base, stress)
    failed = {name: dict(value) for name, value in base.items()}
    failed["fit_2022"]["absolute_return_pct"] = -0.1
    assert not _passes_pre_oos_gate(failed, stress)
    stress["pre_2024"] = _stats(ratio=2.99)
    assert not _passes_pre_oos_gate(base, stress)


def test_accounting_sensitivities_are_explicit() -> None:
    trade = _trade(0, 2)
    engine = SimpleNamespace(
        dates=pd.Series(pd.date_range("2023-01-01", periods=3, freq="8h")),
        funding_times=pd.date_range("2023-01-01 08:00", periods=2, freq="8h")
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64),
        funding_rates=np.asarray([0.01, 0.02]),
        cfg=SimpleNamespace(leverage=0.5),
    )
    repriced = _exclude_entry_boundary_funding([trade], engine)
    assert repriced[0].funding_factor == pytest.approx(1.0 - 0.5 * 0.02)

    cfg = SimpleNamespace(leverage=0.5, fee_rate=0.0005, slippage_rate=0.0001)
    stressed = _liquidation_cost_mdd_stats(
        [trade],
        start="2023-01-01",
        end="2024-01-01",
        cfg=cfg,
    )
    assert stressed["strict_mdd_pct"] == pytest.approx((1.0 - (1.0 - 0.5 * 0.0006) ** 2) * 100.0)


def test_manifest_is_hashed_and_write_once(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "phase": "pre_oos_frozen",
        "selection_end": "2024-01-01",
        "source_prefix_hashes": {},
        "component_manifests": {},
        "component_specs": {},
        "composition": {"future_opened": False},
        "selected_schedule_hashes": {},
        "selected_stats_6bp": {},
        "stress_stats_10bp": {},
        "passes_pre_oos_gate": True,
    }
    assert tuple(payload) == MANIFEST_PAYLOAD_FIELDS
    manifest = {"created_at": "now", "manifest_hash": _manifest_hash(payload), **payload}
    path = tmp_path / "manifest.json"
    assert _write_manifest_once(path, manifest) == manifest
    assert _write_manifest_once(path, {**manifest, "created_at": "later"}) == manifest
    tampered = {**manifest, "selection_end": "2025-01-01"}
    tampered_payload = {key: tampered[key] for key in MANIFEST_PAYLOAD_FIELDS}
    tampered["manifest_hash"] = _manifest_hash(tampered_payload)
    with pytest.raises(RuntimeError, match="refusing"):
        _write_manifest_once(path, tampered)


def test_committed_manifest_is_pre_oos_and_passed() -> None:
    path = Path("results/pullback_inventory_shadow_handoff_manifest_2026-07-15.json")
    if not path.exists():
        pytest.skip("selection artifact is generated after unit tests")
    manifest = json.loads(path.read_text())
    payload = {key: manifest[key] for key in MANIFEST_PAYLOAD_FIELDS}
    assert manifest["manifest_hash"] == _manifest_hash(payload)
    assert manifest["phase"] == "pre_oos_frozen"
    assert manifest["composition"]["future_opened"] is False
    assert manifest["passes_pre_oos_gate"] is True
    assert manifest["selected_stats_6bp"]["pre_2024"]["cagr_to_strict_mdd"] >= 3.0
