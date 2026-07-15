from __future__ import annotations

from dataclasses import replace

import pytest

from training.evaluate_pullback_inventory_shadow_handoff_oos import (
    EXPECTED_FROZEN_MANIFEST_HASH,
    Config,
    _passes_oos_gate,
    _validate_frozen_manifest,
)


def _stats(ratio: float = 3.2, trades: int = 25, return_pct: float = 5.0, mdd: float = 10.0) -> dict:
    return {
        "absolute_return_pct": return_pct,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": mdd,
        "trades": trades,
    }


def test_committed_manifest_matches_external_pin() -> None:
    manifest = _validate_frozen_manifest(Config())
    assert manifest["manifest_hash"] == EXPECTED_FROZEN_MANIFEST_HASH
    assert manifest["composition"]["future_opened"] is False


def test_manifest_validation_rejects_runtime_cost_change() -> None:
    with pytest.raises(RuntimeError, match="cost"):
        _validate_frozen_manifest(replace(Config(), slippage_rate=0.0002))


def test_oos_gate_requires_independent_2024_and_2025_support() -> None:
    stats = {
        "test_2024": _stats(),
        "eval_2025": _stats(),
        "holdout_2026": _stats(trades=10),
        "oos_2024_2026": _stats(trades=60),
    }
    assert _passes_oos_gate(stats)
    stats["eval_2025"] = _stats(ratio=2.99)
    assert not _passes_oos_gate(stats)
    stats["eval_2025"] = _stats(trades=19)
    assert not _passes_oos_gate(stats)
