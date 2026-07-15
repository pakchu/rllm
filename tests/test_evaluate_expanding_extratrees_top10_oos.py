from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.evaluate_expanding_extratrees_top10_oos import (
    ALL_COMBINED,
    FUTURE_COMBINED,
    assert_prefix_parity,
    full_passes,
    future_passes,
    validate_manifest,
)
from training.select_expanding_extratrees_top10_pre2025 import DEFAULT_MANIFEST


def _metric(*, ratio: float = 3.5, trades: int = 20) -> dict[str, float | int]:
    return {
        "absolute_return_pct": 10.0,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": 5.0,
        "trades": trades,
    }


def _stats() -> dict[str, dict[str, float | int]]:
    return {
        "test_2023": _metric(),
        "validation_2024": _metric(),
        "eval_2025": _metric(),
        "holdout_2026h1": _metric(trades=8),
        FUTURE_COMBINED[0]: _metric(trades=28),
        ALL_COMBINED[0]: _metric(trades=68),
    }


def test_committed_manifest_validates() -> None:
    manifest, result = validate_manifest(DEFAULT_MANIFEST)
    assert manifest["manifest_hash"]
    assert len(manifest["top10"]) == len(result["top10"]) == 10


def test_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(Path(DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    payload["top10"][0]["rank_position"] = 999
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_manifest(path)
    except RuntimeError as error:
        assert "manifest hash" in str(error)
    else:
        raise AssertionError("tampered manifest was accepted")


def test_future_and_full_gates_are_independent() -> None:
    stats = _stats()
    assert future_passes(stats)
    assert full_passes(stats)
    stats["test_2023"] = _metric(ratio=2.0)
    assert future_passes(stats)
    assert not full_passes(stats)
    stats["eval_2025"] = _metric(ratio=2.0)
    assert not future_passes(stats)


def test_prefix_parity_detects_feature_change() -> None:
    dates = pd.Series(
        pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:05:00"])
    )
    selection = {
        "context": {
            "dates": dates,
            "matrix": np.zeros((2, 2)),
            "base": np.array([True, False]),
            "funding_leg": np.array([True, False]),
            "premium_leg": np.array([False, False]),
        }
    }
    replay = {
        "context": {
            "dates": pd.concat(
                [dates, pd.Series(pd.to_datetime(["2024-01-01 00:10:00"]))],
                ignore_index=True,
            ),
            "matrix": np.zeros((3, 2)),
            "base": np.array([True, False, False]),
            "funding_leg": np.array([True, False, False]),
            "premium_leg": np.array([False, False, False]),
        }
    }
    assert_prefix_parity(selection, replay)
    replay["context"]["matrix"][1, 0] = 1.0
    try:
        assert_prefix_parity(selection, replay)
    except RuntimeError as error:
        assert "matrix prefix" in str(error)
    else:
        raise AssertionError("changed prefix was accepted")
