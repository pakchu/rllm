from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd

from training.select_expanding_extratrees_top10_pre2025 import (
    DEFAULT_DOCS,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    _json_hash,
    _render_docs,
    annual_masks,
    build_selection_base,
    exact_labels,
    semantic_tie_break,
    selection_passes,
    selection_rank,
    unique_schedule_rows,
)
from training.evaluate_stable_ensemble_conditional_pullback_oos import Config


def _stats() -> dict[str, dict[str, float | int]]:
    return {
        "test_2023": {
            "absolute_return_pct": 12.0,
            "cagr_to_strict_mdd": 4.0,
            "strict_mdd_pct": 4.0,
            "trades": 20,
        },
        "validation_2024": {
            "absolute_return_pct": 10.0,
            "cagr_to_strict_mdd": 3.5,
            "strict_mdd_pct": 5.0,
            "trades": 18,
        },
        "selection_2023_2024": {
            "absolute_return_pct": 23.0,
            "cagr_to_strict_mdd": 3.7,
            "strict_mdd_pct": 5.0,
            "trades": 38,
        },
    }


def test_selection_rank_ignores_future_metrics() -> None:
    clean = _stats()
    contaminated = copy.deepcopy(clean)
    contaminated["eval_2025"] = {
        "absolute_return_pct": -99.0,
        "cagr_to_strict_mdd": -999.0,
        "strict_mdd_pct": 99.0,
        "trades": 0,
    }
    assert selection_passes(clean)
    assert selection_rank(clean) == selection_rank(contaminated)


def test_annual_masks_purge_labels_exiting_at_cutoff() -> None:
    base = {
        "signal_dates": pd.Series(
            pd.to_datetime(["2022-12-20", "2022-12-30", "2023-02-01"])
        ),
        "targets": np.ones((3, 2), dtype=float),
        "exit_dates": pd.to_datetime(
            ["2022-12-25", "2023-01-01", "2023-02-03"]
        ).to_numpy(),
    }
    fit, predict = annual_masks(base, "2023-01-01", "2024-01-01")
    assert fit.tolist() == [True, False, False]
    assert predict.tolist() == [False, False, True]


def test_json_hash_is_order_independent_and_content_sensitive() -> None:
    assert _json_hash({"a": 1, "b": [2, 3]}) == _json_hash(
        {"b": [2, 3], "a": 1}
    )
    assert _json_hash({"a": 1}) != _json_hash({"a": 2})


def test_unique_schedule_rows_deduplicates_and_preserves_ranked_order() -> None:
    first = {
        "activation_hash": "a",
        "schedule_hashes": {"test": "1"},
        "learner": {"max_depth": 2, "min_samples_leaf": 32, "max_features": 0.5},
        "selection": {
            "risk_lambda": 0.5,
            "funding_quantile": 0.4,
            "premium_quantile": 0.5,
            "risk_quantile": 0.8,
        },
    }
    duplicate = copy.deepcopy(first)
    duplicate["selection"]["risk_quantile"] = 0.85
    distinct = copy.deepcopy(first)
    distinct["activation_hash"] = "b"
    rows = unique_schedule_rows([first, duplicate, distinct])
    assert rows == [first, distinct]
    assert semantic_tie_break(first) > semantic_tie_break(duplicate)


def test_bad_physical_cutoff_is_rejected_before_loading_data() -> None:
    try:
        build_selection_base(replace(Config(), exclude_from="2026-01-01"))
    except ValueError as error:
        assert "selection cutoff" in str(error)
    else:
        raise AssertionError("bad cutoff was accepted")


def test_exact_labels_apply_both_sides_cost_and_realized_funding() -> None:
    cfg = replace(Config(), leverage=0.5, fee_rate=0.0005, slippage_rate=0.0001)
    trade = SimpleNamespace(
        price_factor=1.02,
        funding_factor=0.999,
        funding_debit_factor=0.999,
        adverse_price_factor=0.98,
    )
    net, adverse = exact_labels(cast(Any, trade), cfg)
    fee = 1.0 - 0.5 * 0.0006
    assert np.isclose(net, fee * 1.02 * 0.999 * fee - 1.0)
    assert np.isclose(adverse, 1.0 - fee * 0.999 * 0.98)


def test_frozen_artifact_integrity_and_unique_top10() -> None:
    result_path = Path(DEFAULT_OUTPUT)
    manifest_path = Path(DEFAULT_MANIFEST)
    docs_path = Path(DEFAULT_DOCS)
    if not (result_path.exists() and manifest_path.exists() and docs_path.exists()):
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded_hash = manifest.pop("manifest_hash")
    assert _json_hash(manifest) == embedded_hash
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == manifest["selection_result_sha256"]
    manifest["manifest_hash"] = embedded_hash
    assert docs_path.read_text(encoding="utf-8") == _render_docs(result, manifest)
    identities = {
        (
            row["activation_hash"],
            tuple(sorted(row["schedule_hashes"].items())),
        )
        for row in result["top10"]
    }
    assert len(identities) == len(result["top10"]) == 10
    for row in result["top10"]:
        assert set(row["stats"]) == {
            "test_2023",
            "validation_2024",
            "selection_2023_2024",
        }
