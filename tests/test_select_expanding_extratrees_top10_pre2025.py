from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from training.select_expanding_extratrees_top10_pre2025 import (
    _json_hash,
    annual_masks,
    selection_passes,
    selection_rank,
)


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
