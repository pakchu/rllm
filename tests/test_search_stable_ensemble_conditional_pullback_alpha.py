import json

import numpy as np
import pytest

from training.search_stable_ensemble_conditional_pullback_alpha import (
    EXPECTED_SELECTED_QUANTILES,
    SEARCH_SPEC,
    _result_hash,
    _spec_hash,
    adjacent_specs,
    conditional_activation,
    deterministic_forest_predict,
    source_thresholds,
)


def test_source_thresholds_are_calibrated_independently():
    predictions = np.array([0.1, 0.9, 0.2, 0.8])
    funding = np.array([True, True, False, False])
    funding_threshold, premium_threshold = source_thresholds(
        predictions,
        funding,
        funding_q=0.5,
        premium_q=0.5,
    )
    assert funding_threshold == pytest.approx(0.5)
    assert premium_threshold == pytest.approx(0.5)


def test_conditional_activation_truth_table():
    active = conditional_activation(
        size=6,
        anchor_positions=np.array([0, 1, 2, 3, 4]),
        anchor_predictions=np.array([0.9, 0.9, 0.9, 0.9, 0.4]),
        anchor_is_funding=np.array([True, True, True, False, False]),
        anchor_width=np.array([0.2, 0.05, 0.05, 0.0, 0.0]),
        anchor_pullback=np.array([0.0, -0.5, 0.1, 0.0, 0.0]),
        funding_threshold=0.8,
        premium_threshold=0.5,
        width_threshold=0.1,
        pullback_threshold=-0.2,
    )
    assert active.tolist() == [True, True, False, True, False, False]


def test_forest_prediction_forces_fixed_single_thread_reduction():
    class FakeForest:
        n_jobs = -1

        def predict(self, matrix):
            assert self.n_jobs == 1
            return np.asarray(matrix).sum(axis=1)

    model = FakeForest()
    prediction = deterministic_forest_predict(model, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert model.n_jobs == 1
    assert prediction.tolist() == [3.0, 7.0]


def test_adjacent_specs_require_exactly_one_grid_step():
    base = {"spec": dict(EXPECTED_SELECTED_QUANTILES)}
    width_neighbor = {"spec": {**EXPECTED_SELECTED_QUANTILES, "low_width_q": 0.3}}
    diagonal = {
        "spec": {
            **EXPECTED_SELECTED_QUANTILES,
            "low_width_q": 0.3,
            "pullback_q": 0.3,
        }
    }
    assert adjacent_specs(base, width_neighbor)
    assert not adjacent_specs(base, diagonal)
    assert not adjacent_specs(base, base)


def test_search_spec_keeps_oos_sealed_and_hashes_deterministically():
    assert SEARCH_SPEC["selection_end_exclusive"] == "2024-01-01"
    assert SEARCH_SPEC["grid_cells"] == 240
    assert SEARCH_SPEC["trees_per_seed"] == 2_000
    assert SEARCH_SPEC["prediction_reduction"].startswith("single-threaded")
    assert SEARCH_SPEC["seeds"] == [7, 71, 715, 2026, 71515]
    assert _spec_hash() == _spec_hash()
    payload = {
        "phase": "x",
        "oos_opened": False,
        "sealed_windows": ["2024+"],
        "selection_end_exclusive": "2024-01-01",
        "search_spec": SEARCH_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": "x",
        "source_prefix_hashes": {},
        "feature_prefix_hash": "x",
        "model": {},
        "search_summary": {},
        "selected_candidate": {},
        "passing_cells": [],
        "top_rows": [],
    }
    assert _result_hash(payload) == _result_hash(json.loads(json.dumps(payload)))
