from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

import training.audit_gross9_residual_direction_classifier_marginal as mod


def test_preregistration_is_hash_bound_and_family_final() -> None:
    payload = mod.load_preregistration(mod.PREREGISTRATION)
    assert payload["learner_contract"]["sklearn_version"] == "1.7.2"
    assert payload["learner_contract"]["class_encoding"]["long"] == 1
    assert payload["candidate_universe"]["portfolio_cells"] == 24
    assert "final battery" in payload["prior_battery_boundary"][
        "familywise_stop"
    ]
    assert payload["future_veto_contract"]["historical_future_can_certify"] is (
        False
    )


def test_inventory_fit_mask_is_strictly_fold_local() -> None:
    dates = pd.Series(
        pd.to_datetime(
            [
                "2020-10-14 23:55",
                "2020-10-15 00:00",
                "2021-12-31 23:55",
                "2022-01-01 00:00",
            ]
        )
    )
    mask = mod.inventory_fit_mask(
        dates, np.ones(4, dtype=bool), "2022-01-01"
    )
    assert np.array_equal(mask, [False, True, True, False])


def test_empirical_confidence_uses_mean_direction_and_population_std() -> None:
    p_long = np.asarray(
        [
            [0.7, 0.3, 0.5],
            [0.8, 0.4, 0.5],
            [0.6, 0.2, 0.5],
        ]
    )
    score, side = mod.empirical_confidence(p_long)
    mean = p_long.mean(axis=0)
    expected = np.maximum(mean, 1.0 - mean) - 0.5 * p_long.std(
        axis=0, ddof=0
    )
    assert np.allclose(score, expected)
    assert np.array_equal(side, [1, -1, 1])


def test_calibration_threshold_is_linear_q90() -> None:
    scores = np.arange(1000, dtype=float) / 1000.0
    assert mod.calibration_threshold(scores) == np.quantile(
        scores, 0.90, method="linear"
    )


def test_classifier_uses_explicit_long_class_column() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(2200, 4))
    labels = (matrix[:, 0] + 0.1 * matrix[:, 1] >= 0.0).astype(np.int8)
    fit = np.zeros(2200, dtype=bool)
    fit[:2000] = True
    predict = ~fit
    preregistration = {
        "learner_contract": {
            "n_estimators": 8,
            "max_depth": 3,
            "min_samples_leaf": 8,
            "max_features": 0.75,
            "class_weight": "balanced",
        }
    }
    probabilities, meta = mod._fit_predict_classifier(
        matrix, labels, fit, predict, preregistration
    )
    assert probabilities.shape == (3, 200)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert meta["fit_class_counts"]["short_0"] > 0
    assert meta["fit_class_counts"]["long_1"] > 0


def test_candidate_universe_and_future_surface_are_closed() -> None:
    assert len(mod.CANDIDATE_NAMES) == 6
    assert len(set(mod.CANDIDATE_NAMES)) == 6
    assert 'choices=("pre2025",)' in inspect.getsource(mod.main)
    source = inspect.getsource(mod.run_pre2025)
    assert '"future_opened": False' in source
    assert '"family_closed_if_rejected": True' in source
