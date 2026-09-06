from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from training import train_high_volatility_intrinsic_topology_ridge_relay as subject


def test_temporal_rank_concordance_tracks_monotone_direction_and_average_ties() -> None:
    assert np.isclose(subject.temporal_rank_concordance(np.arange(480, dtype=float)), 1.0)
    assert np.isclose(subject.temporal_rank_concordance(np.arange(480, 0, -1, dtype=float)), -1.0)
    tied = np.repeat(np.arange(240, dtype=float), 2)
    assert 0.99 < subject.temporal_rank_concordance(tied) <= 1.0


def test_directional_change_count_distinguishes_smooth_and_alternating_paths() -> None:
    smooth = np.linspace(0.0, 1.0, 480)
    alternating = np.tile([0.0, 0.02], 240)
    assert subject.directional_change_count(smooth, 0.01) == 0
    assert subject.directional_change_count(alternating, 0.01) == 478


def test_intrinsic_topology_features_match_frozen_feature_contract() -> None:
    index = pd.date_range("2022-01-01", periods=480, freq="1min", tz="UTC")
    phase = np.linspace(0.0, 12.0 * np.pi, 480)
    close = 100.0 * np.exp(0.0004 * np.arange(480) + 0.003 * np.sin(phase))
    open_ = np.r_[close[0] * 0.999, close[:-1]]
    window = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
        },
        index=index,
    )

    values, full_variation = subject.intrinsic_topology_features(window)

    assert set(values) == set(subject.FEATURES) - {"variation_rank"}
    assert np.isfinite(list(values.values())).all()
    assert full_variation > 0
    assert -1.0 <= values["close_location"] <= 1.0
    assert values["directional_change_density_0p5"] >= values["directional_change_density_1p0"]
    assert values["directional_change_density_1p0"] >= values["directional_change_density_2p0"]


def test_fit_model_is_closed_form_standardized_ridge_with_frozen_order() -> None:
    rng = np.random.default_rng(17)
    rows = 300
    decision_time = pd.date_range("2021-01-01", periods=rows, freq="8h", tz="UTC")
    matrix = rng.normal(size=(rows, len(subject.FEATURES)))
    matrix[:, subject.FEATURES.index("variation_rank")] = np.linspace(0.0, 1.0, rows)
    panel = pd.DataFrame(matrix, columns=subject.FEATURES)
    panel.insert(0, "source_valid", True)
    panel.insert(0, "decision_time", decision_time)
    target = matrix @ np.linspace(-0.001, 0.001, len(subject.FEATURES)) + rng.normal(0, 0.0001, rows)
    labels = pd.DataFrame(
        {
            "decision_time": decision_time,
            "entry_time": decision_time + pd.Timedelta(minutes=5),
            "exit_time": decision_time + pd.Timedelta(hours=8, minutes=5),
            "label": target,
        }
    )

    training, model = subject.fit_model(panel, labels)

    assert model["feature_order"] == list(subject.FEATURES)
    assert model["training_rows"] == rows
    assert model["training_high_volatility_rows"] == 105
    assert model["prediction_strength_threshold"] > 0
    assert np.isfinite(model["coefficient"]).all()
    standardized = (matrix - np.asarray(model["mean"])) / np.asarray(model["scale"])
    expected = standardized @ np.asarray(model["coefficient"]) + model["intercept"]
    np.testing.assert_allclose(training.fitted_prediction, expected)


def test_preregistration_artifact_hash_is_frozen() -> None:
    actual = hashlib.sha256(subject.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
    assert actual == subject.PREREG_SHA256
