from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from training import train_high_volatility_path_analog_relay as subject


def test_path_features_follow_frozen_partition_and_sum_variation_shares() -> None:
    index = pd.date_range("2022-01-01", periods=480, freq="1min", tz="UTC")
    increments = 0.0002 + 0.0001 * np.sin(np.linspace(0, 20 * np.pi, 479))
    close = 100 * np.exp(np.r_[0.0, np.cumsum(increments)])
    open_ = np.r_[close[0] / 1.001, close[:-1]]
    window = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.001, "low": np.minimum(open_, close) * 0.999, "close": close}, index=index)
    values, variation = subject.path_features(window)
    assert set(values) == set(subject.FEATURES) - {"variation_rank"}
    assert variation > 0
    assert np.isclose(sum(values[name] for name in subject.prereg.VARIATION_FEATURES), 1.0)
    assert np.isfinite(list(values.values())).all()


def test_analog_prediction_uses_fixed_inverse_distance_neighbors() -> None:
    references = np.arange(70, dtype=float).reshape(-1, 1)
    labels = np.arange(70, dtype=float)
    order = np.arange(70, dtype=np.int64)
    excluded = np.zeros(70, dtype=bool)
    excluded[0] = True
    prediction = subject.analog_prediction(np.asarray([0.0]), references, labels, order, excluded)
    distances = np.arange(1, 65, dtype=float)
    expected = np.sum((1 / distances) * np.arange(1, 65)) / np.sum(1 / distances)
    assert np.isclose(prediction, expected)


def test_purged_reference_mask_excludes_current_and_two_forward_boundaries() -> None:
    times = pd.date_range("2022-01-01", periods=6, freq="8h", tz="UTC")
    assert subject.purged_reference_mask(times[1], times).tolist() == [False, True, True, True, False, False]


def test_zero_distance_ties_are_broken_by_decision_order() -> None:
    references = np.zeros((70, 1))
    labels = np.arange(70, dtype=float)
    order = np.arange(70, dtype=np.int64)[::-1]
    prediction = subject.analog_prediction(np.asarray([0.0]), references, labels, order)
    assert np.isclose(prediction, np.mean(np.arange(6, 70)))


def test_fit_model_freezes_reference_population_and_loo_threshold() -> None:
    rng = np.random.default_rng(31)
    rows = 300
    times = pd.date_range("2021-01-01", periods=rows, freq="8h", tz="UTC")
    matrix = rng.normal(size=(rows, len(subject.FEATURES)))
    matrix[:, subject.FEATURES.index("variation_rank")] = np.linspace(0, 1, rows)
    panel = pd.DataFrame(matrix, columns=subject.FEATURES)
    panel.insert(0, "source_valid", True)
    panel.insert(0, "decision_time", times)
    labels = pd.DataFrame({"decision_time": times, "entry_time": times + pd.Timedelta(minutes=5), "exit_time": times + pd.Timedelta(hours=8, minutes=5), "label": rng.normal(0, 0.01, rows)})
    training, model = subject.fit_model(panel, labels)
    assert model["neighbors"] == 64
    assert model["reference_rows"] == 105
    assert training.reference_row.sum() == 105
    assert training.leave_one_out_prediction.notna().sum() == 105
    assert model["prediction_strength_threshold"] > 0


def test_preregistration_artifact_hash_is_frozen() -> None:
    actual = hashlib.sha256(subject.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
    assert actual == subject.PREREG_SHA256
