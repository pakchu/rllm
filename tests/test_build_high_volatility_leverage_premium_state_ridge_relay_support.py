import numpy as np
import pandas as pd

from training import build_high_volatility_leverage_premium_state_ridge_relay_support as support


def frame():
    return pd.DataFrame({"signal_valid": [True] * 5, "prediction": [0.002, -0.003, 0.0005, 0.002, -0.002], "variation_rank": [0.8, 0.9, 0.9, 0.4, 0.8]})


def model():
    return {"prediction_strength_threshold": 0.001}


def test_primary_follows_strong_frozen_prediction_in_high_volatility():
    active, side = support.conditions(frame(), model(), "primary")
    assert active.tolist() == [True, True, False, False, True]
    assert side[active].tolist() == [1, -1, -1]


def test_controls_are_diagnostic_only():
    candidate = frame()
    assert support.conditions(candidate, model(), "no_volatility_gate")[0].tolist() == [True, True, False, True, True]
    assert support.conditions(candidate, model(), "no_prediction_strength_gate")[0].tolist() == [True, True, True, False, True]
    active, side = support.conditions(frame(), model(), "direction_flip")
    assert side[active].tolist() == [-1, 1, 1]
    active, side = support.conditions(frame(), model(), "forced_long")
    assert side[active].tolist() == [1, 1, 1]


def test_prediction_is_reconstructed_from_frozen_standardization_and_coefficients():
    frozen = support.load_json(support.trained.MODEL)
    features = support.features(frozen)
    first = features.loc[features.signal_valid].iloc[0]
    vector = first.loc[list(support.trained.FEATURES)].to_numpy(float)
    expected = ((vector - np.asarray(frozen["mean"])) / np.asarray(frozen["scale"])) @ np.asarray(frozen["coefficient"]) + frozen["intercept"]
    assert np.isclose(first.prediction, expected)
