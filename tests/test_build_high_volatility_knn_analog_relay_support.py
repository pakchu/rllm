import numpy as np

from training import build_high_volatility_knn_analog_relay_support as support


def test_entropy_is_finite_and_bounded():
    value = support._entropy(np.asarray([1.0, -1.0, 0.0] * 96))
    assert 0.0 <= value <= 1.0


def test_frozen_feature_and_stage_contract():
    assert len(support.FEATURES) == 9
    assert support.FIT_END.isoformat() == "2023-07-01T00:00:00+00:00"
    assert support.MINIMUM == {"train": 8, "test": 12, "eval": 12, "final": 8}
