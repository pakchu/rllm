import numpy as np

from training.build_high_volatility_online_ftrl_path_relay_support import (
    ftrl_update,
    ftrl_weights,
)


def test_ftrl_positive_label_increases_same_direction_score() -> None:
    z = np.zeros(3)
    n = np.zeros(3)
    x = np.array([1.0, 1.0, -1.0])
    assert np.dot(ftrl_weights(z, n), x) == 0
    ftrl_update(z, n, x, 1)
    assert np.dot(ftrl_weights(z, n), x) > 0


def test_ftrl_update_is_deterministic() -> None:
    first_z = np.zeros(2); first_n = np.zeros(2)
    second_z = np.zeros(2); second_n = np.zeros(2)
    x = np.array([1.0, 0.25])
    for label in (1, 0, 1, 1):
        ftrl_update(first_z, first_n, x, label)
        ftrl_update(second_z, second_n, x, label)
    assert np.array_equal(first_z, second_z)
    assert np.array_equal(first_n, second_n)
