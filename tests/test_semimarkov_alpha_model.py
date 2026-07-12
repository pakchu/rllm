import numpy as np

from training.search_semimarkov_duration_alpha import causal_run_age, duration_key


def test_run_age_is_prefix_causal_and_resets_on_state_change():
    state = np.array([-1, 2, 2, 2, 5, 5, -1, 5, 5, 5, 5])

    full = causal_run_age(state)
    prefix = causal_run_age(state[:8])

    np.testing.assert_array_equal(full[:8], prefix)
    np.testing.assert_array_equal(full, np.array([0, 1, 2, 3, 1, 2, 0, 1, 2, 3, 4]))


def test_run_age_resets_across_missing_hour():
    state = np.array([2, 2, 2])
    timestamps = np.array(
        ["2026-01-01T00:00", "2026-01-01T01:00", "2026-01-01T04:00"],
        dtype="datetime64[m]",
    )

    np.testing.assert_array_equal(causal_run_age(state, timestamps), np.array([1, 2, 1]))


def test_duration_key_uses_fixed_causal_age_buckets():
    state = np.array([1, 1, 1, 1, 1, 2])
    key, age = duration_key(state, (1, 3))

    np.testing.assert_array_equal(age, np.array([1, 2, 3, 4, 5, 1]))
    np.testing.assert_array_equal(key, np.array([3, 4, 4, 5, 5, 6]))
