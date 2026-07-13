from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_positioning_lifecycle_hazard_alpha import (
    lifecycle_signals,
    positioning_valid_mask,
    static_tail_onsets,
)


def test_aged_positive_disagreement_contraction_resolves_short() -> None:
    values = np.array([1.6, 1.8, 1.7, 0.8, 0.7])

    long_active, short_active, diagnostics = lifecycle_signals(
        values,
        np.ones(len(values), dtype=bool),
        min_age=4,
        trigger="contraction",
    )

    assert not long_active.any()
    np.testing.assert_array_equal(np.flatnonzero(short_active), [3])
    assert diagnostics["episode_age"][3] == 4
    np.testing.assert_allclose(diagnostics["resolution_fraction"][3], 0.8 / 1.8)


def test_aged_negative_disagreement_zero_cross_resolves_long() -> None:
    values = np.array([-1.6, -1.4, -0.8, 0.2, 0.3])

    long_active, short_active, _ = lifecycle_signals(
        values,
        np.ones(len(values), dtype=bool),
        min_age=4,
        trigger="zero_cross",
    )

    np.testing.assert_array_equal(np.flatnonzero(long_active), [3])
    assert not short_active.any()


def test_invalid_row_resets_episode_and_prevents_false_resolution() -> None:
    values = np.array([1.6, 1.8, 1.7, 0.8, 0.7])
    valid = np.array([True, True, False, True, True])

    long_active, short_active, _ = lifecycle_signals(
        values,
        valid,
        min_age=4,
        trigger="contraction",
    )

    assert not long_active.any() and not short_active.any()


def test_lifecycle_prefix_is_suffix_independent_and_flip_is_exact() -> None:
    prefix = np.array([1.6, 1.8, 1.7, 0.8, 0.7])
    first = np.r_[prefix, [100.0, -100.0]]
    second = np.r_[prefix, [-100.0, 100.0]]
    valid = np.ones(len(first), dtype=bool)
    kwargs = {"min_age": 4, "trigger": "contraction"}

    first_long, first_short, first_diagnostics = lifecycle_signals(first, valid, **kwargs)
    second_long, second_short, second_diagnostics = lifecycle_signals(second, valid, **kwargs)
    flip_long, flip_short, _ = lifecycle_signals(first, valid, flip=True, **kwargs)

    np.testing.assert_array_equal(first_long[: len(prefix)], second_long[: len(prefix)])
    np.testing.assert_array_equal(first_short[: len(prefix)], second_short[: len(prefix)])
    for key in first_diagnostics:
        np.testing.assert_allclose(
            first_diagnostics[key][: len(prefix)],
            second_diagnostics[key][: len(prefix)],
            equal_nan=True,
        )
    np.testing.assert_array_equal(flip_long, first_short)
    np.testing.assert_array_equal(flip_short, first_long)


def test_2022_quarantine_forcibly_invalidates_lifecycle_rows() -> None:
    dates = pd.Series(
        pd.to_datetime(
            [
                "2021-12-31 23:55",
                "2022-01-01 00:00",
                "2022-12-31 23:55",
                "2023-01-01 00:00",
            ]
        )
    )

    actual = positioning_valid_mask(dates, np.ones(len(dates), dtype=bool))

    np.testing.assert_array_equal(actual, [True, False, False, True])


def test_static_tail_control_emits_onset_only() -> None:
    values = np.array([0.0, 1.6, 1.7, 1.8, 0.0, -1.6, -1.7])

    long_active, short_active = static_tail_onsets(
        values,
        np.ones(len(values), dtype=bool),
    )

    np.testing.assert_array_equal(np.flatnonzero(long_active), [5])
    np.testing.assert_array_equal(np.flatnonzero(short_active), [1])
