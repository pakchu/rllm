from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_liquidation_scar_first_passage_alpha import (
    replay_first_passage,
    touch_signals,
)


def _one_down_scar_inputs() -> tuple[pd.DataFrame, pd.Series]:
    inputs = pd.DataFrame(
        {
            "log_price": np.log([100.0, 100.0, 101.0, 100.1, 100.0, 100.0]),
            "contraction_z": [2.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            "flow_z": [-2.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    dates = pd.Series(pd.date_range("2022-01-01", periods=len(inputs), freq="5min"))
    return inputs, dates


def test_scar_must_leave_then_first_revisit_and_is_consumed_once() -> None:
    inputs, dates = _one_down_scar_inputs()

    touches = replay_first_passage(
        inputs,
        dates,
        contraction_threshold=1.0,
        max_age=10,
        zone_width=0.002,
    )

    assert touches.loc[0, "touch_count"] == 0
    assert touches.loc[1, "touch_count"] == 0
    assert touches.loc[2, "touch_count"] == 0
    assert touches.loc[3, "down_touch_mass"] > 0.0
    assert touches.loc[3, "touch_count"] == 1
    assert touches.loc[4:, "touch_count"].sum() == 0


def test_scar_does_not_touch_without_required_departure() -> None:
    inputs = pd.DataFrame(
        {
            "log_price": np.log([100.0] * 6),
            "contraction_z": [2.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            "flow_z": [2.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    dates = pd.Series(pd.date_range("2022-01-01", periods=len(inputs), freq="5min"))

    touches = replay_first_passage(
        inputs,
        dates,
        contraction_threshold=1.0,
        max_age=10,
        zone_width=0.002,
    )

    assert touches["touch_count"].sum() == 0


def test_first_passage_prefix_does_not_depend_on_future_suffix() -> None:
    prefix, prefix_dates = _one_down_scar_inputs()
    suffix = pd.DataFrame(
        {
            "log_price": np.log([1_000_000.0] * 5),
            "contraction_z": 100.0,
            "flow_z": 100.0,
        }
    )
    full = pd.concat([prefix, suffix], ignore_index=True)
    full_dates = pd.Series(pd.date_range("2022-01-01", periods=len(full), freq="5min"))

    expected = replay_first_passage(
        prefix,
        prefix_dates,
        contraction_threshold=1.0,
        max_age=10,
        zone_width=0.002,
    )
    actual = replay_first_passage(
        full,
        full_dates,
        contraction_threshold=1.0,
        max_age=10,
        zone_width=0.002,
    )
    pd.testing.assert_frame_equal(actual.iloc[: len(prefix)].reset_index(drop=True), expected.reset_index(drop=True))


def test_fade_and_permeability_are_exact_opposite_touch_mappings() -> None:
    features = pd.DataFrame(
        {
            "up_touch_mass": [0.0, 2.0, 0.0],
            "down_touch_mass": [0.0, 0.0, 3.0],
        }
    )

    fade_long, fade_short = touch_signals(features, "fade")
    permeability_long, permeability_short = touch_signals(features, "permeability")
    np.testing.assert_array_equal(fade_long, permeability_short)
    np.testing.assert_array_equal(fade_short, permeability_long)
