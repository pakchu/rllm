import math

import numpy as np
import pandas as pd

from training import build_high_volatility_alt_breadth_diffusion_slope_relay_support as s


def test_participation_slope_has_fixed_time_orientation():
    assert s.participation_slope(np.arange(8, dtype=float)) > 0
    assert s.participation_slope(np.arange(7, -1, -1, dtype=float)) < 0
    assert s.participation_slope(np.ones(8)) == 0
    assert math.isnan(s.participation_slope(np.ones(7)))


def test_onset_uses_immediately_previous_valid_boundary():
    eligible = pd.Series([False, True, False, True])
    valid = pd.Series([True, True, False, True])
    assert s.previous_valid_onset(eligible, valid).tolist() == [False, True, False, False]


def test_source_schema_is_outcome_blind_and_controls_are_frozen():
    forbidden = {"pnl", "funding", "execution_price", "gross9", "future_return"}
    columns = {column.lower() for column in (*s.PANEL_COLUMNS, *s.CLOCK_COLUMNS)}
    assert not columns.intersection(forbidden)
    assert s.CONTROLS == (
        "no_slope_tail_gate", "no_variation_gate", "endpoint_static_breadth",
        "one_block_stale_geometry", "direction_flip", "forced_long",
    )
