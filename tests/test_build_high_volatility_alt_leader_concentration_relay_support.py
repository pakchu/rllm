import math

import numpy as np

from training import build_high_volatility_alt_leader_concentration_relay_support as s


def test_leader_geometry_requires_unique_nonzero_leader():
    index, value, concentration = s.leader_geometry(np.array([1, 2, 3, 4, 5, -10.0]))
    assert index == 5
    assert value == -10
    assert 1 / 6 <= concentration <= 1
    assert s.leader_geometry(np.array([1, 2, 3, 4, 10, -10.0]))[0] == -1
    assert s.leader_geometry(np.array([1, 2, 3, 4, 5, 0.0]))[0] == -1
    assert math.isnan(s.leader_geometry(np.ones(5))[1])


def test_schema_is_outcome_blind_and_controls_are_frozen():
    forbidden = {"pnl", "funding", "execution_price", "gross9", "future_return"}
    assert not {column.lower() for column in (*s.PANEL_COLUMNS, *s.CLOCK_COLUMNS)}.intersection(forbidden)
    assert len(s.CLOCK_COLUMNS) == len(set(s.CLOCK_COLUMNS))
    assert s.CONTROLS == (
        "no_concentration_tail_gate", "no_variation_gate", "equal_weight_alt_direction",
        "one_block_stale_geometry", "direction_flip", "forced_long",
    )
