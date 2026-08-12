import numpy as np

from training.build_high_volatility_spot_first_passage_leadership_relay_support import (
    first_side,
    passage_indices,
)


def test_passage_indices_and_direction() -> None:
    closes = np.ones(96)
    closes[7:20] = 1.02
    closes[20:] = 0.98
    up, down = passage_indices(closes, reference=1.0, barrier=0.01)
    assert (up, down) == (7, 20)
    assert first_side(up, down) == 1


def test_one_sided_passage_is_directional() -> None:
    closes = np.ones(96)
    closes[11:] = 0.98
    up, down = passage_indices(closes, reference=1.0, barrier=0.01)
    assert (up, down) == (-1, 11)
    assert first_side(up, down) == -1


def test_no_passage_is_ineligible() -> None:
    assert first_side(-1, -1) == 0
