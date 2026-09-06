import numpy as np
import pandas as pd

from training import build_high_volatility_large_ticket_temporal_clustering_relay_support as subject


def state_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "source_valid": [True] * 5,
        "large_ticket_clustering": [0.01] * 5,
        "clustering_rank": [0.7, 0.81, 0.9, 0.7, 0.9],
        "turnover_hhi_rank": [0.9, 0.7, 0.9, 0.7, 0.9],
        "variation_rank": [0.8, 0.8, 0.8, 0.8, 0.8],
        "block_return": [0.1, 0.1, 0.1, -0.1, -0.1],
        "final_hour_return": [0.1, 0.1, 0.1, -0.1, -0.1],
    })


def test_primary_uses_fresh_tail_onsets() -> None:
    active, side = subject.conditions(state_frame(), "primary")
    assert active.tolist() == [False, True, False, False, True]
    assert side[active].tolist() == [1.0, -1.0]


def test_turnover_control_is_not_primary_formula() -> None:
    active, _ = subject.conditions(state_frame(), "turnover_hhi_only")
    assert active.tolist() == [False, False, True, False, True]


def test_strict_prior_midrank_excludes_current() -> None:
    values = pd.Series(np.arange(1441, dtype=float))
    assert subject.strict_prior_midrank(values).iloc[1440] == 1.0


def test_pinned_registration() -> None:
    assert subject.PREREG_SHA == "f5da0987ff2d7f8ec0081c3eff806c0b46c44ebd93fb06ec4497163c4b7565f1"
