from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import select_cross_collateral_near_pressure_pre2024 as selector


def _feature_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    shells: dict[str, object] = {"source_complete": [True, True, False]}
    credibility: dict[str, object] = {"source_complete": [True, True, True]}
    for venue in ("um", "cm"):
        for side in ("m", "p"):
            for shell in range(1, 6):
                shells[f"{venue}_shell_flow_net_{side}{shell}"] = [
                    float(shell if side == "m" else -shell),
                    float(2 * shell if side == "m" else shell),
                    999.0,
                ]
                shells[f"{venue}_shell_flow_efficiency_{side}{shell}"] = [0.5, 1.0, 1.0]
                credibility[f"{venue}_log_step_{side}{shell}"] = [0.0, 0.05, 0.0]
    return pd.DataFrame(shells), pd.DataFrame(credibility)


def test_near_pressure_uses_only_first_two_nonoverlapping_shells() -> None:
    shells, credibility = _feature_frames()
    value = selector.raw_pressure(
        shells,
        credibility,
        venue="um",
        weights=selector.WEIGHT_SETS["near"],
        credibility_weighted=False,
    )
    assert value.iloc[0] == pytest.approx((1.0 + 0.5 * 2.0) - (-1.0 + 0.5 * -2.0))
    assert value.iloc[1] == pytest.approx((2.0 + 0.5 * 4.0) - (1.0 + 0.5 * 2.0))
    assert np.isnan(value.iloc[2])


def test_event_mask_emits_threshold_onset_or_side_flip_only() -> None:
    score = pd.Series([0.0, 5.0, 6.0, -5.0, -6.0, 0.0, 5.0])
    event, side = selector.event_mask(score, 4.0)
    assert np.flatnonzero(event).tolist() == [1, 3, 6]
    assert side.tolist() == [0, 1, 1, -1, -1, 0, 1]


def test_candidate_grid_discloses_unique_multiplicity() -> None:
    grid = selector.candidate_grid()
    assert len(grid) == selector.EXPECTED_GRID_CELLS == 104
    keys = {(row["feature"], row["quantile"], row["hold_bars"]) for row in grid}
    assert len(keys) == len(grid)
    assert ("near_plain", 0.985, 288) in keys


def test_unknown_venue_fails_closed() -> None:
    shells, credibility = _feature_frames()
    with pytest.raises(ValueError, match="venue"):
        selector.raw_pressure(
            shells,
            credibility,
            venue="spot",
            weights=selector.WEIGHT_SETS["near"],
            credibility_weighted=False,
        )
