from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_premium_intrabar_shape_alpha import _signals, build_features


def _market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "premium_shape_available": [True, True, True],
            "premium_index_1m_close": [0.0, 0.2, 0.1],
            "premium_index_1m_low": [-1.0, -0.1, 0.0],
            "premium_index_1m_high": [0.1, 1.0, 0.2],
        }
    )


def test_shape_features_identify_lower_and_upper_wick_rejections() -> None:
    features = build_features(_market())
    assert features.loc[0, "psi_wick_imbalance"] > 0.5
    assert features.loc[1, "psi_wick_imbalance"] < -0.5


def test_signal_only_fires_on_range_z_onset() -> None:
    frame = pd.DataFrame(
        {
            "psi_range_z_2016": [1.0, 2.5, 3.0],
            "psi_wick_imbalance": [0.0, 0.8, 0.9],
            "psi_close_location": [0.0, 0.8, 0.8],
        }
    )
    spec = {"window": 2016, "range_z": 2.0, "shape_threshold": 0.5, "mode": "agreement", "direction": "follow", "hold": 24}
    active, side = _signals(frame, spec)
    assert active.tolist() == [False, True, False]
    assert side.tolist() == [0, 1, 0]


def test_disagreement_uses_wick_direction_and_can_flip() -> None:
    frame = pd.DataFrame(
        {
            "psi_range_z_2016": [1.0, 3.0],
            "psi_wick_imbalance": [0.0, 0.8],
            "psi_close_location": [0.0, -0.8],
        }
    )
    spec = {"window": 2016, "range_z": 2.0, "shape_threshold": 0.5, "mode": "disagreement", "direction": "follow", "hold": 24}
    active, side = _signals(frame, spec, flip=True)
    assert active.tolist() == [False, True]
    assert side.tolist() == [0, -1]
