import numpy as np
import pandas as pd

from training import build_high_volatility_alt_modularity_fragmentation_relay_support as s


def test_modularity_recovers_two_dense_communities():
    correlation = np.full((6, 6), 0.05)
    np.fill_diagonal(correlation, 1.0)
    for group, weight in (((0, 1, 2), 0.92), ((3, 4, 5), 0.78)):
        for position, left in enumerate(group):
            for right in group[position + 1 :]:
                correlation[left, right] = correlation[right, left] = weight
    first, second, modularity = s.modularity_partition(correlation)
    assert first == (0, 1, 2)
    assert second == (3, 4, 5)
    assert modularity > 0.4


def test_rank_excludes_current():
    result = s.prior_rank(pd.Series(range(181), dtype=float))
    assert result.iloc[:180].isna().all()
    assert result.iloc[180] == 1.0


def test_onset_and_direction_controls_are_frozen():
    decisions = pd.date_range("2024-07-01T04:00:00Z", periods=4, freq="8h")
    frame = pd.DataFrame(
        {
            "decision_time": decisions,
            "feature_available_time": decisions,
            "source_valid": True,
            "community_a": ["ADAUSDT|BNBUSDT|DOGEUSDT"] * 4,
            "community_b": ["ETHUSDT|SOLUSDT|XRPUSDT"] * 4,
            "modularity": [0.3] * 4,
            "modularity_rank": [0.7, 0.8, 0.9, 0.7],
            "community_a_final_hour_return": [0.01] * 4,
            "community_b_final_hour_return": [-0.005] * 4,
            "dominant_community_return": [0.01] * 4,
            "all_alt_final_hour_return": [-0.002] * 4,
            "btc_realized_variation": [0.01] * 4,
            "variation_rank": [0.7] * 4,
            "side": [1] * 4,
            "all_alt_side": [-1] * 4,
            "eligible": [False, True, True, False],
        },
        columns=s.PANEL_COLS,
    )
    onset, side, _ = s.active(frame)
    assert onset.tolist() == [False, True, False, False]
    assert side.tolist() == [1] * 4
    assert s.active(frame, "all_alt_final_hour_median")[1].tolist() == [-1] * 4
    assert s.PREREG_SHA == "578613ce3da6717f0a2ba5fd799bfa829fa5e1f17e575a9b47fda9ed2f4a66d4"
