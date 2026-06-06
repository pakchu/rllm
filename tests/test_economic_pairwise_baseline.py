from training.economic_pairwise_baseline import build_pair_indices


def test_build_pair_indices_pairs_best_against_losers():
    rows = [
        {"date": "d", "signal_pos": 1, "utility": 0.02},
        {"date": "d", "signal_pos": 1, "utility": 0.0},
        {"date": "d", "signal_pos": 1, "utility": -0.01},
    ]
    pairs = build_pair_indices(rows, max_pairs_per_signal=2, min_utility_gap=0.001)
    assert pairs == [(0, 1, 1), (0, 2, 1)]
