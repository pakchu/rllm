import pandas as pd

from training import build_high_volatility_cross_alt_barrier_rejection_reversal_support as s


def test_four_clean_upper_rejections_define_upward_sweep():
    up = pd.DataFrame([[1, 1, 1, 1, 0, 0]], dtype=bool)
    down = pd.DataFrame([[0, 0, 0, 0, 0, 0]], dtype=bool)
    side, pos, neg = s.rejection_side(up, down, 4)
    assert side.iloc[0] == 1 and pos.iloc[0] == 4 and neg.iloc[0] == 0


def test_opposite_rejection_invalidates_direction():
    up = pd.DataFrame([[1, 1, 1, 1, 0, 0]], dtype=bool)
    down = pd.DataFrame([[0, 0, 0, 0, 1, 0]], dtype=bool)
    assert s.rejection_side(up, down, 4)[0].iloc[0] == 0


def test_primary_clock_fades_rejected_sweep():
    panel = pd.DataFrame(
        {
            "source_valid": [True, True],
            "rejection_side": [0, 1],
            "upper_rejection_count": [0, 4],
            "lower_rejection_count": [0, 0],
            "close_outside_side": [0, 0],
            "close_outside_up_count": [0, 0],
            "close_outside_down_count": [0, 0],
            "variation_active": [True, True],
        }
    )
    active, side, _ = s.active(panel)
    assert active.tolist() == [False, True]
    assert side.iloc[1] == -1


def test_pinned_registration():
    assert s.PREREG_SHA == "3da9c8a42036ba3dadc9553ed350018ab71f560dcbc411ce056901433bc2e0ac"
