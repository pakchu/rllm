import pandas as pd

from training import build_weekend_high_volatility_overreaction_fade_support as support


def frame() -> pd.DataFrame:
    decision = pd.Timestamp("2024-07-07T12:00:00Z")
    return pd.DataFrame({
        "decision_time": [decision], "block_valid": [True], "vol_valid": [True],
        "weekend": [True], "block_return": [0.03], "prior_abs_block_q60": [0.02],
        "bvol_close": [60.0], "prior_bvol_q60": [50.0],
        "dvol_close": [65.0], "prior_dvol_q60": [55.0],
    })


def test_whvof_fades_large_high_volatility_weekend_move():
    clock = support.build_clock(frame())
    assert len(clock) == 1 and clock.iloc[0].side == -1
    assert clock.iloc[0].entry_time == pd.Timestamp("2024-07-07T12:05:00Z")
    assert clock.iloc[0].exit_time - clock.iloc[0].entry_time == pd.Timedelta(hours=6)


def test_whvof_requires_weekend_tail_and_dual_high_volatility():
    candidate = frame(); candidate.loc[0, "weekend"] = False
    assert support.build_clock(candidate).empty
    candidate = frame(); candidate.loc[0, "block_return"] = 0.01
    assert support.build_clock(candidate).empty
    candidate = frame(); candidate.loc[0, "dvol_close"] = 40.0
    assert support.build_clock(candidate).empty


def test_whvof_direction_flip_is_clock_identical():
    primary = support.build_clock(frame())
    flipped = support.build_clock(frame(), "direction_flip")
    assert len(primary) == len(flipped) == 1
    assert primary.iloc[0].entry_time == flipped.iloc[0].entry_time
    assert primary.iloc[0].side == -flipped.iloc[0].side
