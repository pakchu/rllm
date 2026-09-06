import pandas as pd

from training import build_funding_polarity_volatility_ignition_relay_support as support


def frame() -> pd.DataFrame:
    settlement = pd.Timestamp("2024-07-01T08:00:00Z")
    return pd.DataFrame({
        "settlement_time": [settlement], "signal_valid": [True],
        "funding_rate": [0.0002], "previous_funding_rate": [-0.0001],
        "funding_amplitude_ratio": [2.0], "high_volatility": [True],
        "bvol_close": [60.0], "dvol_close": [65.0],
        "prior_bvol_q60": [50.0], "prior_dvol_q60": [55.0],
    })


def test_fpvir_uses_new_funding_polarity_after_dual_high_volatility():
    clock = support.build_clock(frame())
    assert len(clock) == 1 and clock.iloc[0].side == 1
    assert clock.iloc[0].entry_time == pd.Timestamp("2024-07-01T08:05:00Z")
    assert clock.iloc[0].exit_time - clock.iloc[0].entry_time == pd.Timedelta(hours=6)


def test_fpvir_requires_rotation_amplitude_and_high_volatility():
    candidate = frame(); candidate.loc[0, "previous_funding_rate"] = 0.0001
    assert support.build_clock(candidate).empty
    candidate = frame(); candidate.loc[0, "funding_amplitude_ratio"] = 0.5
    assert support.build_clock(candidate).empty
    candidate = frame(); candidate.loc[0, "high_volatility"] = False
    assert support.build_clock(candidate).empty


def test_fpvir_direction_flip_is_clock_identical():
    primary = support.build_clock(frame())
    flipped = support.build_clock(frame(), "direction_flip")
    assert len(primary) == len(flipped) == 1
    assert primary.iloc[0].entry_time == flipped.iloc[0].entry_time
    assert primary.iloc[0].side == -flipped.iloc[0].side
