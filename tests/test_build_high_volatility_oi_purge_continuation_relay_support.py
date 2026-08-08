import pandas as pd

from training import build_high_volatility_oi_purge_continuation_relay_support as support


def frame() -> pd.DataFrame:
    decision = pd.Timestamp("2024-07-01T08:00:00Z")
    return pd.DataFrame(
        {
            "decision_time": [decision - pd.Timedelta(hours=1), decision],
            "source_valid": [True, True],
            "bvol_close": [70.0, 70.0],
            "prior_bvol_q60": [60.0, 60.0],
            "dvol_close": [65.0, 65.0],
            "prior_dvol_q60": [60.0, 60.0],
            "oi_current_time": [decision - pd.Timedelta(hours=1), decision],
            "oi_prior_time": [decision - pd.Timedelta(hours=2), decision - pd.Timedelta(hours=1)],
            "oi_change": [0.0, -0.02],
            "prior_abs_oi_change_q75": [0.01, 0.01],
            "hour_return": [0.001, -0.03],
            "prior_abs_return_q60": [0.01, 0.01],
        }
    )


def test_hvopcr_follows_price_direction_after_large_oi_purge():
    clocks = support.clock(frame())
    assert len(clocks) == 1
    assert clocks.iloc[0].side == -1
    assert clocks.iloc[0].entry_time == pd.Timestamp("2024-07-01T08:05:00Z")
    assert clocks.iloc[0].exit_time - clocks.iloc[0].entry_time == pd.Timedelta(hours=6)


def test_hvopcr_rejects_oi_build_small_purge_or_low_volatility():
    oi_build = frame()
    oi_build.loc[1, "oi_change"] = 0.02
    assert support.clock(oi_build).empty
    small_purge = frame()
    small_purge.loc[1, "oi_change"] = -0.005
    assert support.clock(small_purge).empty
    low_volatility = frame()
    low_volatility.loc[1, "dvol_close"] = 50.0
    assert support.clock(low_volatility).empty


def test_hvopcr_direction_flip_is_diagnostic_only_variant():
    primary = support.clock(frame())
    flipped = support.clock(frame(), "direction_flip")
    assert len(primary) == len(flipped) == 1
    assert primary.iloc[0].side == -flipped.iloc[0].side
    assert flipped.iloc[0].control == "direction_flip"
