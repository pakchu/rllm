import pandas as pd

from training import build_funding_settlement_volatility_ignition_breakout_relay_support as support


def frame() -> pd.DataFrame:
    settlement = pd.Timestamp("2024-07-01T08:00:00Z")
    return pd.DataFrame(
        {
            "settlement_time": [settlement - pd.Timedelta(hours=8), settlement],
            "decision_time": [settlement - pd.Timedelta(hours=7), settlement + pd.Timedelta(hours=1)],
            "signal_valid": [True, True],
            "funding_rate": [0.0001, -0.0001],
            "post_settlement_return_1h": [0.001, 0.03],
            "prior_abs_post_return_q60": [0.01, 0.01],
            "bvol_body": [-0.01, 0.02],
            "dvol_body": [-0.01, 0.03],
        }
    )


def test_fsvibr_follows_post_settlement_breakout_after_joint_ignition():
    clocks = support.clock(frame())
    assert len(clocks) == 1
    assert clocks.iloc[0].side == 1
    assert clocks.iloc[0].entry_time == pd.Timestamp("2024-07-01T09:05:00Z")
    assert clocks.iloc[0].exit_time - clocks.iloc[0].entry_time == pd.Timedelta(hours=6)


def test_fsvibr_rejects_single_venue_expansion_or_small_move():
    single_venue = frame()
    single_venue.loc[1, "dvol_body"] = -0.01
    assert support.clock(single_venue).empty
    assert len(support.clock(single_venue, "bvol_only_expansion")) == 1
    small_move = frame()
    small_move.loc[1, "post_settlement_return_1h"] = 0.005
    assert support.clock(small_move).empty


def test_fsvibr_direction_flip_preserves_clock_only():
    primary = support.clock(frame())
    flipped = support.clock(frame(), "direction_flip")
    assert len(primary) == len(flipped) == 1
    assert primary.iloc[0].side == -flipped.iloc[0].side
    assert primary.iloc[0].entry_time == flipped.iloc[0].entry_time
