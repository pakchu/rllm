import pandas as pd

from training import build_daily_cross_venue_participation_rotation_relay_support as support


def frame() -> pd.DataFrame:
    decision = pd.Timestamp("2024-07-02T00:00:00Z")
    return pd.DataFrame(
        {
            "decision_time": [decision],
            "source_valid": [True],
            "participation_z": [1.2],
            "rotation": [0.8],
            "absolute_rotation_rank": [0.75],
            "absolute_level_rank": [0.70],
            "realized_variation": [0.06],
            "realized_variation_rank": [0.80],
        }
    )


def test_dcvpr_follows_daily_participation_rotation():
    clocks = support.clock(frame())
    assert len(clocks) == 1
    assert clocks.iloc[0].side == 1
    assert clocks.iloc[0].entry_time == pd.Timestamp("2024-07-02T00:05:00Z")
    assert clocks.iloc[0].exit_time - clocks.iloc[0].entry_time == pd.Timedelta(hours=12)


def test_dcvpr_rejects_low_rotation_or_low_volatility_rank():
    low_rotation = frame()
    low_rotation.loc[0, "absolute_rotation_rank"] = 0.64
    assert support.clock(low_rotation).empty
    low_volatility = frame()
    low_volatility.loc[0, "realized_variation_rank"] = 0.64
    assert support.clock(low_volatility).empty


def test_dcvpr_direction_flip_is_diagnostic_only():
    primary = support.clock(frame())
    flipped = support.clock(frame(), "direction_flip")
    assert len(primary) == len(flipped) == 1
    assert primary.iloc[0].side == -flipped.iloc[0].side
    assert flipped.iloc[0].control == "direction_flip"


def test_strict_prior_midrank_excludes_current():
    values = pd.Series([float(i) for i in range(127)])
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[:126].isna().all()
    assert ranks.iloc[126] == 1.0
