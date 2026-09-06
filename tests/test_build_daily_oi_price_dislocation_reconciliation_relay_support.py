import pandas as pd

from training import build_daily_oi_price_dislocation_reconciliation_relay_support as support


def frame() -> pd.DataFrame:
    decision = pd.Timestamp("2024-07-02T00:00:00Z")
    return pd.DataFrame(
        {
            "decision_time": [decision],
            "source_valid": [True],
            "price_return": [-0.04],
            "oi_return": [0.03],
            "realized_variation": [0.06],
            "realized_variation_rank": [0.80],
            "displacement": [0.07],
            "displacement_rank": [0.75],
            "oi_current_time": [decision],
            "oi_prior_time": [decision - pd.Timedelta(hours=24)],
        }
    )


def test_dopdr_daily_opposition_follows_oi_direction():
    clocks = support.clock(frame())
    assert len(clocks) == 1
    assert clocks.iloc[0].side == 1
    assert clocks.iloc[0].entry_time == pd.Timestamp("2024-07-02T00:05:00Z")
    assert clocks.iloc[0].exit_time - clocks.iloc[0].entry_time == pd.Timedelta(hours=12)


def test_dopdr_rejects_same_direction_or_low_rank():
    same = frame()
    same.loc[0, "price_return"] = 0.04
    assert support.clock(same).empty
    low_vol = frame()
    low_vol.loc[0, "realized_variation_rank"] = 0.64
    assert support.clock(low_vol).empty
    low_displacement = frame()
    low_displacement.loc[0, "displacement_rank"] = 0.64
    assert support.clock(low_displacement).empty


def test_dopdr_controls_are_diagnostic_only_variants():
    primary = support.clock(frame())
    flipped = support.clock(frame(), "direction_flip")
    assert len(primary) == len(flipped) == 1
    assert primary.iloc[0].side == -flipped.iloc[0].side
    assert flipped.iloc[0].control == "direction_flip"
    same = frame()
    same.loc[0, "price_return"] = 0.04
    assert len(support.clock(same, "same_direction_only")) == 1


def test_strict_prior_midrank_excludes_current_observation():
    values = pd.Series([float(i) for i in range(127)])
    ranks = support.strict_prior_midrank(values, lookback=180, minimum=126)
    assert ranks.iloc[:126].isna().all()
    assert ranks.iloc[126] == 1.0
