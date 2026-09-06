import numpy as np
import pandas as pd

from training import build_high_volatility_intraday_hour_reversal_support as support


def test_prior_rank_excludes_current():
    ranks = support.prior_rank(pd.Series(np.arange(121, dtype=float)))
    assert np.isnan(ranks.iloc[119])
    assert ranks.iloc[120] == 1.0


def panel():
    return pd.DataFrame(
        {
            "source_valid": [True] * 6,
            "predictor_return": [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
            "predictor_magnitude_rank": [0.5, 0.8, 0.8, 0.5, 0.8, 0.8],
            "variation_rank": [0.8] * 6,
            "feature_available_time": pd.date_range("2024-01-01 03:00", periods=6, freq="1d", tz="UTC"),
        }
    )


def test_primary_and_controls_are_outcome_blind():
    activity, side, _ = support.active(panel())
    assert activity.tolist() == [False, True, True, False, True, True]
    assert side[activity].tolist() == [-1, -1, 1, -1]
    changed = panel()
    changed.loc[1, "variation_rank"] = 0.4
    assert support.active(changed, "no_variation_gate")[0].iloc[1]
    forced_activity, forced_side, _ = support.active(panel(), "forced_long")
    assert forced_activity.equals(activity)
    assert forced_side[forced_activity].eq(1).all()


def test_clock_uses_frozen_target_hour():
    source = panel().iloc[[1]].copy()
    source["source_day"] = [pd.Timestamp("2024-01-02", tz="UTC")]
    source["predictor_magnitude"] = [1.0]
    source["realized_variation"] = [1.0]
    source["eligible"] = [True]
    clock = support.build_clock(source)
    assert clock.entry_time.iloc[0] == pd.Timestamp("2024-01-02 18:05", tz="UTC")
    assert clock.exit_time.iloc[0] == pd.Timestamp("2024-01-02 19:00", tz="UTC")
    assert clock.feature_available_time.iloc[0] < clock.entry_time.iloc[0]
