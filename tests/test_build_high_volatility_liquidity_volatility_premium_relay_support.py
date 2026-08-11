import numpy as np
import pandas as pd

from training import build_high_volatility_liquidity_volatility_premium_relay_support as support


def test_prior_rank_excludes_current():
    ranks = support.prior_rank(pd.Series(np.arange(121, dtype=float)))
    assert np.isnan(ranks.iloc[119])
    assert ranks.iloc[120] == 1.0


def panel():
    return pd.DataFrame(
        {
            "source_valid": [True] * 6,
            "innovation": [-1.0, 1.0, 0.2, -0.2, -1.0, 1.0],
            "mean_amihud_innovation": [1.0, -1.0, 0.2, -0.2, -1.0, 1.0],
            "innovation_rank": [0.1, 0.9, 0.6, 0.4, 0.1, 0.9],
            "mean_amihud_innovation_rank": [0.9, 0.1, 0.6, 0.4, 0.1, 0.9],
            "variation_rank": [0.8] * 6,
            "feature_available_time": pd.date_range("2024-01-02", periods=6, freq="1d", tz="UTC"),
        }
    )


def test_primary_and_controls_are_outcome_blind():
    activity, side, _ = support.active(panel())
    assert activity.tolist() == [True, True, False, False, True, True]
    assert side[activity].tolist() == [-1, 1, -1, 1]
    changed = panel()
    changed.loc[1, "variation_rank"] = 0.4
    assert support.active(changed, "no_variation_gate")[0].iloc[1]
    raw_activity, raw_side, _ = support.active(panel(), "daily_mean_amihud_innovation")
    assert raw_activity.tolist() == [True, True, False, False, True, True]
    assert raw_side[raw_activity].tolist() == [1, -1, -1, 1]
    forced_activity, forced_side, _ = support.active(panel(), "forced_long")
    assert forced_activity.equals(activity)
    assert forced_side[forced_activity].eq(1).all()


def test_query_contains_no_outcomes():
    assert "quote_asset_volume" in support.QUERY
    assert "funding" not in support.QUERY.lower()
    assert "entry" not in support.QUERY.lower()
