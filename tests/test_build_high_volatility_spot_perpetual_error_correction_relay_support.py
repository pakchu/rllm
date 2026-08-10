import numpy as np
import pandas as pd

from training import build_high_volatility_spot_perpetual_error_correction_relay_support as support


def test_strict_prior_midrank_excludes_current_value():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    ranks = support.strict_prior_midrank(values, lookback=3, minimum=2)
    assert np.isnan(ranks.iloc[0]) and np.isnan(ranks.iloc[1])
    assert ranks.iloc[2] == 1.0
    assert ranks.iloc[3] == 1.0


def test_error_correction_recovers_stable_spot_follower():
    rng = np.random.default_rng(7)
    spot = np.empty(2016)
    perpetual = np.empty(2016)
    spot[0] = perpetual[0] = 10.0
    for index in range(1, 2016):
        innovation = rng.normal(0.0, 0.0004)
        spread = perpetual[index - 1] - spot[index - 1]
        spot[index] = spot[index - 1] + 0.12 * spread + innovation + rng.normal(0.0, 0.00005)
        perpetual[index] = perpetual[index - 1] - 0.03 * spread + innovation
    alpha_spot, alpha_perpetual, share = support.fit_error_correction(spot, perpetual)
    assert alpha_spot > 0.0
    assert alpha_perpetual < 0.0
    assert share >= 0.60


def test_empty_clock_has_frozen_schema():
    states = pd.DataFrame(
        {
            "decision_time": pd.to_datetime([], utc=True),
            "source_valid": pd.Series([], dtype=bool),
            "model_valid": pd.Series([], dtype=bool),
            "perpetual_leadership_share": pd.Series([], dtype=float),
            "perpetual_hour_return": pd.Series([], dtype=float),
            "lead_innovation": pd.Series([], dtype=float),
            "innovation_rank": pd.Series([], dtype=float),
            "variation_rank": pd.Series([], dtype=float),
        }
    )
    clock = support.make_clock(states)
    assert tuple(clock.columns) == support.CLOCK_COLUMNS
    assert clock.empty
