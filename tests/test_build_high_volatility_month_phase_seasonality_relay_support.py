import numpy as np
import pandas as pd

from training import build_high_volatility_month_phase_seasonality_relay_support as support


def test_rank_excludes_current():
    ranked = support.strict_prior_midrank(pd.Series(np.arange(181, dtype=float)))
    assert ranked.iloc[:180].isna().all() and ranked.iloc[180] == 1.0


def test_frozen_phase_side_and_timing():
    states = pd.DataFrame({"decision_time": pd.to_datetime(["2023-07-01T00:00:00Z", "2023-07-04T00:00:00Z"]), "day_of_month": [1, 4], "btc_variation": [1.0, 1.0], "btc_variation_rank": [0.8, 0.8]})
    model = {"selected": [{"day_of_month": 1, "side": 1, "fit_mean_log_return": 0.01}, {"day_of_month": 4, "side": -1, "fit_mean_log_return": -0.01}]}
    clock = support.build_clock(states, model)
    assert clock.side.tolist() == [1, -1]
    assert (clock.exit_time - clock.entry_time).eq(pd.Timedelta(hours=12)).all()
