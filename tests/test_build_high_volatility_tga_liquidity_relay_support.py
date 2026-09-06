import numpy as np
import pandas as pd

from training import build_high_volatility_tga_liquidity_relay_support as support


def test_rank_excludes_current():
    ranked = support.strict_prior_midrank(pd.Series(np.arange(181, dtype=float)))
    assert ranked.iloc[:180].isna().all()
    assert ranked.iloc[180] == 1.0


def test_primary_side_is_inverse_five_observation_change():
    dates = pd.date_range("2023-06-20", periods=12, freq="D")
    states = pd.DataFrame({"record_date": dates, "decision_time": dates.tz_localize("UTC") + pd.Timedelta(days=5), "source_valid": True, "tga_close_millions": np.arange(12), "change_1": 1.0, "change_5": 5.0, "change_10": 10.0, "btc_variation": 1.0, "btc_variation_rank": 0.8})
    clock = support.build_clock(states)
    assert set(clock.side) == {-1}
    assert (clock.exit_time - clock.entry_time).eq(pd.Timedelta(hours=24)).all()
