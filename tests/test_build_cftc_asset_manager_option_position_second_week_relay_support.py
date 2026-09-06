import numpy as np
import pandas as pd

from training import build_cftc_asset_manager_option_position_second_week_relay_support as support


def test_strict_prior_midrank_excludes_current_and_uses_midrank():
    values = pd.Series([1.0, 2.0, 2.0, 3.0])
    ranked = support.strict_prior_midrank(values, lookback=3, minimum=2)
    assert np.isnan(ranked.iloc[:2]).all()
    assert ranked.iloc[2] == 0.75
    assert ranked.iloc[3] == 1.0


def test_isolated_option_change_and_second_week_clock():
    dates = pd.date_range("2023-07-04", periods=4, freq="7D", tz="UTC")
    cftc = pd.DataFrame({"report_date": dates, "fut_net": [10, 11, 13, 12], "com_net": [12, 15, 18, 14]})
    feature_times = dates + pd.Timedelta(days=14, minutes=5)
    variation = pd.DataFrame({"feature_available_time": feature_times, "btc_variation": [1.0] * 4, "btc_variation_rank": [0.7] * 4})
    states = support.score_states(cftc, variation)
    assert states.isolated_option_net.tolist() == [2, 4, 5, 2]
    assert np.isnan(states.position_change.iloc[0])
    assert states.position_change.iloc[1:].tolist() == [2, 1, -3]
    clock = support.build_clock(states)
    assert clock.entry_time.tolist() == feature_times[1:].tolist()
    assert clock.side.tolist() == [1, 1, -1]
    assert (clock.exit_time - clock.entry_time).eq(pd.Timedelta(hours=168)).all()


def test_source_surface_is_outcome_blind():
    forbidden = {"pnl", "funding", "execution_price", "gross9"}
    assert not forbidden.intersection({name.lower() for name in support.COLUMNS})
