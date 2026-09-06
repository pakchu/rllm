import numpy as np
import pandas as pd

from training import build_dual_volatility_index_upper_rejection_fade_support as support


def frame():
    return pd.DataFrame({"decision_time": pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC"), "source_valid": [True]*4, "btc_hour_return": [.02, -.02, .02, .02], "bvol_net_upper_rejection": [.6, .6, -.1, .6], "dvol_net_upper_rejection": [.5, .5, .5, .5], "joint_rejection": [.5, .5, -.1, .5], "joint_range": [.1]*4, "joint_rejection_rank": [.8, .8, .8, .8], "joint_range_rank": [.8, .4, .8, .8], "bvol_rejection_rank": [.8]*4, "dvol_rejection_rank": [.8]*4, "btc_absolute_return_rank": [.8, .8, .8, .5]})


def test_rank_is_strict_prior():
    ranked = support.strict_prior_midrank(pd.Series([1., 2., 3.]), lookback=2, minimum=2)
    assert np.isnan(ranked.iloc[:2]).all() and ranked.iloc[2] == 1.


def test_primary_and_controls():
    values = frame(); active, side = support.conditions(values)
    assert active.tolist() == [True, False, False, False]
    assert side.tolist() == [-1., 1., -1., -1.]
    assert support.conditions(values, "btc_shock_only")[0].tolist() == [True, True, True, False]
    assert support.conditions(values, "bvol_rejection_only")[0].tolist() == [True, False, False, False]
    assert support.conditions(values, "dvol_rejection_only")[0].tolist() == [True, False, True, False]
    assert support.conditions(values, "no_joint_range_floor")[0].tolist() == [True, True, False, False]
    assert support.conditions(values, "direction_flip")[1].tolist() == [1., -1., 1., 1.]
    assert support.conditions(values, "forced_long")[1].tolist() == [1.]*4


def test_clock_reserves_six_hours():
    clock = support.build_clock(frame())
    assert len(clock) == 1
    assert clock.iloc[0].entry_time == pd.Timestamp("2024-01-01T00:05:00Z")
    assert clock.iloc[0].exit_time == pd.Timestamp("2024-01-01T06:05:00Z")


def test_binding_and_sealed_outcomes():
    assert support.sha256(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    source = support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
