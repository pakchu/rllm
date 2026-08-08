import pandas as pd

from training import build_cboe_volatility_surface_regime_crossing_relay_support as support


def _frame(surface, previous, vix=0.8):
    dates = pd.to_datetime(["2024-06-03", "2024-06-04", "2024-06-05"])
    return pd.DataFrame({
        "observation_date": dates, "surface": [previous, surface, 0.5], "previous_surface": [float("nan"), previous, surface],
        "tail": [previous, surface, 0.5], "previous_tail": [float("nan"), previous, surface], "term": [previous, surface, 0.5],
        "previous_term": [float("nan"), previous, surface], "vix_rank": [vix] * 3, "skew_rank": [0.5] * 3,
        "vvix_relative_rank": [0.5] * 3, "front_rank": [0.5] * 3, "broad_rank": [0.5] * 3,
    })


def test_cvsrc_crossing_direction_and_next_source_session_entry():
    short = support.clock(_frame(0.80, 0.40))
    long = support.clock(_frame(0.20, 0.60))
    assert len(short) == len(long) == 1
    assert short.iloc[0].side == -1 and long.iloc[0].side == 1
    assert short.iloc[0].entry_time == pd.Timestamp("2024-06-05T13:35:00Z")
    assert short.iloc[0].exit_time - short.iloc[0].entry_time == pd.Timedelta(hours=24)


def test_cvsrc_requires_elevated_vix_except_frozen_control():
    frame = _frame(0.80, 0.40, vix=0.59)
    assert support.clock(frame).empty
    assert len(support.clock(frame, "no_vix_high")) == 1


def test_strict_prior_midrank_excludes_current_value():
    ranks = support.strict_prior_midrank(pd.Series([1.0, 2.0, 3.0]), lookback=2, minimum=2)
    assert ranks.iloc[:2].isna().all()
    assert ranks.iloc[2] == 1.0


def test_frozen_source_features_are_finite_after_warmup():
    frame = support.features()
    assert frame.loc[126:, ["tail", "term", "surface", "vix_rank"]].notna().all().all()
