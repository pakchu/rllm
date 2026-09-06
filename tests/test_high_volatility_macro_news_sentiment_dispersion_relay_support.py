import numpy as np
import pandas as pd

from training import build_high_volatility_macro_news_sentiment_dispersion_relay_support as support


def test_rank_excludes_current_and_requires_180_prior_values():
    ranked = support.rank(pd.Series(np.arange(181, dtype=float)))
    assert ranked.iloc[:180].isna().all()
    assert ranked.iloc[180] == 1.0


def test_score_sentiment_uses_two_exact_population_dispersion_windows():
    dates = pd.date_range("2023-01-01", periods=15, freq="D", tz="UTC")
    values = np.array([0, 1, 0, 1, 0, 1, 0, 0, 2, 0, 2, 0, 2, 0, 4], dtype=float)
    scored = support.score_sentiment(
        pd.DataFrame({"date": dates, "news_sentiment": values})
    )
    row = scored.iloc[13]
    assert row["current_dispersion"] == np.std(values[7:14], ddof=0)
    assert row["prior_dispersion"] == np.std(values[0:7], ddof=0)
    assert row["dispersion_change"] == np.log(
        np.std(values[7:14], ddof=0) / np.std(values[0:7], ddof=0)
    )
    assert row["decision_time"] == dates[13] + pd.Timedelta(days=8)


def test_missing_exact_calendar_day_invalidates_dispersion():
    dates = pd.date_range("2023-01-01", periods=14, freq="D", tz="UTC").delete(5)
    values = np.arange(len(dates), dtype=float)
    scored = support.score_sentiment(
        pd.DataFrame({"date": dates, "news_sentiment": values})
    )
    assert not bool(scored.iloc[-1]["dispersion_valid"])


def _states(**updates):
    row = {
        "source_day": pd.Timestamp("2023-06-23T00:00:00Z"),
        "decision_time": pd.Timestamp("2023-07-01T00:00:00Z"),
        "dispersion_valid": True,
        "current_dispersion": 2.0,
        "prior_dispersion": 1.0,
        "dispersion_change": np.log(2.0),
        "signed_tone_change": 0.1,
        "btc_variation": 0.1,
        "btc_variation_rank": 0.8,
    }
    row.update(updates)
    return pd.DataFrame([row])


def test_primary_shorts_rising_dispersion_and_controls_are_diagnostic():
    states = _states()
    primary = support.build_clock(states)
    assert primary["side"].tolist() == [-1]
    assert primary["entry_time"].iloc[0] == pd.Timestamp("2023-07-01T00:05:00Z")
    assert support.build_clock(states, "direction_flip")["side"].tolist() == [1]
    assert support.build_clock(states, "same_clock_forced_long")["side"].tolist() == [1]
    assert support.build_clock(states, "signed_tone_change")["side"].tolist() == [1]


def test_variation_gate_is_removed_only_for_named_control():
    states = _states(btc_variation_rank=0.2)
    assert support.build_clock(states).empty
    assert support.build_clock(states, "no_btc_variation_gate")["side"].tolist() == [-1]


def test_one_week_stale_control_uses_old_dispersion_and_current_clock():
    old = _states(
        source_day=pd.Timestamp("2023-06-16T00:00:00Z"),
        decision_time=pd.Timestamp("2023-06-24T00:00:00Z"),
        dispersion_change=-0.2,
    )
    current = _states(dispersion_change=0.3)
    states = pd.concat([old, current], ignore_index=True)
    stale = support.build_clock(states, "one_week_stale_dispersion")
    assert stale["source_day"].tolist() == [pd.Timestamp("2023-06-23T00:00:00Z")]
    assert stale["feature_source_day"].tolist() == [pd.Timestamp("2023-06-16T00:00:00Z")]
    assert stale["side"].tolist() == [1]
