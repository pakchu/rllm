import numpy as np
import pandas as pd

from training import build_high_volatility_dollar_factor_response_memory_relay_support as support


def synthetic_scores(n=24):
    decision = pd.date_range("2023-01-02 21:00", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "decision_time": decision,
        "entry_time": decision + pd.Timedelta(minutes=5),
        "exit_time": decision + pd.Timedelta(hours=12, minutes=5),
        "source_valid": True,
        "dollar_factor": np.where(np.arange(n) % 2, -1., 1.),
        "variation": 1., "variation_rank": .8, "valid_factor": True,
        "eligible": True, "signed_response": np.r_[np.ones(n - 1), -100.],
    })


def test_current_response_is_excluded_from_memory_and_side():
    state = support.causal_state(synthetic_scores())
    last = state.iloc[-1]
    assert last.active
    assert last.memory_count == 23
    assert last.memory_mean_response == 1
    assert last.side == -1


def test_unmatured_response_is_excluded():
    scores = synthetic_scores()
    scores.loc[22, "exit_time"] = scores.loc[23, "decision_time"] + pd.Timedelta(minutes=1)
    state = support.causal_state(scores)
    assert state.iloc[-1].memory_count == 22
    assert state.iloc[-1].memory_mean_response == 1


def test_controls_are_diagnostic_and_fixed_clock_sides_are_exact():
    scores = synthetic_scores()
    direct = support.causal_state(scores, "fixed_direct_factor")
    inverse = support.causal_state(scores, "fixed_inverse_factor")
    forced = support.causal_state(scores, "same_clock_forced_long")
    active = direct.active
    assert direct.loc[active, "side"].equals(np.sign(direct.loc[active, "dollar_factor"]).astype(int))
    assert inverse.loc[active, "side"].equals(-direct.loc[active, "side"])
    assert forced.loc[active, "side"].eq(1).all()


def test_stale_control_omits_latest_but_keeps_up_to_32_older_labels():
    scores = synthetic_scores(40)
    scores["signed_response"] = np.r_[np.ones(38), -10., -100.]
    state = support.causal_state(scores, "one_observation_stale_memory")
    last = state.iloc[-1]
    assert last.memory_count == 32
    assert last.memory_mean_response == 1


def test_strict_prior_variation_rank_excludes_current():
    decision = pd.date_range("2023-01-03 21:00", periods=61, freq="D", tz="UTC")
    times = pd.date_range(decision[0] - pd.Timedelta(hours=24), decision[-1] + pd.Timedelta(hours=13), freq="5min")
    market = pd.DataFrame({"date": times, "open": np.exp(np.arange(len(times)) * 1e-4)})
    factors = pd.DataFrame({"decision_time": decision, "source_valid": True, "dollar_factor": 1.})
    states = support.score_states(factors, market)
    assert states.variation_rank.iloc[:60].isna().all()
    assert np.isfinite(states.variation_rank.iloc[60])
