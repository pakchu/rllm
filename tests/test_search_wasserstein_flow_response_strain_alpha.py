import numpy as np
import pandas as pd

from training import search_wasserstein_flow_response_strain_alpha as wfrs


def _market(n: int = 2200) -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.Series(pd.date_range("2020-10-01", periods=n, freq="5min"))
    base = np.linspace(100.0, 110.0, n)
    market = pd.DataFrame(
        {
            "date": dates,
            "open": base,
            "close": base * (1.0 + 0.001 * np.sin(np.arange(n))),
            "quote_asset_volume": np.full(n, 100.0),
            "taker_buy_quote": 50.0 + 20.0 * np.sin(np.arange(n) / 3.0),
        }
    )
    return market, dates


def test_prior_volatility_excludes_current_bar(monkeypatch):
    market, dates = _market()
    monkeypatch.setattr(wfrs, "VOL_WINDOW", 20)
    monkeypatch.setattr(wfrs, "VOL_MIN_PERIODS", 10)
    monkeypatch.setitem(wfrs.WINDOWS, "fit", ("2020-10-01", "2021-01-01"))
    first, _ = wfrs.build_response_inputs(market, dates)
    changed = market.copy()
    changed.loc[100, "close"] *= 2.0
    second, _ = wfrs.build_response_inputs(changed, dates)
    assert first.loc[100, "prior_volatility"] == second.loc[100, "prior_volatility"]
    assert first.loc[101, "prior_volatility"] != second.loc[101, "prior_volatility"]


def test_fit_flow_tail_is_unchanged_by_selection_rows(monkeypatch):
    market, dates = _market()
    monkeypatch.setattr(wfrs, "VOL_WINDOW", 20)
    monkeypatch.setattr(wfrs, "VOL_MIN_PERIODS", 10)
    monkeypatch.setitem(wfrs.WINDOWS, "fit", ("2020-10-01", "2020-10-05"))
    _, first = wfrs.build_response_inputs(market, dates)
    changed = market.copy()
    changed.loc[dates >= pd.Timestamp("2020-10-05"), "taker_buy_quote"] = 100.0
    _, second = wfrs.build_response_inputs(changed, dates)
    assert first == second


def test_transport_score_detects_buy_side_dominance():
    plus = np.linspace(1.0, 2.0, 100)
    minus = np.linspace(0.0, 1.0, 100)
    result = wfrs.transport_components(plus, minus)
    assert result["score"] > 0.0
    assert np.isclose(result["w1"], 1.0)
    assert result["mean_only"] > 0.0


def test_transport_state_uses_only_window_prefix(monkeypatch):
    n = 180
    response = np.sin(np.arange(n) / 5.0)
    flow = np.tile(np.r_[-np.ones(6), np.ones(6)], 15)
    inputs = pd.DataFrame(
        {
            "response": response,
            "flow": flow,
            "decision": np.arange(n) % 12 == 11,
        }
    )
    monkeypatch.setattr(wfrs, "MIN_SIDE_OBSERVATIONS", 5)
    first = wfrs.build_transport_state(inputs, lookback=60, flow_tail=0.5)
    changed = inputs.copy()
    changed.loc[120:, "response"] = 999.0
    second = wfrs.build_transport_state(changed, lookback=60, flow_tail=0.5)
    assert np.allclose(first.loc[:119, "score"], second.loc[:119, "score"], equal_nan=True)


def test_policy_masks_are_next_open_signal_masks_not_shifted():
    score = np.array([np.nan, 2.0, -3.0, 0.0])
    decision = np.array([False, True, True, True])
    long_active, short_active = wfrs.policy_masks(score, decision, 1.0)
    assert long_active.tolist() == [False, True, False, False]
    assert short_active.tolist() == [False, False, True, False]


def test_onset_only_suppresses_persistent_hourly_state():
    score = np.zeros(36)
    decision = np.zeros(36, dtype=bool)
    decision[[11, 23, 35]] = True
    score[[11, 23, 35]] = 2.0
    long_active, _ = wfrs.policy_masks(score, decision, 1.0, onset_only=True)
    assert np.flatnonzero(long_active).tolist() == [11]


def test_loader_physically_stops_before_cutoff():
    market, dates = wfrs.load_pre2024()
    assert len(market) == len(dates)
    assert dates.max() < pd.Timestamp(wfrs.CUTOFF)
    assert dates.diff().dropna().eq(pd.Timedelta("5min")).all()


def test_admission_requires_fit_and_selection_ratio_three():
    template = {
        "return_pct": 10.0,
        "ratio": 3.1,
        "trades": 100,
        "longs": 50,
        "shorts": 50,
    }
    stats = {name: dict(template) for name in wfrs.WINDOWS}
    assert wfrs.admission(stats)
    stats["fit"]["ratio"] = 2.99
    assert not wfrs.admission(stats)
