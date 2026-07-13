import numpy as np
import pandas as pd

from training import search_rg_domain_wall_relaxation_alpha as rg


def _market(n: int = 1200):
    dates = pd.Series(pd.date_range("2020-01-01", periods=n, freq="5min"))
    close = np.exp(np.linspace(4.0, 4.2, n) + 0.01 * np.sin(np.arange(n) / 20))
    return pd.DataFrame({"close": close}), dates


def test_directional_field_is_prefix_independent():
    market, _ = _market()
    log_price = np.log(market["close"])
    first, _ = rg.directional_field(log_price, log_price.diff(), 24)
    changed = log_price.copy()
    changed.iloc[800:] += 5.0
    second, _ = rg.directional_field(changed, changed.diff(), 24)
    assert np.allclose(first[:800], second[:800], equal_nan=True)


def test_rg_state_only_uses_price_and_completed_dates():
    market, dates = _market()
    state = rg.build_rg_state(market, dates, base_scale=24)
    assert len(state) == len(market)
    assert state.loc[~dates.dt.minute.eq(55), "candidate"].sum() == 0


def test_fit_threshold_ignores_selection_values(monkeypatch):
    dates = pd.Series(pd.date_range("2022-12-31", periods=576, freq="5min"))
    monkeypatch.setitem(rg.WINDOWS, "fit", ("2022-12-31", "2023-01-01"))
    score = np.arange(len(dates), dtype=float)
    first = rg.fit_threshold(score, dates, 0.8)
    score[dates >= pd.Timestamp("2023-01-01")] = 1e9
    second = rg.fit_threshold(score, dates, 0.8)
    assert first == second


def test_masks_follow_coarse_fixed_point():
    state = pd.DataFrame(
        {
            "score": [0.1, 2.0, 3.0],
            "candidate": [True, True, True],
            "coarse_side": [1, 1, -1],
        }
    )
    long_active, short_active = rg.masks(state, 1.0)
    assert long_active.tolist() == [False, True, False]
    assert short_active.tolist() == [False, False, True]


def test_flip_is_exact_direction_swap():
    state = pd.DataFrame(
        {"score": [2.0, 3.0], "candidate": [True, True], "coarse_side": [1, -1]}
    )
    long_active, short_active = rg.masks(state, 1.0)
    flip_long, flip_short = rg.masks(state, 1.0, flip=True)
    assert np.array_equal(long_active, flip_short)
    assert np.array_equal(short_active, flip_long)


def test_loader_keeps_2024_sealed():
    _, dates = rg.load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")


def test_admission_requires_selection_support():
    base = {"return_pct": 1.0, "ratio": 3.1, "trades": 30, "longs": 15, "shorts": 15}
    stats = {name: dict(base) for name in rg.WINDOWS}
    assert rg.admission(stats)
    stats["select_2023"]["trades"] = 19
    assert not rg.admission(stats)
