import numpy as np
import pandas as pd

from training import search_occupation_saddle_escape_alpha as saddle
from training.search_positioning_disagreement_alpha import _simulate_no_stop


def _market(n: int = 12_000):
    dates = pd.Series(pd.date_range("2020-01-01", periods=n, freq="5min"))
    price = np.exp(4.0 + 0.04 * np.sin(np.arange(n) / 500.0))
    quote = 1000.0 + 100.0 * np.cos(np.arange(n) / 31.0)
    return pd.DataFrame(
        {"close": price, "quote_asset_volume": quote, "high": price, "low": price, "open": price}
    ), dates


def test_profile_finds_deep_saddle_between_two_modes(monkeypatch):
    monkeypatch.setattr(saddle, "MIN_PROFILE_ROWS", 100)
    rng = np.random.default_rng(7)
    price = np.r_[rng.normal(-1.0, 0.1, 1000), rng.normal(1.0, 0.1, 1000)]
    quote = np.ones(len(price))
    profile = saddle.occupation_profile(price, quote)
    assert profile is not None
    assert profile.saddles
    assert all(item.ratio <= saddle.MAX_SADDLE_RATIO for item in profile.saddles)


def test_joint_profile_changes_when_price_volume_alignment_is_destroyed(monkeypatch):
    monkeypatch.setattr(saddle, "MIN_PROFILE_ROWS", 10)
    price = np.repeat(np.linspace(-2.0, 2.0, 80), 5)
    quote = np.linspace(1.0, 100.0, len(price))
    joint = saddle.occupation_profile(price, quote, mode="joint")
    reversed_volume = saddle.occupation_profile(price, quote, mode="reversed_volume")
    assert joint is not None and reversed_volume is not None
    assert not np.allclose(joint.density, reversed_volume.density)


def test_state_is_prefix_independent(monkeypatch):
    market, dates = _market()
    monkeypatch.setattr(saddle, "LOOKBACK_BARS", 576)
    monkeypatch.setattr(saddle, "MIN_PROFILE_ROWS", 288)
    first = saddle.build_saddle_state(market, dates)
    changed = market.copy()
    changed.loc[9000:, "close"] *= 5.0
    changed.loc[9000:, "quote_asset_volume"] *= 10.0
    second = saddle.build_saddle_state(changed, dates)
    for column in ("side", "barrier_depth", "saddle_count"):
        assert np.allclose(first.loc[:8999, column], second.loc[:8999, column], equal_nan=True)


def test_state_freezes_profile_at_utc_day_when_source_starts_midday(monkeypatch):
    market, dates = _market(300)
    dates = pd.Series(pd.date_range("2020-01-01 15:00:00", periods=len(market), freq="5min"))
    captured: list[np.ndarray] = []

    def capture_profile(log_price, quote_volume, *, mode="joint"):
        captured.append(np.asarray(log_price).copy())
        return saddle.FrozenProfile(
            edges=np.array([0.0, 10.0]),
            density=np.array([1.0]),
            saddles=(),
        )

    monkeypatch.setattr(saddle, "LOOKBACK_BARS", 12)
    monkeypatch.setattr(saddle, "occupation_profile", capture_profile)
    saddle.build_saddle_state(market, dates)

    expected_day_start = int(np.searchsorted(dates.to_numpy(), np.datetime64("2020-01-02")))
    expected = np.log(market["close"].to_numpy()[expected_day_start - 12 : expected_day_start])
    assert captured
    assert np.allclose(captured[0], expected)


def test_policy_direction_flip_is_exact():
    state = pd.DataFrame(
        {
            "side": [1, -1],
            "barrier_depth": [0.7, 0.8],
            "saddle_count": [1, 1],
            "decision": [True, True],
        }
    )
    long_active, short_active = saddle.policy_masks(state)
    flip_long, flip_short = saddle.policy_masks(state, flip=True)
    assert np.array_equal(long_active, flip_short)
    assert np.array_equal(short_active, flip_long)


def test_policy_requires_completed_decision_and_crossing():
    state = pd.DataFrame(
        {
            "side": [1, 1, 1],
            "barrier_depth": [0.7, np.nan, 0.8],
            "saddle_count": [1, 1, 1],
            "decision": [False, True, True],
        }
    )
    long_active, _ = saddle.policy_masks(state)
    assert long_active.tolist() == [False, False, True]


def test_boolean_lag_has_no_wraparound():
    values = np.array([True, False, True, True])
    assert saddle.lag_boolean(values, 2).tolist() == [False, False, True, False]


def test_simulator_wrapper_pins_execution_contract(monkeypatch):
    import training.search_dual_intrinsic_clock_alpha as clock

    captured = []

    def fake_simulator(*args, **kwargs):
        captured.append(kwargs)
        return {"window": kwargs["window"]}

    monkeypatch.setattr(clock, "_simulate_no_stop", fake_simulator)
    market, dates = _market(20)
    active = np.zeros(len(market), dtype=bool)
    extremes = (np.ones(len(market)), np.ones(len(market)))
    result = saddle.simulate(market, dates, active, active, 72, extremes)

    assert set(result) == set(clock.WINDOWS)
    assert len(captured) == len(clock.WINDOWS)
    assert all(call["hold_bars"] == 72 for call in captured)
    assert all(call["stride_bars"] == 1 for call in captured)
    assert all(call["leverage"] == 0.5 for call in captured)
    assert all(call["fee_rate"] == 0.0006 for call in captured)
    assert all(call["slippage_rate"] == 0.0 for call in captured)
    assert all(call["extremes"] is extremes for call in captured)
    assert all(call["windows"] is clock.WINDOWS for call in captured)


def test_canonical_simulator_enters_next_open_and_counts_strict_path():
    dates = pd.Series(pd.date_range("2023-01-01", periods=5, freq="5min"))
    market = pd.DataFrame(
        {
            "open": [100.0, 100.0, 110.0, 110.0, 110.0],
            "high": [100.0, 100.0, 110.0, 110.0, 110.0],
            "low": [100.0, 80.0, 110.0, 110.0, 110.0],
        }
    )
    long_active = np.array([True, False, False, False, False])
    result = _simulate_no_stop(
        market,
        dates,
        long_active,
        np.zeros(len(market), dtype=bool),
        window="sample",
        hold_bars=1,
        stride_bars=1,
        leverage=0.5,
        fee_rate=0.0,
        slippage_rate=0.0,
        windows={"sample": ("2023-01-01", "2023-01-02")},
    )

    assert result["trades"] == 1
    assert np.isclose(result["return_pct"], 5.0)
    assert np.isclose(result["strict_mdd_pct"], 10.0)


def test_loader_keeps_2024_sealed():
    _, dates = saddle.load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")
