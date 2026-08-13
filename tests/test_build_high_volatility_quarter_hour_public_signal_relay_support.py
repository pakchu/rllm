import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_quarter_hour_public_signal_relay_support as support


def _bars15(count: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(20260813)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, count)))
    spread = rng.uniform(0.1, 1.0, count)
    return pd.DataFrame(
        {
            "open": close * np.exp(rng.normal(0, 0.0005, count)),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.uniform(10, 100, count),
        },
        index=pd.date_range("2024-01-01", periods=count, freq="15min", tz="UTC"),
    )


def test_ti28_count_and_bollinger_block_are_non_degenerate() -> None:
    indicators = support.technical_indicators(_bars15())
    assert indicators.shape[1] == 28
    names = [name for name in indicators if name.startswith("bollinger_")]
    finite = indicators[names].dropna()
    assert names == [
        "bollinger_lower_distance",
        "bollinger_middle_distance",
        "bollinger_upper_distance",
        "bollinger_width",
    ]
    assert np.linalg.matrix_rank(finite.to_numpy()) == 4


def test_rolling_components_exclude_current_response() -> None:
    rng = np.random.default_rng(7)
    index = pd.RangeIndex(100)
    response = pd.Series(rng.normal(size=100), index=index)
    indicators = pd.DataFrame(rng.normal(size=(100, 3)), index=index)
    first = support.rolling_components(response, indicators, lag_count=2, lookback=40, minimum=20)
    changed = response.copy(); changed.iloc[50] = 1_000_000
    second = support.rolling_components(changed, indicators, lag_count=2, lookback=40, minimum=20)
    for left, right in zip(first, second):
        assert left.iloc[50] == pytest.approx(right.iloc[50])
    assert first[0].iloc[51] != pytest.approx(second[0].iloc[51])


def test_component_allocation_excludes_intercept_and_lag_block() -> None:
    rng = np.random.default_rng(11)
    index = pd.RangeIndex(140)
    indicator = pd.DataFrame({"public": rng.normal(size=140)}, index=index)
    response = pd.Series(np.nan, index=index)
    response.iloc[:2] = [0.2, -0.1]
    for position in range(2, 140):
        response.iloc[position] = 0.7 + 0.4 * response.iloc[position - 1] - 0.2 * response.iloc[position - 2] + 1.5 * indicator.iloc[position, 0]
    public, lagged, intercept = support.rolling_components(response, indicator, lag_count=2, lookback=80, minimum=40)
    position = public.last_valid_index(); assert position is not None
    fitted = public.at[position] + lagged.at[position] + intercept.at[position]
    assert public.at[position] == pytest.approx(1.5 * indicator.iloc[position, 0], rel=1e-5)
    assert fitted == pytest.approx(response.at[position], rel=1e-5)


def test_exact_minute_grid_is_required() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    times = pd.date_range(start, periods=100, freq="1min")
    bars = pd.DataFrame({"ts": times, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0, "taker_buy_base": 5.0}).drop(index=20)
    with pytest.raises(RuntimeError, match="exact requested one-minute grid"):
        support.validate_bars(bars, start, times[-1] + pd.Timedelta(minutes=1))


def _features() -> pd.DataFrame:
    times = pd.date_range("2024-07-01", periods=8, freq="15min", tz="UTC")
    return pd.DataFrame({
        "decision_time": times, "source_valid": True,
        "public_component": [0.1, 0.2, 0.3, -0.2, -0.4, -0.5, 0.2, 0.3],
        "lagged_component": [-0.1] * 8, "ols_intercept": [0.01] * 8,
        "latest_public_component": [-0.2] * 8, "model_valid": True, "latest_model_valid": True,
        "public_strength_rank": [0.96, 0.97, 0.8, 0.98, 0.99, 0.7, 0.97, 0.98],
        "lagged_strength_rank": [0.96] * 8, "intercept_strength_rank": [0.96] * 8,
        "latest_public_strength_rank": [0.96] * 8, "realized_variation": 0.1,
        "variation_rank": 0.8,
    })


def test_onset_and_control_direction_are_frozen() -> None:
    active, side = support.active_and_side(_features())
    assert active.tolist() == [True, False, False, True, False, False, True, False]
    assert side[active].tolist() == [1, -1, 1]
    latest_active, latest_side = support.active_and_side(_features(), "latest_indicator_bar_included")
    assert latest_active.iloc[0]
    assert latest_side[latest_active].eq(-1).all()


def test_support_gate_boundaries() -> None:
    passing = {split: {"events": support.MINIMUM[split], "longs": 1, "shorts": 1, "minority_side_share": 0.2, "max_month_share": 0.45} for split in support.SPLITS}
    assert all(support.support_checks(passing).values())
    passing["train"]["events"] = 7
    assert support.support_checks(passing)["train_minimum_events"] is False


def test_preregistration_hash_is_pinned() -> None:
    assert support.sha256(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
