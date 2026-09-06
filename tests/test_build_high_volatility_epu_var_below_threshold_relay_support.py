import numpy as np
import pandas as pd

from training import build_high_volatility_epu_var_below_threshold_relay_support as support


def test_strict_prior_midrank_excludes_current() -> None:
    result = support.strict_prior_midrank(pd.Series(np.arange(181, dtype=float)))
    assert result.iloc[:180].isna().all()
    assert result.iloc[180] == 1.0


def test_expanding_var_uses_current_completed_transition_and_forecasts_next() -> None:
    btc = np.arange(8, dtype=float)
    epu = np.arange(8, dtype=float) * 2.0 + np.array([0, 1, 0, 1, 0, 1, 0, 1])
    frame = pd.DataFrame({"btc_return": btc, "epu_change": epu})
    result = support.expanding_var_forecasts(frame, minimum=3)
    assert result.iloc[:3].isna().all()
    assert np.isfinite(result.iloc[3:]).all()


def test_forecast_dispersion_is_strictly_prior() -> None:
    forecasts = pd.Series([1.0, 2.0, 3.0, 100.0])
    result = support.expanding_forecast_dispersion(forecasts, minimum=3)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == np.std([1.0, 2.0, 3.0], ddof=1)


def _features(forecast: float, dispersion: float, variation_rank: float = 0.8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_time": [pd.Timestamp("2023-07-01T00:00:00Z")],
            "source_day": [pd.Timestamp("2023-06-29T00:00:00Z")],
            "epu": [100.0],
            "epu_change": [1.0],
            "btc_open": [30_000.0],
            "btc_return": [0.01],
            "btc_variation": [0.04],
            "btc_variation_rank": [variation_rank],
            "btc_forecast": [forecast],
            "forecast_dispersion": [dispersion],
            "state_valid": [True],
        }
    )


def test_primary_uses_forecast_sign_below_or_equal_threshold() -> None:
    assert support.build_clock(_features(-0.01, 0.01)).side.tolist() == [-1]
    assert support.build_clock(_features(0.01, 0.02)).side.tolist() == [1]


def test_primary_forces_long_above_threshold() -> None:
    clock = support.build_clock(_features(-0.03, 0.01))
    assert clock.side.tolist() == [1]
    assert clock.below_threshold.tolist() == [False]


def test_variation_gate_and_diagnostic_control() -> None:
    features = _features(-0.01, 0.02, variation_rank=0.2)
    assert support.build_clock(features).empty
    assert support.build_clock(features, "no_btc_variation_gate").side.tolist() == [-1]


def test_clock_is_causal_and_holds_exactly_one_day() -> None:
    clock = support.build_clock(_features(-0.01, 0.02))
    assert clock.entry_time.iloc[0] == pd.Timestamp("2023-07-01T00:05:00Z")
    assert clock.exit_time.iloc[0] - clock.entry_time.iloc[0] == pd.Timedelta(hours=24)
