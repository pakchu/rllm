import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_vix_transfer_entropy_forecast_relay_support as support


def test_cboe_decision_uses_next_source_date_and_dst() -> None:
    vix = pd.DataFrame({"observation_date": ["2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12"], "VIX_close": [14.0, 15.0, 14.5, 16.0]})
    frame = support.cboe_decisions(vix)
    assert frame.iloc[0]["source_date"] == "2024-03-08"
    assert frame.iloc[0]["decision_time"] == pd.Timestamp("2024-03-11T13:35:00Z")
    assert frame.iloc[1]["decision_time"] == pd.Timestamp("2024-03-12T13:35:00Z")


def test_transition_forecast_never_uses_uncompleted_target() -> None:
    times = pd.date_range("2020-01-01", periods=12, freq="D", tz="UTC")
    frame = pd.DataFrame({"decision_time": times, "source_valid": True, "vix_state": [1] * 12, "btc_state": [1] * 12, "target_state": [1, -1] * 6, "target_completion_time": times + pd.Timedelta(days=1)})
    first = support.transition_forecasts(frame, lookback=10, minimum=4, cell_minimum=4, target_minimum=1)
    changed = frame.copy(); changed.loc[5, "target_state"] *= -1
    second = support.transition_forecasts(changed, lookback=10, minimum=4, cell_minimum=4, target_minimum=1)
    assert first.loc[5, "conditional_probability"] == pytest.approx(second.loc[5, "conditional_probability"])
    assert first.loc[6, "conditional_probability"] != pytest.approx(second.loc[6, "conditional_probability"])


def test_probability_and_cell_support_are_exact() -> None:
    times = pd.date_range("2020-01-01", periods=9, freq="D", tz="UTC")
    frame = pd.DataFrame({"decision_time": times, "source_valid": True, "vix_state": [1] * 9, "btc_state": [1] * 9, "target_state": [1, 1, 1, -1, 1, -1, 1, -1, np.nan], "target_completion_time": list(times[1:]) + [pd.NaT]})
    result = support.transition_forecasts(frame, lookback=8, minimum=4, cell_minimum=4, target_minimum=1)
    assert result.loc[8, "conditioning_count"] == 8
    assert result.loc[8, "conditional_probability"] == pytest.approx(5 / 8)
    assert result.loc[8, "forecast_strength"] == pytest.approx(0.125)


def _features() -> pd.DataFrame:
    times = pd.date_range("2024-07-01", periods=8, freq="D", tz="UTC")
    return pd.DataFrame({"source_date": times.strftime("%Y-%m-%d"), "decision_time": times, "source_valid": True, "vix_state": [1, -1] * 4, "btc_state": 1, "target_state": 1, "target_completion_time": times + pd.Timedelta(days=1), "conditioning_count": 40, "target_positive_count": 25, "target_negative_count": 15, "conditional_probability": [0.7, 0.7, 0.55, 0.3, 0.3, 0.55, 0.8, 0.8], "forecast_strength": [0.2, 0.2, 0.05, 0.2, 0.2, 0.05, 0.3, 0.3], "strength_rank": [0.8, 0.9, 0.5, 0.8, 0.9, 0.5, 0.8, 0.9], "unconditional_probability": 0.6, "unconditional_strength": 0.1, "unconditional_strength_rank": 0.8, "realized_variation": 0.1, "variation_rank": 0.8, "forecast_valid": True})


def test_onset_side_and_reservation() -> None:
    frame = _features(); active, side, _ = support.active_and_side(frame)
    assert active.tolist() == [True, False, False, True, False, False, True, False]
    assert side[active].tolist() == [1, -1, 1]
    clock = support.make_clock(frame)
    assert len(clock) == 3
    assert (clock["exit_time"] - clock["entry_time"]).eq(pd.Timedelta(hours=24)).all()


def test_support_gate_boundaries() -> None:
    passing = {split: {"events": support.MINIMUM[split], "longs": 1, "shorts": 1, "minority_side_share": 0.2, "max_month_share": 0.45} for split in support.SPLITS}
    assert all(support.checks(passing).values())
    passing["final"]["max_month_share"] = 0.5
    assert support.checks(passing)["final_month_concentration"] is False


def test_preregistration_hash_is_pinned() -> None:
    assert support.sha256(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
