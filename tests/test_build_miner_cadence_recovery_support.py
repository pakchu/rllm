from __future__ import annotations

import numpy as np
import pandas as pd

from training import build_miner_cadence_recovery_support as support
from training import preregister_miner_cadence_recovery as prereg


def test_hash_change_requires_the_lagged_row_to_be_available() -> None:
    values = np.log(np.arange(1.0, 10.0))
    available = pd.date_range("2021-01-02", periods=9, freq="D").to_numpy()
    available[0] = available[7]
    result = support.causal_hash_change(values, available, 7)
    assert np.isnan(result[7])
    assert np.isfinite(result[8])


def test_cadence_reference_excludes_the_current_observation() -> None:
    blocks = np.full(31, 100.0)
    blocks[-1] = 400.0
    available = pd.date_range("2021-01-02", periods=31, freq="D").to_numpy()
    short, reference, gap = support.causal_cadence(
        np.log(blocks), available, short_days=3, reference_days=30
    )
    expected_short = np.mean(np.log([100.0, 100.0, 400.0]))
    assert np.isclose(short[-1], expected_short)
    assert np.isclose(reference[-1], np.log(100.0))
    assert np.isclose(gap[-1], expected_short - np.log(100.0))


def test_schedule_reserves_full_seven_day_hold() -> None:
    policy = prereg.Policy()
    entries = pd.to_datetime(["2022-01-01", "2022-01-03", "2022-01-09"])
    frame = pd.DataFrame(
        {
            "observation_date": entries - pd.Timedelta(days=1),
            "available_at": entries,
            "HashRate": [1.0, 1.0, 1.0],
            "BlkCnt": [144.0, 144.0, 144.0],
            "hash_change": [0.1, 0.1, 0.1],
            "hash_change_z": [0.1, 0.1, 0.1],
            "prior_hash_change_z": [-0.1, -0.1, -0.1],
            "recent_stress_min_z": [-1.2, -1.2, -1.2],
            "cadence_short": [1.0, 1.0, 1.0],
            "cadence_reference": [0.9, 0.9, 0.9],
            "cadence_gap": [0.1, 0.1, 0.1],
            "source_lag_days": [1.0, 1.0, 1.0],
            "hash_reference_count": [180, 180, 180],
            "event": [True, True, True],
        }
    )
    clock = support.schedule_clock(frame, policy)
    assert len(clock) == 2
    assert (clock["exit_date"] - clock["entry_date"] == pd.Timedelta(days=7)).all()


def test_support_summary_fails_closed_on_empty_clock() -> None:
    payload = prereg.manifest()
    clock = pd.DataFrame({"entry_date": pd.to_datetime([])})
    result = support.support_summary(clock, payload)
    assert result["passed"] is False
    assert result["maximum_single_month_share"] == 1.0
