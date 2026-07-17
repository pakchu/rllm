from __future__ import annotations

import numpy as np
import pandas as pd

from training.build_cross_collateral_basis_snapback_support import (
    _bucket_counts,
    build_signal_features,
    candidate_events,
    clock_overlap,
    support_passes,
)


def _frame(rows: int = 18) -> pd.DataFrame:
    time = pd.date_range("2021-01-01", periods=rows, freq="5min", tz="UTC")
    log_wedge = np.resize(np.array([-0.002, 0.001, 0.003, -0.001]), rows)
    log_wedge[rows - 5] = 0.04
    return pd.DataFrame(
        {
            "open_time": time,
            "available_time": time + pd.Timedelta(minutes=5),
            "um_close": np.exp(log_wedge) * 100.0,
            "um_ohlc_valid": True,
            "cm_close": 100.0,
            "cm_ohlc_valid": True,
            "source_complete": True,
            "delivery_time": pd.Timestamp("2021-03-26 08:00", tz="UTC"),
            "contract_segment": "20210326",
        }
    )


def test_signal_features_are_strictly_prior_and_reset_by_segment() -> None:
    frame = _frame()
    features = build_signal_features(frame, lookback_bars=4, minimum_prior_bars=2)
    changed = frame.copy()
    changed.loc[10:, "um_close"] *= 5.0
    changed_features = build_signal_features(changed, lookback_bars=4, minimum_prior_bars=2)
    pd.testing.assert_series_equal(
        features.loc[:9, "center"],
        changed_features.loc[:9, "center"],
    )
    split = frame.copy()
    split.loc[9:, "contract_segment"] = "20210625"
    reset = build_signal_features(split, lookback_bars=4, minimum_prior_bars=2)
    assert reset.loc[9:10, "center"].isna().all()


def test_candidate_clock_uses_t_plus_10_and_full_reservation() -> None:
    features = build_signal_features(_frame(), lookback_bars=4, minimum_prior_bars=2)
    events = candidate_events(
        features,
        1.0,
        dislocation_floor=0.01,
        maximum_hold_bars=2,
    )
    assert len(events) == 1
    row = events.iloc[0]
    assert row["entry_time"] == row["open_time"] + pd.Timedelta(minutes=10)
    assert row["maximum_exit_time"] == row["entry_time"] + pd.Timedelta(minutes=10)


def test_support_gate_checks_every_frozen_bucket() -> None:
    times: list[pd.Timestamp] = []
    for year in (2021, 2022):
        for month in range(1, 13):
            for day in range(1, 7):
                times.append(pd.Timestamp(year, month, day, tz="UTC"))
    events = pd.DataFrame(
        {
            "entry_time": times,
            "zscore": np.resize(np.array([-2.0, 2.0]), len(times)),
        }
    )
    counts = _bucket_counts(events)
    passed, failures = support_passes(counts)
    assert passed, failures
    passed, failures = support_passes(_bucket_counts(events.iloc[:-50]))
    assert not passed
    assert failures


def test_clock_overlap_uses_ccbs_denominator_and_day_union() -> None:
    ccbs = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2023-01-01 00:00Z", "2023-01-02 00:00Z", "2023-01-03 00:00Z"]
            )
        }
    )
    anchor = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2023-01-01 00:00Z", "2023-01-01 01:00Z", "2023-01-04 00:00Z"]
            )
        }
    )
    result = clock_overlap(ccbs, anchor)
    assert result["exact_5m_overlap_share_of_ccbs"] == 1 / 3
    assert result["entry_day_jaccard"] == 1 / 4
