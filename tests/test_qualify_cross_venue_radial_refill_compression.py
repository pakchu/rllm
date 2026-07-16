from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from training import qualify_cross_venue_radial_refill_compression as qualify


def test_lagged_quantile_is_prefix_invariant_and_excludes_current_row() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    baseline = qualify.lagged_quantile(values, 0.5, window=4, minimum=3)
    assert np.isnan(baseline.iloc[2])
    assert baseline.iloc[3] == 2.0
    changed = values.copy()
    changed.iloc[-1] = 1_000_000.0
    replay = qualify.lagged_quantile(changed, 0.5, window=4, minimum=3)
    pd.testing.assert_series_equal(baseline.iloc[:-1], replay.iloc[:-1])


def test_classify_requires_cross_venue_agreement_and_flattens_conflict() -> None:
    index = pd.RangeIndex(3)
    raw = {}
    thresholds = {}
    for venue in ("um", "cm"):
        for side in ("m", "p"):
            for metric in ("add", "withdraw", "net", "flicker"):
                value = 0.5 if metric == "flicker" else 2.0
                raw[(venue, side, metric)] = pd.Series([value, value, value], index=index)
                thresholds[(venue, side, metric)] = pd.Series(
                    [1.0, 1.0, 1.0], index=index
                )
    # Row 0: both bid venues pass, asks fail => long.
    for venue in ("um", "cm"):
        raw[(venue, "p", "add")].iloc[0] = 0.0
    # Row 1: one bid venue fails and asks fail => flat.
    raw[("cm", "m", "add")].iloc[1] = 0.0
    for venue in ("um", "cm"):
        raw[(venue, "p", "add")].iloc[1] = 0.0
    # Row 2: both bid and ask pass => conflict-flat.
    state = qualify.classify(raw, pd.Series([True, True, True]), thresholds)
    assert state["side"].tolist() == [1, 0, 0]
    assert state["conflict"].tolist() == [False, False, True]


def test_zero_or_nonfinite_threshold_fails_closed() -> None:
    raw = {}
    thresholds = {}
    for venue in ("um", "cm"):
        for side in ("m", "p"):
            for metric in ("add", "withdraw", "net", "flicker"):
                raw[(venue, side, metric)] = pd.Series([2.0])
                thresholds[(venue, side, metric)] = pd.Series([1.0])
    thresholds[("um", "m", "net")].iloc[0] = 0.0
    thresholds[("um", "p", "net")].iloc[0] = np.inf
    state = qualify.classify(raw, pd.Series([True]), thresholds)
    assert state.loc[0, "side"] == 0


def test_scheduler_enters_t_plus_two_and_allows_close_then_open() -> None:
    cfg = replace(qualify.Config(), hold_bars=3, entry_delay_bars=2)
    dates = pd.date_range("2023-01-01", periods=20, freq="5min")
    side = np.zeros(20, dtype=np.int8)
    side[[1, 4, 6]] = 1
    signal = pd.DataFrame(
        {
            "date": dates,
            "side": side,
            "branch": np.where(side, "bid_refill_compression", "none"),
        }
    )
    schedule = qualify.quarter_schedule(signal, cfg)
    assert schedule[["signal_position", "entry_position", "exit_position"]].values.tolist() == [
        [1, 3, 6],
        [4, 6, 9],
    ]


def test_support_summary_applies_concentration_and_side_gates() -> None:
    cfg = replace(
        qualify.Config(),
        minimum_events=4,
        minimum_half_events=2,
        minimum_quarter_events=1,
        minimum_side_share=0.25,
        maximum_month_share=1.0,
        maximum_quarter_share=0.25,
    )
    schedule = pd.DataFrame(
        {
            "quarter": ["q1", "q2", "q3", "q4"],
            "entry_date": [
                "2023-01-01",
                "2023-04-01",
                "2023-07-01",
                "2023-10-01",
            ],
            "side": [1, -1, 1, -1],
        }
    )
    result = qualify.support_summary(schedule, cfg)
    assert result["passes_support"] is True
    assert result["h1"] == result["h2"] == 2


def test_overlap_metrics_use_entry_clock_and_position_time() -> None:
    cfg = replace(
        qualify.Config(),
        exact_entry_jaccard_maximum=1.0,
        tolerant_bars=2,
        tolerant_candidate_match_share_maximum=1.0,
        position_time_jaccard_maximum=1.0,
    )
    current = pd.DataFrame(
        {"entry_position": [10, 30], "exit_position": [20, 40], "side": [1, -1]}
    )
    prior = pd.DataFrame(
        {"entry_position": [10, 32], "exit_position": [15, 37], "side": [1, 1]}
    )
    result = qualify.overlap_summary(current, prior, cfg)
    assert result["exact_entry_matches"] == 1
    assert result["tolerant_matches"] == 2
    assert result["tolerant_candidate_match_share"] == 1.0
    assert result["position_intersection_bars"] == 10
    assert result["position_union_bars"] == 20
    assert result["position_time_jaccard"] == 0.5
