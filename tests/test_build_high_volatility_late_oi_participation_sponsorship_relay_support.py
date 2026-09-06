import numpy as np
import pandas as pd

from training import (
    build_high_volatility_late_oi_participation_sponsorship_relay_support as support,
)


def test_strict_prior_midrank_excludes_current() -> None:
    rank = support.strict_prior_midrank(
        pd.Series([1.0, 2.0, 3.0, 2.0]), lookback=2, minimum=2
    )
    assert np.isnan(rank.iloc[:2]).all()
    assert rank.iloc[2] == 1.0
    assert rank.iloc[3] == 0.25


def feature_frame() -> pd.DataFrame:
    decisions = pd.date_range("2024-01-01", periods=5, freq="8h", tz="UTC")
    return pd.DataFrame(
        {
            "decision_time": decisions,
            "feature_available_time": decisions,
            "source_valid": [True] * 5,
            "completed_return": [0.02, 0.01, -0.02, -0.01, 0.03],
            "realized_variation": [0.1] * 5,
            "gross_oi_activity": [0.1] * 5,
            "late_oi_activity_share": [0.4] * 5,
            "gross_oi_activity_rank": [0.7, 0.7, 0.7, 0.7, 0.7],
            "late_oi_activity_share_rank": [0.8, 0.8, 0.7, 0.8, 0.8],
            "variation_rank": [0.7, 0.7, 0.7, 0.7, 0.6],
        }
    )


def test_primary_uses_false_to_true_onset_and_frozen_gates() -> None:
    data = feature_frame()
    active, side, _ = support.conditions(data)
    assert active.tolist() == [False, False, False, True, False]
    assert side.tolist() == [1, 1, -1, -1, 1]
    assert support.conditions(data, "no_late_oi_activity_share_gate")[0].tolist() == [
        False,
        False,
        False,
        False,
        False,
    ]
    assert support.conditions(data, "direction_flip")[1].tolist() == [-1, -1, 1, 1, -1]
    assert support.conditions(data, "forced_long")[1].tolist() == [1, 1, 1, 1, 1]


def test_clock_enters_after_five_minutes_and_holds_eight_hours() -> None:
    clock = support.build_clock(feature_frame())
    assert len(clock) == 1
    assert clock.iloc[0]["entry_time"] == pd.Timestamp("2024-01-02T00:05:00Z")
    assert clock.iloc[0]["exit_time"] == pd.Timestamp("2024-01-02T08:05:00Z")
    assert clock.iloc[0]["side"] == -1


def test_queries_are_source_only_and_preregistration_is_hash_bound() -> None:
    combined = support.BAR_QUERY + support.OI_QUERY
    assert "bars_binance" in combined
    assert "open_interest_binance" in combined
    assert "funding_rates_binance" not in combined
    assert support.PREREG_SHA == (
        "73b7020c09b116d072c7b237f300ee5787a6556bcdd444354149529e2346f80b"
    )
