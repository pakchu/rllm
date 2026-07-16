from __future__ import annotations

import numpy as np
import pandas as pd

from training import check_cross_venue_temporal_torsion_support as support


def test_prior_eligible_count_excludes_current_row() -> None:
    eligible = pd.Series([True, False, True, True])
    counts = support.prior_eligible_counts(eligible, window=3)
    assert counts.tolist() == [0.0, 1.0, 1.0, 2.0]


def test_quarantine_is_current_plus_following_twenty_four() -> None:
    unavailable = pd.Series([False] * 35)
    unavailable.iloc[3] = True
    quarantined = support.quarantine(unavailable)
    assert quarantined.iloc[:3].eq(False).all()
    assert quarantined.iloc[3:28].eq(True).all()
    assert quarantined.iloc[28:].eq(False).all()


def test_route_support_uses_crossed_clock_and_source_side() -> None:
    frame = pd.DataFrame(
        {
            "source_quarantined": [0, 0],
            "spot_flow_fraction": [0.3, -0.2],
            "um_flow_fraction": [0.2, -0.3],
            "spot_log_return_5m": [0.001, -0.001],
            "um_log_return_5m": [0.002, -0.002],
            "spot_flow_time_centroid": [0.2, 0.8],
            "spot_return_time_centroid": [0.7, 0.2],
            "um_flow_time_centroid": [0.8, 0.2],
            "um_return_time_centroid": [0.3, 0.7],
        }
    )
    routes = support.route_support(frame)
    assert routes["spot_preload_um_echo"].tolist() == [True, False]
    assert routes["um_preload_spot_echo"].tolist() == [False, True]


def test_route_support_fails_closed_on_direction_disagreement() -> None:
    frame = pd.DataFrame(
        {
            "source_quarantined": [0],
            "spot_flow_fraction": [0.3],
            "um_flow_fraction": [-0.2],
            "spot_log_return_5m": [0.001],
            "um_log_return_5m": [0.002],
            "spot_flow_time_centroid": [0.2],
            "spot_return_time_centroid": [0.7],
            "um_flow_time_centroid": [0.8],
            "um_return_time_centroid": [0.3],
        }
    )
    routes = support.route_support(frame)
    assert not any(mask.any() for mask in routes.values())


def test_source_quality_enforces_monthly_gate() -> None:
    dates = pd.date_range("2020-01-01", periods=1000, freq="5min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "source_available": np.ones(1000, dtype=np.int8),
            "source_quarantined": np.zeros(1000, dtype=np.int8),
        }
    )
    frame.loc[:39, "source_available"] = 0
    frame.loc[:39, "source_quarantined"] = 1
    metrics = support.source_quality(frame)
    assert metrics["global_fraction"] == 0.04
    assert metrics["pass"] is False
