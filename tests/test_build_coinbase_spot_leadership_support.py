from __future__ import annotations

import numpy as np
import pandas as pd

from training import build_coinbase_spot_leadership_support as support
from training.preregister_coinbase_spot_leadership_alpha import Policy


def test_lagged_robust_z_is_exact_and_strictly_prior() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 100.0, 4.0])
    z = support.lagged_exact_robust_zscore(
        values, window=4, minimum=3, block_rows=2
    )
    assert z.iloc[:3].isna().all()
    assert np.isclose(z.iloc[3], 98.0 / 1.4826)
    assert np.isclose(z.iloc[4], 1.5 / 1.4826)


def test_missing_quarantine_is_current_plus_next_twelve() -> None:
    complete = pd.Series([True] * 20)
    complete.iloc[2] = False
    quarantined = support.missing_quarantine(complete)
    assert quarantined.iloc[:2].eq(False).all()
    assert quarantined.iloc[2:15].eq(True).all()
    assert quarantined.iloc[15:].eq(False).all()


def test_policy_masks_follow_frozen_directional_rules() -> None:
    features = pd.DataFrame(
        {
            "ZR": [2.1, -2.1, 0.0],
            "ZP": [0.0, 0.0, 0.0],
            "ZV": [0.0, 0.0, 0.0],
            "ZCB": [1.1, -1.1, 0.0],
            "ZBN": [1.4, -1.4, 0.0],
            "source_quarantined": [0, 0, 1],
        }
    )
    long_policy = Policy("L", "relative_return_lead", 1, 1)
    short_policy = Policy("S", "relative_return_lead", -1, 1)
    assert support.policy_mask(features, long_policy).tolist() == [True, False, False]
    assert support.policy_mask(features, short_policy).tolist() == [False, True, False]


def test_nonoverlap_uses_entry_to_exit_clock_and_drops_tail() -> None:
    mask = np.ones(10, dtype=bool)
    assert support.schedule_nonoverlap(mask, 3).tolist() == [0, 3]
    assert support.schedule_nonoverlap(mask, 1).tolist() == list(range(8))


def test_source_quality_uses_only_missing_and_next_twelve() -> None:
    dates = pd.date_range("2020-01-01", periods=1000, freq="5min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "source_complete": np.ones(1000, dtype=np.int8),
            "source_quarantined": np.zeros(1000, dtype=np.int8),
        }
    )
    frame.loc[:4, "source_complete"] = 0
    frame.loc[:20, "source_quarantined"] = 1
    metrics = support.source_quality(frame)
    assert metrics["source_missing_rows"] == 5
    assert metrics["missing_or_next_12_quarantined_rows"] == 21
    assert metrics["global_missing_or_quarantined_fraction"] == 0.021
    assert metrics["pass"] is False
