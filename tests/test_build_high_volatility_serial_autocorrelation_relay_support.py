import numpy as np
import pandas as pd

from training import build_high_volatility_serial_autocorrelation_relay_support as support


def test_strict_prior_midrank_excludes_current_and_caps_history() -> None:
    values = np.arange(300, dtype=float)
    ranks = support._strict_prior_midrank(values, lookback=270, minimum=252)
    assert np.isnan(ranks[251])
    assert ranks[252] == 1.0
    assert ranks[299] == 1.0


def test_serial_dependence_sign_selects_momentum_or_reversal() -> None:
    scores = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2023-07-02T00:00:00Z", "2023-07-03T00:00:00Z"], utc=True
            ),
            "completed_return_12h": [0.02, 0.02],
            "realized_variation": [0.03, 0.03],
            "lag_one_autocorrelation": [0.2, -0.2],
            "absolute_autocorrelation": [0.2, 0.2],
            "variation_rank": [0.8, 0.8],
            "absolute_autocorrelation_rank": [0.9, 0.9],
            "source_valid": [True, True],
        }
    )
    clock = support.build_clock(scores)
    assert clock.side.tolist() == [1, -1]
    assert clock.entry_time.tolist() == [
        pd.Timestamp("2023-07-02T00:05:00Z"),
        pd.Timestamp("2023-07-03T00:05:00Z"),
    ]


def test_controls_do_not_change_primary_thresholds_or_promote() -> None:
    scores = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(["2023-07-02T00:00:00Z"], utc=True),
            "completed_return_12h": [-0.02],
            "realized_variation": [0.03],
            "lag_one_autocorrelation": [-0.2],
            "absolute_autocorrelation": [0.2],
            "variation_rank": [0.8],
            "absolute_autocorrelation_rank": [0.9],
            "source_valid": [True],
        }
    )
    assert support.build_clock(scores).side.tolist() == [1]
    assert support.build_clock(scores, "fixed_momentum").side.tolist() == [-1]
    assert support.build_clock(scores, "fixed_reversal").side.tolist() == [1]
    assert support.build_clock(scores, "direction_flip").side.tolist() == [-1]
