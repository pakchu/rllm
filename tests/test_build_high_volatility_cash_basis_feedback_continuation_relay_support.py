import numpy as np
import pandas as pd

from training import build_high_volatility_cash_basis_feedback_continuation_relay_support as s


def test_prior_rank_excludes_current():
    ranks = s.prior_rank(pd.Series(range(181), dtype=float))
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_feedback_score_respects_lag_direction():
    spot = np.linspace(-1.0, 1.0, 480)
    basis = np.zeros(480)
    basis[1:] = spot[:-1]
    assert s.feedback_score(spot, basis, lag=1) > 0.999
    assert s.feedback_score(spot, -basis, lag=1) < -0.999
    assert np.isnan(s.feedback_score(np.ones(480), basis, lag=1))


def test_onset_side_and_contract():
    states = pd.DataFrame(
        {
            "source_valid": [True] * 4,
            "block_return": [0.01, 0.01, -0.01, -0.01],
            "variation_rank": [0.6, 0.7, 0.8, 0.8],
            "feedback_rank": [0.8, 0.8, 0.8, 0.7],
            "cash_basis_feedback": [1.0] * 4,
            "contemporaneous_feedback": [1.0] * 4,
            "contemporaneous_rank": [0.8] * 4,
        }
    )
    onset, side = s.active(states, "primary")
    assert onset.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, -1, -1]
    assert s.PREREG_SHA == (
        "ed3757cd63e27ca374bce692b581948a249450c4f9f65cda44e416c53ec6583c"
    )
