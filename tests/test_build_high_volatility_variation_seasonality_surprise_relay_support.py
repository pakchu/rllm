import numpy as np
import pandas as pd

from training import build_high_volatility_variation_seasonality_surprise_relay_support as s


def test_prior_rank_excludes_current():
    ranks = s.prior_rank(pd.Series(range(181), dtype=float))
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_seasonal_surprise_uses_eight_same_week_slots():
    counts = pd.Series(np.full(170, 100.0))
    counts.iloc[168] = 200.0
    reference, surprise = s.seasonal_surprise(counts)
    assert np.isnan(reference.iloc[167])
    assert reference.iloc[168] == 100.0
    assert surprise.iloc[168] == np.log(2.0)
    damaged = counts.copy()
    damaged.iloc[168 - 21 * 3] = np.nan
    assert np.isnan(s.seasonal_surprise(damaged)[1].iloc[168])


def test_onset_side_and_contract():
    states = pd.DataFrame(
        {
            "source_valid": [True] * 4,
            "block_return": [0.01, 0.01, -0.01, -0.01],
            "variation_rank": [0.6, 0.7, 0.8, 0.8],
            "surprise_rank": [0.8, 0.8, 0.8, 0.7],
            "variation_surprise": [1.0] * 4,
        }
    )
    onset, side = s.active(states, "primary")
    assert onset.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, -1, -1]
    assert s.PREREG_SHA == (
        "5627622554546ceccaed79bf0ec77795a2cc0e0094486f08627b521dcf5eff33"
    )
