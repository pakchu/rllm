import numpy as np
import pandas as pd

from training import build_high_volatility_sample_entropy_collapse_continuation_support as support


def test_prior_rank_excludes_current() -> None:
    ranks = support.prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_sample_entropy_is_lower_for_repeated_path() -> None:
    repeated = np.tile(np.array([-1.0, 1.0, -0.5, 0.5]), 24)[:95]
    noisy = np.random.default_rng(7).normal(size=95)
    repeated_entropy, repeated_b, repeated_a = support.sample_entropy(repeated)
    noisy_entropy, noisy_b, noisy_a = support.sample_entropy(noisy)
    assert np.isfinite([repeated_entropy, noisy_entropy]).all()
    assert repeated_entropy < noisy_entropy
    assert repeated_b >= repeated_a > 0
    assert noisy_b >= noisy_a > 0


def test_primary_onset_and_side_use_frozen_low_tail() -> None:
    states = pd.DataFrame({
        "source_valid": [True] * 4,
        "block_return": [0.01, 0.01, -0.01, -0.01],
        "variation_rank": [0.6, 0.7, 0.8, 0.8],
        "entropy_rank": [0.2, 0.2, 0.2, 0.3],
        "sign_entropy_rank": [0.8] * 4,
    })
    active, side = support.active(states, "primary")
    assert active.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, -1, -1]


def test_contract_is_frozen() -> None:
    assert support.PREREG_SHA == "3c8e6b15e084895e16ee7c66952ee7ae1cf683d45064f5756889b90bd58789d2"
    assert support.CONTROLS == (
        "no_entropy_gate", "no_variation_gate", "sign_entropy_low_tail",
        "one_decision_stale_entropy", "direction_flip", "forced_long",
    )
    assert "FROM bars_binance" in support.QUERY
