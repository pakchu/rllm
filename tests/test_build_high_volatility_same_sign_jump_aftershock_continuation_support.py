import numpy as np
import pandas as pd

from training import build_high_volatility_same_sign_jump_aftershock_continuation_support as support


def test_prior_rank_excludes_current() -> None:
    ranks = support.prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_jump_features_measure_short_gap_excitation_and_signed_mass() -> None:
    returns = np.full(95, 0.001)
    returns[[10, 11, 13, 40]] = [0.01, 0.012, 0.011, -0.01]
    excitation, count, signed_mass, scale = support.jump_features(returns)
    assert count == 4
    assert excitation == 2 / 3
    assert signed_mass > 0
    assert scale > 0


def test_primary_onset_and_side_use_frozen_excitation() -> None:
    states = pd.DataFrame({
        "source_valid": [True] * 4,
        "signed_jump_mass": [0.01, 0.01, -0.01, -0.01],
        "variation_rank": [0.6, 0.7, 0.8, 0.8],
        "excitation_rank": [0.8, 0.8, 0.8, 0.7],
        "jump_count_rank": [0.2] * 4,
    })
    active, side = support.active(states, "primary")
    assert active.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, -1, -1]


def test_contract_is_frozen() -> None:
    assert support.PREREG_SHA == "9500eb575e5536a9fe2278843ebc356703a471532d7e6031ac8a2d9583602ccf"
    assert support.CONTROLS == (
        "no_excitation_gate", "no_variation_gate", "jump_count_tail",
        "one_decision_stale_excitation", "direction_flip", "forced_long",
    )
    assert "FROM bars_binance" in support.QUERY
