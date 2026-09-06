import numpy as np
import pandas as pd

from training import build_high_volatility_return_volume_transfer_entropy_relay_support as support


def test_prior_rank_excludes_current() -> None:
    ranks = support.prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_conditional_information_detects_perfect_turnover_to_next_sign_transfer() -> None:
    rng = np.random.default_rng(19)
    turnover = np.array([0] * 48 + [1] * 48)
    rng.shuffle(turnover)
    signs = np.empty(96, dtype=int)
    signs[0] = 0
    signs[1:] = turnover[:-1]
    information, lift, cell_min = support.conditional_information(turnover, signs)
    assert information > 0.5
    assert lift == 1.0
    assert cell_min >= 5


def test_conditional_information_rejects_sparse_conditioning_cells() -> None:
    turnover = np.array([0] * 95 + [1])
    signs = np.tile([0, 1], 48)
    information, lift, cell_min = support.conditional_information(turnover, signs)
    assert np.isnan(information)
    assert np.isnan(lift)
    assert cell_min < 5


def test_onset_compares_previous_source_valid_decision() -> None:
    source_valid = pd.Series([True, False, True, True, True])
    eligible = pd.Series([False, False, True, True, False])
    onset = support.onset_after_previous_source_valid(source_valid, eligible)
    assert onset.tolist() == [False, False, True, False, False]


def test_contract_is_frozen() -> None:
    assert support.PREREG_SHA == "6921cf0a08a7f816bf95ccf0dfdc4aea6152a161404e34f916e5ab2cbbba95c6"
    assert support.CONTROLS == (
        "no_information_tail",
        "no_variation_gate",
        "unconditional_transition_lift",
        "contemporaneous_conditioned_information",
        "one_decision_stale_information",
        "direction_flip",
        "same_clock_forced_long",
    )
    assert "quote_asset_volume" in support.QUERY
    assert "taker_buy" not in support.QUERY
