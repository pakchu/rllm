import numpy as np
import pandas as pd

from training import build_high_volatility_taker_imbalance_concentration_continuation_support as support


def test_prior_rank_excludes_current() -> None:
    ranks = support.prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_flow_features_capture_scale_free_signed_concentration() -> None:
    quote = np.full(96, 100.0)
    taker = np.full(96, 50.0)
    taker[0] = 75.0
    concentration, net_flow = support.flow_features(quote, taker)
    scaled_concentration, scaled_net_flow = support.flow_features(quote * 7, taker * 7)
    assert concentration == 1.0
    assert np.isclose(net_flow, 50.0 / 9600.0)
    assert np.isclose(scaled_concentration, concentration)
    assert np.isclose(scaled_net_flow, net_flow)


def test_opposing_equal_concentrated_flows_have_zero_signed_hhi() -> None:
    quote = np.full(96, 100.0)
    taker = np.full(96, 50.0)
    taker[0], taker[1] = 75.0, 25.0
    concentration, net_flow = support.flow_features(quote, taker)
    assert concentration == 0.0
    assert net_flow == 0.0


def test_primary_onset_and_side_use_frozen_concentration() -> None:
    states = pd.DataFrame({
        "source_valid": [True] * 4,
        "signed_concentration": [0.1, 0.2, -0.3, -0.1],
        "concentration_rank": [0.6, 0.8, 0.8, 0.7],
        "aggregate_net_flow": [0.01, 0.02, -0.03, -0.01],
        "aggregate_net_flow_rank": [0.2] * 4,
        "variation_rank": [0.8] * 4,
    })
    active, side = support.active(states, "primary")
    assert active.tolist() == [False, True, False, False]
    assert side.tolist() == [1.0, 1.0, -1.0, -1.0]


def test_contract_is_frozen() -> None:
    assert support.PREREG_SHA == "85b7b62a4076b39ec5ae274c42d2d9022b75230237bde67ebf21131d03ed2a02"
    assert support.CONTROLS == (
        "no_concentration_tail",
        "no_variation_gate",
        "aggregate_net_flow_tail",
        "one_decision_stale_concentration",
        "direction_flip",
        "forced_long",
    )
    assert "quote_asset_volume,taker_buy_quote" in support.QUERY
