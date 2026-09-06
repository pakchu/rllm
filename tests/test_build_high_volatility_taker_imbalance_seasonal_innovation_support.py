import numpy as np
import pandas as pd

from training import build_high_volatility_taker_imbalance_seasonal_innovation_support as support


def test_prior_rank_excludes_current() -> None:
    ranks = support.prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_net_flow_feature_uses_aggregate_taker_imbalance() -> None:
    quote = np.full(96, 100.0)
    taker = np.full(96, 60.0)
    assert np.isclose(support.net_flow_feature(quote, taker), 0.2)


def test_same_slot_innovation_excludes_current_and_separates_slots() -> None:
    decisions = pd.Series(pd.date_range("2023-01-01T01:00:00Z", periods=183, freq="8h"))
    slot_sequence = np.repeat(np.arange(61, dtype=float), 3)
    slot_offsets = np.tile(np.array([0.0, 1000.0, -1000.0]), 61)
    net_flow = pd.Series(slot_sequence + slot_offsets)
    valid = pd.Series(True, index=net_flow.index)

    innovation = support.same_slot_innovation(decisions, net_flow, valid)

    assert innovation.flow_innovation.iloc[:180].isna().all()
    for index in range(180, 183):
        same_slot_prior = net_flow.iloc[index % 3:index:3].iloc[-90:].to_numpy()
        median = np.median(same_slot_prior)
        q25, q75 = np.quantile(same_slot_prior, [0.25, 0.75], method="linear")
        expected = (net_flow.iloc[index] - median) / ((q75 - q25) / 1.349)
        assert np.isclose(innovation.flow_innovation.iloc[index], expected)


def test_primary_onset_and_side_use_frozen_innovation() -> None:
    states = pd.DataFrame({
        "source_valid": [True] * 4,
        "net_flow": [0.01, 0.02, -0.03, -0.01],
        "flow_innovation": [0.1, 0.2, -0.3, -0.1],
        "innovation_rank": [0.6, 0.8, 0.8, 0.7],
        "raw_net_flow_rank": [0.2] * 4,
        "variation_rank": [0.8] * 4,
    })
    active, side = support.active(states, "primary")
    assert active.tolist() == [False, True, False, False]
    assert side.tolist() == [1.0, 1.0, -1.0, -1.0]


def test_contract_is_frozen() -> None:
    assert support.PREREG_SHA == "f134927a7dc62e3b1981cfe8dbeaf57d8cdb4937ee08d13a1fdd4fddd05c6b8f"
    assert support.CONTROLS == (
        "no_innovation_tail",
        "no_variation_gate",
        "raw_net_flow_tail",
        "one_decision_stale_innovation",
        "direction_flip",
        "forced_long",
    )
    assert "quote_asset_volume,taker_buy_quote" in support.QUERY
