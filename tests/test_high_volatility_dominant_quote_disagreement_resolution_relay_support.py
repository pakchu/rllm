import pandas as pd

from training import build_high_volatility_dominant_quote_disagreement_resolution_relay_support as support


def _state(**changes):
    row = {
        "decision_time": pd.Timestamp("2023-07-02T08:00:00Z"), "disagreement": True,
        "activity_gate": True, "variation_rank": 0.8, "btc_variation": 0.1,
        "intensity_BTCUSDT": 0.2, "intensity_BTCUSDC": -0.3, "intensity_BTCFDUSD": -0.4,
        "activity_threshold_BTCUSDT": 0.1, "activity_threshold_BTCUSDC": 0.1, "activity_threshold_BTCFDUSD": 0.1,
    }
    row.update(changes)
    return pd.DataFrame([row])


def test_primary_follows_dominant_and_controls_do_not_promote():
    states = _state()
    assert support.build_clock(states).side.tolist() == [1]
    assert support.build_clock(states, "alternative_direction").side.tolist() == [-1]
    assert support.build_clock(states, "direction_flip").side.tolist() == [-1]
    assert support.build_clock(states, "same_clock_forced_long").side.tolist() == [1]


def test_activity_gate_is_required_only_by_primary():
    states = _state(activity_gate=False)
    assert support.build_clock(states).empty
    assert support.build_clock(states, "no_activity_gate").side.tolist() == [1]


def test_block_panel_uses_normalized_unsigned_volume():
    rows = []
    for symbol in support.SYMBOLS:
        for date in pd.date_range("2023-07-01", periods=8, freq="1h", tz="UTC"):
            rows.append({"date": date, "symbol": symbol, "base_volume_btc": 10.0, "trade_count": 1, "signed_taker_flow_btc": -2.0, "source_complete": True})
    panel = support.block_panel(pd.DataFrame(rows))
    assert panel.block_valid.tolist() == [True]
    assert panel.intensity_BTCUSDT.tolist() == [-0.2]
