import pandas as pd

from training import build_high_volatility_cross_quote_ticket_rotation_relay_support as support


def _state(**changes):
    row = {
        "decision_time": pd.Timestamp("2023-07-02T08:00:00Z"), "flow_consensus": True,
        "ticket_rotation": True, "all_ticket_expand": False, "variation_rank": 0.8,
        "ticket_change_BTCUSDT": -0.1, "ticket_change_BTCUSDC": 0.2, "ticket_change_BTCFDUSD": 0.3,
        "flow_BTCUSDC": 2.0, "flow_BTCFDUSD": 3.0, "btc_variation": 0.1,
    }
    row.update(changes)
    return pd.DataFrame([row])


def test_primary_and_direction_controls_use_consensus_side():
    states = _state()
    assert support.build_clock(states).side.tolist() == [1]
    assert support.build_clock(states, "direction_flip").side.tolist() == [-1]
    assert support.build_clock(states, "same_clock_forced_long").side.tolist() == [1]


def test_no_ticket_rotation_is_only_control_active_without_rotation():
    states = _state(ticket_rotation=False, flow_BTCUSDC=-2.0, flow_BTCFDUSD=-3.0)
    assert support.build_clock(states).empty
    assert support.build_clock(states, "no_ticket_rotation").side.tolist() == [-1]


def test_block_panel_requires_complete_exact_hours_and_unsigned_ticket():
    dates = pd.date_range("2023-07-01", periods=8, freq="1h", tz="UTC")
    rows = []
    for symbol in support.SYMBOLS:
        for date in dates:
            rows.append({"date": date, "symbol": symbol, "base_volume_btc": 16.0, "trade_count": 100, "signed_taker_flow_btc": -1.0, "source_complete": True})
    panel = support.block_panel(pd.DataFrame(rows))
    assert panel.block_valid.tolist() == [True]
    assert panel.ticket_BTCUSDT.tolist() == [0.16]
