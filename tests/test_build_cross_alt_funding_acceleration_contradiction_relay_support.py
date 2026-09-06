import pandas as pd

from training import build_cross_alt_funding_acceleration_contradiction_relay_support as support


def _rows(times, current_rates):
    rows = []
    for time, rates in zip(pd.to_datetime(times, utc=True), current_rates):
        for symbol, rate in zip(support.SYMBOLS, rates):
            rows.append({"symbol": symbol, "funding_time": time, "funding_rate": rate})
    return pd.DataFrame(rows)


def test_four_alt_majority_opposed_by_btc_maps_to_fade_side():
    prior = [0.0] * 7
    current = [-1.0, 1.0, 2.0, 3.0, 4.0, -1.0, 0.0]
    features = support.build_features(_rows(["2024-07-01T00:00Z", "2024-07-01T08:00Z"], [prior, current]))
    row = features.iloc[-1]
    assert row.alt_majority_side == 1
    assert row.btc_change_side == -1
    assert row.btc_contradiction
    assert support.signal(features, "primary").iloc[-1] == -1
    assert support.signal(features, "follow_alt_majority").iloc[-1] == 1


def test_tie_missing_symbol_and_nonconsecutive_settlement_are_ineligible():
    prior = [0.0] * 7
    tied = [-1.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0]
    frame = _rows(["2024-07-01T00:00Z", "2024-07-01T08:00Z"], [prior, tied])
    assert support.signal(support.build_features(frame)).iloc[-1] == 0
    missing = frame[~((frame.funding_time == pd.Timestamp("2024-07-01T08:00Z")) & (frame.symbol == "DOGEUSDT"))]
    assert not support.build_features(missing).iloc[-1].common_valid
    gap = _rows(["2024-07-01T00:00Z", "2024-07-01T16:00Z"], [prior, tied])
    assert not support.build_features(gap).iloc[-1].prior_common_consecutive


def test_clock_and_source_gates_are_frozen():
    assert support.MINIMUM == {"train": 8, "test": 12, "eval": 12, "final": 8}
    prior = [0.0] * 7; current = [-1.0, 1.0, 2.0, 3.0, 4.0, -1.0, 0.0]
    features = support.build_features(_rows(["2024-07-01T00:00Z", "2024-07-01T08:00Z"], [prior, current]))
    clock = support.build_clock(features)
    assert len(clock) == 1
    assert clock.iloc[0].entry_time == pd.Timestamp("2024-07-01T08:05Z")
    assert clock.iloc[0].exit_time == pd.Timestamp("2024-07-01T16:05Z")


def test_builder_never_opens_btc_prices_or_outcomes():
    source = open(support.__file__).read()
    assert "bars_binance" not in support.QUERY
    assert '"btc_price_rows_opened": False' in source
    assert '"postentry_return_or_pnl_opened": False' in source
    assert '"gross9_rows_opened": False' in source
