import numpy as np
import pandas as pd

from training import build_confirmation_ladder_cash_sponsorship_support as support


def _minutes(start, periods, spot=True):
    times = pd.date_range(start, periods=periods, freq="1min")
    quote = np.linspace(100.0, 120.0, periods) if spot else np.full(periods, 100.0)
    return pd.DataFrame({
        "ts": times, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "quote_asset_volume": quote, "taker_buy_quote": quote * 0.6,
        "duplicate_count": 1,
    })


def test_rank_and_ceil():
    ranks = support.strict_prior_midrank(pd.Series(range(113), dtype=float))
    assert np.isnan(ranks.iloc[111]) and ranks.iloc[112] == 1.0
    assert support.ceil_5m(301) == pd.Timestamp(600, unit="s", tz="UTC")


def test_interval_share_and_flow_are_causal_complete_minutes():
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    spot = support.prepare_minutes(_minutes(start, 10, True), "spot")
    perpetual = support.prepare_minutes(_minutes(start, 10, False), "perpetual")
    values = support.interval_values(spot, perpetual, int(start.timestamp()), int((start + pd.Timedelta(minutes=10)).timestamp()))
    assert values is not None
    share, flow, count = values
    assert share > 0.5 and np.isclose(flow, 0.2) and count == 10


def test_invalid_duration_is_rejected():
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    frame = support.prepare_minutes(_minutes(start, 40), "spot")
    assert support.interval_values(frame, frame, int(start.timestamp()), int((start + pd.Timedelta(minutes=31)).timestamp())) is None


def test_primary_onset_preserves_state_across_invalid_anchor():
    features = pd.DataFrame({
        "source_valid": [True, False, True, True], "eligible_state": [True, False, True, False],
        "migration": [0.1, np.nan, 0.2, -0.1], "flow_sum": [1.0, np.nan, 1.0, -1.0],
        "migration_abs_rank": [0.95, np.nan, 0.95, 0.95], "path_agree_count": [4, 0, 4, 4],
    })
    active, sides = support.active_and_side(features)
    assert active.tolist() == [True, False, False, False]
    assert sides.tolist() == [1, 0, 1, -1]


def test_outcomes_remain_closed():
    source = open(support.__file__).read()
    assert '"execution_prices_opened": False' in source
    assert '"gross9_rows_opened": False' in source
    assert '"rv20_opened": False' in source
