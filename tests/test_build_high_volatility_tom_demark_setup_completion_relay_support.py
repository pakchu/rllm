import numpy as np
import pandas as pd

from training import build_high_volatility_tom_demark_setup_completion_relay_support as b


def _five_minute_from_four_hour_closes(closes: list[float]) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2023-01-01T00:00:00Z")
    previous = closes[0]
    for bar, close in enumerate(closes):
        for offset in range(48):
            value = close if offset == 47 else previous
            rows.append({"bar_time": start + pd.Timedelta(hours=4 * bar, minutes=5 * offset), "valid": True, "open": previous, "high": max(previous, value), "low": min(previous, value), "close": value, "variation_24h": 0.1})
        previous = close
    return pd.DataFrame(rows)


def test_buy_setup_emits_only_on_ninth_qualifying_bar():
    closes = [100, 101, 102, 103, 104, 105, 104, 101, 100, 99, 98, 97, 96, 95, 94, 93]
    states = b.derive_setup_states(_five_minute_from_four_hour_closes(closes))
    events = states.index[states.setup_side.ne(0)].tolist()
    assert events == [15]
    assert states.at[15, "setup_side"] == 1
    assert states.at[15, "four_bar_relation"] == -1


def test_invalid_four_hour_bar_resets_relation_and_setup():
    five = _five_minute_from_four_hour_closes(list(np.linspace(100, 80, 20)))
    five.loc[five.index[48 * 8], "valid"] = False
    states = b.derive_setup_states(five)
    assert not states.setup_side.ne(0).any()
    assert states.loc[8:12, "four_bar_relation"].eq(0).all()


def test_strict_prior_midrank_excludes_current():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    valid = pd.Series([True] * 4)
    rank = b.strict_prior_midrank(values, valid, lookback=3, minimum=3)
    assert rank.iloc[:3].isna().all()
    assert rank.iloc[3] == 1.0


def test_clock_reserves_half_open_twenty_four_hours():
    frame = pd.DataFrame({"decision_time": pd.to_datetime(["2023-07-01T00:00:00Z", "2023-07-01T04:00:00Z", "2023-07-02T00:00:00Z"]), "setup_side": [1, -1, -1], "four_bar_relation": [-1, 1, 1], "btc_realized_variation": [0.1, 0.2, 0.3], "btc_variation_rank": [0.9, 0.9, 0.9]})
    clock = b.build_clock(frame)
    assert len(clock) == 2
    assert clock.side.tolist() == [1, -1]
    assert clock.entry_time.iloc[1] == pd.Timestamp("2023-07-02T00:05:00Z")
