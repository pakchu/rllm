from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_barrier_constitution_alpha import (
    barrier_constitution_signals,
    candle_close_location,
)
from training.search_nested_barrier_witness_alpha import HORIZONS


def _market(origin_close: float, revisit_close: float = 99.95) -> pd.DataFrame:
    close = np.array([90.0, 90.0, origin_close, 90.0, 90.0, 90.0, revisit_close, 90.0])
    high = np.full(len(close), 100.0)
    low = np.full(len(close), 90.0)
    return pd.DataFrame({"close": close, "high": high, "low": low})


def _high_bank(witness_indices: tuple[int, int, int]) -> dict[object, object]:
    rows = 8
    bank: dict[object, object] = {}
    for horizon, witness_index in zip(HORIZONS, witness_indices):
        high_index = np.full(rows, -1, dtype=np.int64)
        high_price = np.full(rows, np.nan)
        low_index = np.full(rows, -1, dtype=np.int64)
        low_price = np.full(rows, np.nan)
        high_index[6] = witness_index
        high_price[6] = 100.0
        bank[horizon] = {
            "high_index": high_index,
            "high_price": high_price,
            "low_index": low_index,
            "low_price": low_price,
        }
    bank["buy_work"] = np.zeros(rows, dtype=float)
    bank["sell_work"] = np.zeros(rows, dtype=float)
    return bank


def test_close_location_is_safe_and_completed_bar_only() -> None:
    market = pd.DataFrame(
        {
            "low": [0.0, 1.0, 2.0, 4.0],
            "high": [2.0, 3.0, 4.0, 4.0],
            "close": [2.0, 2.0, 2.0, 4.0],
        }
    )

    actual = candle_close_location(market)

    np.testing.assert_allclose(actual[:3], [1.0, 0.0, -1.0])
    assert np.isnan(actual[3])


def test_accepted_high_origin_with_depleted_work_continues_long() -> None:
    market = _market(origin_close=99.0)
    bank = _high_bank((2, 2, 2))
    bank["buy_work"][[2, 6]] = [0.8, 0.4]

    long_active, short_active, diagnostics = barrier_constitution_signals(
        market,
        bank,
        min_coalescence=3,
        touch_width=0.001,
    )

    np.testing.assert_array_equal(np.flatnonzero(long_active), [6])
    assert not short_active.any()
    assert diagnostics["origin_clv"][6] == 0.8
    assert diagnostics["work_ratio"][6] == 0.5


def test_rejected_high_origin_with_reinforced_work_fades_short() -> None:
    market = _market(origin_close=91.0)
    bank = _high_bank((2, 2, 2))
    bank["buy_work"][[2, 6]] = [0.4, 0.8]

    long_active, short_active, diagnostics = barrier_constitution_signals(
        market,
        bank,
        min_coalescence=3,
        touch_width=0.001,
    )

    assert not long_active.any()
    np.testing.assert_array_equal(np.flatnonzero(short_active), [6])
    assert diagnostics["origin_clv"][6] == -0.8
    assert diagnostics["work_ratio"][6] == 2.0


def test_longest_horizon_owns_origin_constitution_and_work() -> None:
    market = _market(origin_close=99.0)
    market.loc[3, "close"] = 91.0
    market.loc[4, "close"] = 99.0
    bank = _high_bank((2, 3, 4))
    bank["buy_work"][[2, 3, 4, 6]] = [0.8, 0.4, 0.8, 0.4]

    long_active, short_active, diagnostics = barrier_constitution_signals(
        market,
        bank,
        min_coalescence=3,
        touch_width=0.001,
    )

    np.testing.assert_array_equal(np.flatnonzero(long_active), [6])
    assert not short_active.any()
    assert diagnostics["origin_clv"][6] == 0.8
    assert diagnostics["work_ratio"][6] == 0.5


def test_origin_inversion_and_direction_flip_are_exact_controls() -> None:
    market = _market(origin_close=99.0)
    bank = _high_bank((2, 2, 2))
    bank["buy_work"][[2, 6]] = [0.8, 0.4]
    kwargs = {"min_coalescence": 3, "touch_width": 0.001}

    long_active, short_active, _ = barrier_constitution_signals(market, bank, **kwargs)
    invert_long, invert_short, _ = barrier_constitution_signals(
        market, bank, invert_origin=True, **kwargs
    )
    flip_long, flip_short, _ = barrier_constitution_signals(market, bank, flip=True, **kwargs)

    assert long_active.any() and not short_active.any()
    assert not invert_long.any() and not invert_short.any()
    np.testing.assert_array_equal(flip_long, short_active)
    np.testing.assert_array_equal(flip_short, long_active)
