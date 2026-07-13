from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_nested_barrier_witness_alpha import (
    HORIZONS,
    build_barrier_bank,
    coalesced_barrier_signals,
    rolling_prior_extreme_index,
)


def _market(rows: int) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100.0 + 0.01 * index + 0.5 * np.sin(index / 7.0)
    quote = 1_000_000.0 + 1_000.0 * index
    imbalance = 0.2 * np.sin(index / 5.0)
    return pd.DataFrame(
        {
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "quote_asset_volume": quote,
            "taker_buy_quote": quote * (1.0 + imbalance) / 2.0,
        }
    )


def _manual_bank(rows: int, witness_indices: tuple[int, int, int]) -> dict[object, object]:
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


def test_prior_extreme_excludes_current_and_prefers_most_recent_tie() -> None:
    values = np.array([1.0, 3.0, 2.0, 3.0, 0.0])

    maxima = rolling_prior_extreme_index(values, window=3, kind="max")
    minima = rolling_prior_extreme_index(values, window=3, kind="min")

    assert maxima[3] == 1
    assert maxima[4] == 3
    assert minima[3] == 0
    assert minima[4] == 2


def test_barrier_bank_prefix_does_not_depend_on_future_suffix() -> None:
    prefix = _market(2_100)
    suffix = _market(100) * 100.0
    full = pd.concat([prefix, suffix], ignore_index=True)

    expected = build_barrier_bank(prefix)
    actual = build_barrier_bank(full)

    for horizon in HORIZONS:
        for key in ("high_index", "low_index", "high_price", "low_price"):
            np.testing.assert_allclose(
                actual[horizon][key][: len(prefix)],
                expected[horizon][key],
                equal_nan=True,
            )
    np.testing.assert_allclose(actual["buy_work"][: len(prefix)], expected["buy_work"], equal_nan=True)
    np.testing.assert_allclose(actual["sell_work"][: len(prefix)], expected["sell_work"], equal_nan=True)


def test_longest_scale_ancestor_owns_witness_work() -> None:
    market = pd.DataFrame({"close": [90.0] * 6 + [99.95] + [90.0] * 3})
    bank = _manual_bank(len(market), (2, 3, 4))
    bank["buy_work"][[2, 3, 4, 6]] = [0.4, 0.4, 0.8, 0.4]

    long_active, short_active, diagnostics = coalesced_barrier_signals(
        market,
        bank,
        min_coalescence=3,
        touch_width=0.001,
        branch="depleted_continuation",
    )

    np.testing.assert_array_equal(np.flatnonzero(long_active), [6])
    assert not short_active.any()
    assert diagnostics["high_work_ratio"][6] == 0.5
    assert diagnostics["high_coalescence"][6] == 3


def test_reinforced_high_barrier_fades_and_flip_is_exact() -> None:
    market = pd.DataFrame({"close": [90.0] * 6 + [99.95] + [90.0] * 3})
    bank = _manual_bank(len(market), (2, 2, 2))
    bank["buy_work"][[2, 6]] = [0.4, 0.8]
    kwargs = {
        "min_coalescence": 3,
        "touch_width": 0.001,
        "branch": "reinforced_fade",
    }

    long_active, short_active, diagnostics = coalesced_barrier_signals(market, bank, **kwargs)
    flip_long, flip_short, _ = coalesced_barrier_signals(market, bank, flip=True, **kwargs)

    assert not long_active.any()
    np.testing.assert_array_equal(np.flatnonzero(short_active), [6])
    assert diagnostics["high_work_ratio"][6] == 2.0
    np.testing.assert_array_equal(flip_long, short_active)
    np.testing.assert_array_equal(flip_short, long_active)


def test_noncoalesced_origins_do_not_signal() -> None:
    market = pd.DataFrame({"close": [90.0] * 6 + [99.95] + [90.0] * 3})
    bank = _manual_bank(len(market), (0, 2, 5))
    bank["buy_work"][[0, 2, 5, 6]] = [0.8, 0.8, 0.8, 0.4]

    long_active, short_active, _ = coalesced_barrier_signals(
        market,
        bank,
        min_coalescence=3,
        touch_width=0.001,
        branch="depleted_continuation",
        max_origin_separation=3,
    )

    assert not long_active.any() and not short_active.any()
