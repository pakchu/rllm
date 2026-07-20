from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from execution.binance_aggtrade_stream import parse_aggtrade_payload
from preprocessing.aggressor_frustration import (
    AggTradeTickState,
    advance_tick_state,
    aggregate_frustration_five_minute,
    classify_event_frame,
)


RAW_COLUMNS = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
)


def _frame(rows: list[list[object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_event_tick_state_carries_across_day_and_matches_scalar_replay() -> None:
    midnight = int(pd.Timestamp("2021-01-02", tz="UTC").timestamp() * 1_000)
    raw = _frame(
        [
            [11, 101.0, 1.0, 11, 11, midnight - 1, False],
            [12, 101.0, 2.0, 12, 12, midnight, True],
            [13, 100.0, 3.0, 13, 13, midnight + 1, False],
            [14, 100.0, 4.0, 14, 14, midnight + 2, False],
        ]
    )
    initial = AggTradeTickState(100.0, 1, 10)
    classified, vector_state = classify_event_frame(raw, initial_state=initial)

    scalar_state = initial
    scalar_ticks: list[int] = []
    scalar_raw: list[int] = []
    scalar_resets: list[bool] = []
    for row in raw.itertuples(index=False):
        tick, raw_tick, reset, scalar_state = advance_tick_state(
            scalar_state,
            aggregate_trade_id=row.agg_trade_id,
            price=row.price,
        )
        scalar_ticks.append(tick)
        scalar_raw.append(raw_tick)
        scalar_resets.append(reset)

    assert classified["tick"].tolist() == [1, 1, -1, -1]
    assert classified["raw_tick"].tolist() == [1, 0, -1, 0]
    assert classified["state_reset"].tolist() == [False] * 4
    assert classified["tick"].tolist() == scalar_ticks
    assert classified["raw_tick"].tolist() == scalar_raw
    assert classified["state_reset"].tolist() == scalar_resets
    assert vector_state == scalar_state == AggTradeTickState(100.0, -1, 14)


def test_id_gap_resets_and_equal_prices_remain_unavailable_until_new_move() -> None:
    raw = _frame(
        [
            [1, 100.0, 1.0, 1, 1, 1_000, False],
            [2, 101.0, 1.0, 2, 2, 2_000, False],
            [5, 99.0, 1.0, 5, 5, 3_000, True],
            [6, 99.0, 1.0, 6, 6, 4_000, True],
            [7, 98.0, 1.0, 7, 7, 5_000, False],
            [8, 98.0, 1.0, 8, 8, 6_000, False],
        ]
    )
    classified, state = classify_event_frame(raw)
    assert classified["tick"].tolist() == [0, 1, 0, 0, -1, -1]
    assert classified["state_reset"].tolist() == [True, False, True, False, False, False]
    assert state == AggTradeTickState(98.0, -1, 8)


def test_frustration_orientation_and_component_identity() -> None:
    raw = _frame(
        [
            [1, 100.0, 1.0, 1, 1, 1_000, False],
            [2, 99.0, 2.0, 2, 2, 2_000, False],  # strict frustrated buy
            [3, 99.0, 3.0, 3, 3, 3_000, False],  # carried frustrated buy
            [4, 100.0, 4.0, 4, 4, 4_000, True],  # strict frustrated sell
            [5, 100.0, 5.0, 5, 5, 5_000, True],  # carried frustrated sell
        ]
    )
    bars, _ = aggregate_frustration_five_minute(raw)
    bar = bars.iloc[0]
    assert bar["strict_buy_frustrated_notional"] == 198.0
    assert bar["carried_buy_frustrated_notional"] == 297.0
    assert bar["strict_sell_frustrated_notional"] == 400.0
    assert bar["carried_sell_frustrated_notional"] == 500.0
    assert bar["buy_frustrated_notional"] == 495.0
    assert bar["sell_frustrated_notional"] == 900.0
    assert np.isclose(
        bar["frustration_score"],
        (900.0 - 495.0) / bar["quote_notional"],
    )
    assert bar["frustration_score"] > 0.0
    assert bar["agg_trade_count"] == (
        bar["classified_tick_count"] + bar["unavailable_tick_count"]
    )
    assert np.isclose(
        bar["strict_buy_frustrated_notional"] + bar["carried_buy_frustrated_notional"],
        bar["buy_frustrated_notional"],
    )


def test_score_is_price_scale_invariant() -> None:
    raw = _frame(
        [
            [1, 100.0, 1.0, 1, 1, 1_000, False],
            [2, 99.0, 2.0, 2, 2, 2_000, False],
            [3, 100.0, 3.0, 3, 3, 3_000, True],
        ]
    )
    scaled = raw.copy()
    scaled["price"] *= 37.0
    original_bars, _ = aggregate_frustration_five_minute(raw)
    scaled_bars, _ = aggregate_frustration_five_minute(scaled)
    assert np.allclose(original_bars["frustration_score"], scaled_bars["frustration_score"])
    assert np.sign(original_bars["frustration_score"].iloc[0]) == np.sign(
        scaled_bars["frustration_score"].iloc[0]
    )


def test_later_bucket_mutation_cannot_change_earlier_bucket() -> None:
    raw = _frame(
        [
            [1, 100.0, 1.0, 1, 1, 1_000, False],
            [2, 99.0, 1.0, 2, 2, 2_000, False],
            [3, 101.0, 1.0, 3, 3, 301_000, True],
            [4, 102.0, 1.0, 4, 4, 302_000, True],
        ]
    )
    changed = raw.copy()
    changed.loc[3, ["price", "quantity", "is_buyer_maker"]] = [77.0, 99.0, False]
    original, _ = aggregate_frustration_five_minute(raw)
    mutated, _ = aggregate_frustration_five_minute(changed)
    pd.testing.assert_series_equal(original.iloc[0], mutated.iloc[0])


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [
                [1, 100.0, 1.0, 1, 1, 1_000, False],
                [1, 101.0, 1.0, 2, 2, 2_000, True],
            ],
            "strictly increasing",
        ),
        (
            [
                [2, 100.0, 1.0, 1, 1, 1_000, False],
                [1, 101.0, 1.0, 2, 2, 2_000, True],
            ],
            "strictly increasing",
        ),
    ],
)
def test_duplicate_or_reversed_ids_fail_closed(rows: list[list[object]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        classify_event_frame(_frame(rows))


def test_archive_and_live_message_replay_have_identical_tick_states_and_scores() -> None:
    payloads = [
        {"e": "aggTrade", "a": 1, "p": "100", "q": "1", "f": 1, "l": 1, "T": 1_000, "m": False},
        {"e": "aggTrade", "a": 2, "p": "99", "q": "2", "f": 2, "l": 2, "T": 2_000, "m": False},
        {"e": "aggTrade", "a": 3, "p": "99", "q": "3", "f": 3, "l": 3, "T": 3_000, "m": True},
        {"e": "aggTrade", "a": 4, "p": "100", "q": "4", "f": 4, "l": 4, "T": 4_000, "m": True},
    ]
    archive = _frame(
        [
            [item["a"], float(item["p"]), float(item["q"]), item["f"], item["l"], item["T"], item["m"]]
            for item in payloads
        ]
    )
    live_ticks = [parse_aggtrade_payload(item) for item in payloads]
    live = _frame(
        [
            [
                tick.aggregate_trade_id,
                tick.price,
                tick.quantity,
                tick.first_trade_id,
                tick.last_trade_id,
                tick.event_time_ms,
                tick.is_buyer_maker,
            ]
            for tick in live_ticks
        ]
    )

    archive_events, archive_state = classify_event_frame(archive)
    live_events, live_state = classify_event_frame(live)
    pd.testing.assert_frame_equal(archive_events, live_events)
    assert archive_state == live_state
    archive_bars, _ = aggregate_frustration_five_minute(archive)
    live_bars, _ = aggregate_frustration_five_minute(live)
    pd.testing.assert_frame_equal(archive_bars, live_bars)
