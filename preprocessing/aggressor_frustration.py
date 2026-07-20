"""Causal aggressor-side and trade-price frustration features.

The classifier consumes ordered Binance aggregate trades.  Tick direction is
carried across ordinary chunk boundaries, but resets on an aggregate-trade ID
gap.  A zero tick before the first nonzero move in a segment is unavailable,
not silently assigned a direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FIVE_MINUTES_MS = 5 * 60 * 1_000

BAR_COLUMNS = (
    "date",
    "first_transact_time_ms",
    "last_transact_time_ms",
    "first_agg_trade_id",
    "last_agg_trade_id",
    "agg_trade_count",
    "classified_tick_count",
    "unavailable_tick_count",
    "state_reset_count",
    "quote_notional",
    "classified_quote_notional",
    "buy_quote_notional",
    "sell_quote_notional",
    "signed_quote_notional",
    "up_tick_notional",
    "down_tick_notional",
    "carried_zero_up_notional",
    "carried_zero_down_notional",
    "strict_buy_frustrated_notional",
    "strict_sell_frustrated_notional",
    "carried_buy_frustrated_notional",
    "carried_sell_frustrated_notional",
    "buy_frustrated_notional",
    "sell_frustrated_notional",
    "frustrated_notional_share",
    "frustration_score",
    "tick_notional_imbalance",
)


@dataclass(frozen=True)
class AggTradeTickState:
    previous_price: float | None = None
    last_nonzero_tick: int = 0
    previous_agg_trade_id: int | None = None

    def __post_init__(self) -> None:
        if self.previous_price is not None and (
            not np.isfinite(self.previous_price) or self.previous_price <= 0.0
        ):
            raise ValueError("previous_price must be finite and positive")
        if self.last_nonzero_tick not in (-1, 0, 1):
            raise ValueError("last_nonzero_tick must be -1, 0, or 1")
        if self.previous_agg_trade_id is not None and self.previous_agg_trade_id < 0:
            raise ValueError("previous_agg_trade_id must be nonnegative")
        if (self.previous_price is None) != (self.previous_agg_trade_id is None):
            raise ValueError("previous price and aggregate-trade ID must be set together")


def advance_tick_state(
    state: AggTradeTickState,
    *,
    aggregate_trade_id: int,
    price: float,
) -> tuple[int, int, bool, AggTradeTickState]:
    """Classify one ordered event.

    Returns ``(tick, raw_tick, reset, next_state)``.  Tick ``0`` means the
    direction is unavailable.  ``raw_tick=0`` with a nonzero classified tick
    is an equal-price event carrying the preceding nonzero direction.
    """

    identifier = int(aggregate_trade_id)
    value = float(price)
    if identifier < 0:
        raise ValueError("aggregate_trade_id must be nonnegative")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("price must be finite and positive")

    previous_id = state.previous_agg_trade_id
    if previous_id is not None and identifier <= previous_id:
        raise ValueError("aggregate-trade IDs must be strictly increasing")
    contiguous = previous_id is not None and identifier == previous_id + 1
    if not contiguous:
        next_state = AggTradeTickState(
            previous_price=value,
            last_nonzero_tick=0,
            previous_agg_trade_id=identifier,
        )
        return 0, 0, True, next_state

    assert state.previous_price is not None
    if value > state.previous_price:
        raw_tick = tick = 1
    elif value < state.previous_price:
        raw_tick = tick = -1
    else:
        raw_tick = 0
        tick = state.last_nonzero_tick
    next_state = AggTradeTickState(
        previous_price=value,
        last_nonzero_tick=tick,
        previous_agg_trade_id=identifier,
    )
    return tick, raw_tick, False, next_state


def classify_tick_arrays(
    aggregate_trade_ids: np.ndarray,
    prices: np.ndarray,
    *,
    initial_state: AggTradeTickState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, AggTradeTickState]:
    """Vectorized equivalent of :func:`advance_tick_state`."""

    ids = np.asarray(aggregate_trade_ids, dtype=np.int64)
    values = np.asarray(prices, dtype=np.float64)
    if ids.ndim != 1 or values.ndim != 1 or len(ids) != len(values):
        raise ValueError("aggregate-trade IDs and prices must be equal one-dimensional arrays")
    if len(ids) == 0:
        raise ValueError("aggregate-trade arrays must not be empty")
    if (ids < 0).any() or (np.diff(ids) <= 0).any():
        raise ValueError("aggregate-trade IDs must be strictly increasing and nonnegative")
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("prices must be finite and positive")

    state = initial_state or AggTradeTickState()
    if state.previous_agg_trade_id is not None and ids[0] <= state.previous_agg_trade_id:
        raise ValueError("first aggregate-trade ID must follow the initial state")

    raw_tick = np.zeros(len(ids), dtype=np.int8)
    reset = np.zeros(len(ids), dtype=bool)
    first_contiguous = (
        state.previous_agg_trade_id is not None
        and ids[0] == state.previous_agg_trade_id + 1
    )
    if first_contiguous:
        assert state.previous_price is not None
        raw_tick[0] = np.int8(np.sign(values[0] - state.previous_price))
    else:
        reset[0] = True

    if len(ids) > 1:
        raw_tick[1:] = np.sign(np.diff(values)).astype(np.int8)
        internal_resets = ids[1:] != ids[:-1] + 1
        reset[1:] = internal_resets
        raw_tick[1:][internal_resets] = 0

    tick = np.zeros(len(ids), dtype=np.int8)
    starts = np.flatnonzero(reset)
    if not reset[0]:
        starts = np.concatenate((np.array([0], dtype=np.int64), starts))
    ends = np.concatenate((starts[1:], np.array([len(ids)], dtype=np.int64)))
    for segment_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        segment_raw = raw_tick[start:end]
        nonzero_index = np.where(
            segment_raw != 0,
            np.arange(len(segment_raw), dtype=np.int64),
            -1,
        )
        np.maximum.accumulate(nonzero_index, out=nonzero_index)
        known = nonzero_index >= 0
        segment_tick = tick[start:end]
        segment_tick[known] = segment_raw[nonzero_index[known]]
        seed = (
            state.last_nonzero_tick
            if segment_index == 0 and not reset[0] and first_contiguous
            else 0
        )
        segment_tick[~known] = np.int8(seed)

    final_state = AggTradeTickState(
        previous_price=float(values[-1]),
        last_nonzero_tick=int(tick[-1]),
        previous_agg_trade_id=int(ids[-1]),
    )
    return tick, raw_tick, reset, final_state


def _required_event_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    required = (
        "agg_trade_id",
        "price",
        "quantity",
        "transact_time",
        "is_buyer_maker",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing aggregate-trade columns: {missing}")
    if frame.empty:
        raise ValueError("aggregate-trade frame must not be empty")
    ids = frame["agg_trade_id"].to_numpy(dtype=np.int64, copy=False)
    prices = frame["price"].to_numpy(dtype=np.float64, copy=False)
    quantities = frame["quantity"].to_numpy(dtype=np.float64, copy=False)
    timestamps = frame["transact_time"].to_numpy(dtype=np.int64, copy=False)
    buyer_maker = frame["is_buyer_maker"].to_numpy(dtype=bool, copy=False)
    if not np.isfinite(quantities).all() or (quantities <= 0.0).any():
        raise ValueError("quantities must be finite and positive")
    if (timestamps < 0).any() or (np.diff(timestamps) < 0).any():
        raise ValueError("aggregate-trade timestamps must be monotonic and nonnegative")
    return ids, prices, quantities, timestamps, buyer_maker


def classify_event_frame(
    frame: pd.DataFrame,
    *,
    initial_state: AggTradeTickState | None = None,
) -> tuple[pd.DataFrame, AggTradeTickState]:
    """Return a compact classified-event frame for replay/parity tests."""

    ids, prices, quantities, timestamps, buyer_maker = _required_event_arrays(frame)
    tick, raw_tick, reset, state = classify_tick_arrays(
        ids,
        prices,
        initial_state=initial_state,
    )
    side = np.where(buyer_maker, -1, 1).astype(np.int8)
    return (
        pd.DataFrame(
            {
                "agg_trade_id": ids,
                "price": prices,
                "quantity": quantities,
                "transact_time": timestamps,
                "aggressor_side": side,
                "raw_tick": raw_tick,
                "tick": tick,
                "state_reset": reset,
                "event_notional": prices * quantities,
            }
        ),
        state,
    )


def _reduce_sum(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(values, starts).astype(np.float64, copy=False)


def aggregate_frustration_five_minute(
    frame: pd.DataFrame,
    *,
    initial_state: AggTradeTickState | None = None,
) -> tuple[pd.DataFrame, AggTradeTickState]:
    """Aggregate one ordered raw chunk without losing cross-chunk state."""

    ids, prices, quantities, timestamps, buyer_maker = _required_event_arrays(frame)
    tick, raw_tick, reset, state = classify_tick_arrays(
        ids,
        prices,
        initial_state=initial_state,
    )
    side = np.where(buyer_maker, -1, 1).astype(np.int8)
    notional = prices * quantities
    bar_ms = timestamps - timestamps % FIVE_MINUTES_MS
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(bar_ms[1:] != bar_ms[:-1]).astype(np.int64) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.array([len(ids)], dtype=np.int64)))

    def weighted(mask: np.ndarray) -> np.ndarray:
        return _reduce_sum(np.where(mask, notional, 0.0), starts)

    def counted(mask: np.ndarray) -> np.ndarray:
        return np.add.reduceat(mask.astype(np.int64), starts).astype(np.int64, copy=False)

    classified = tick != 0
    equal_carried = (raw_tick == 0) & classified
    strict_buy_frustrated = (side > 0) & (raw_tick < 0)
    strict_sell_frustrated = (side < 0) & (raw_tick > 0)
    carried_buy_frustrated = (side > 0) & equal_carried & (tick < 0)
    carried_sell_frustrated = (side < 0) & equal_carried & (tick > 0)

    quote_notional = _reduce_sum(notional, starts)
    classified_notional = weighted(classified)
    buy_notional = weighted(side > 0)
    sell_notional = weighted(side < 0)
    up_notional = weighted(tick > 0)
    down_notional = weighted(tick < 0)
    strict_buy = weighted(strict_buy_frustrated)
    strict_sell = weighted(strict_sell_frustrated)
    carried_buy = weighted(carried_buy_frustrated)
    carried_sell = weighted(carried_sell_frustrated)
    buy_frustrated = strict_buy + carried_buy
    sell_frustrated = strict_sell + carried_sell
    total_frustrated = buy_frustrated + sell_frustrated

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(bar_ms[starts], unit="ms", utc=True).tz_localize(None),
            "first_transact_time_ms": timestamps[starts],
            "last_transact_time_ms": timestamps[ends - 1],
            "first_agg_trade_id": ids[starts],
            "last_agg_trade_id": ids[ends - 1],
            "agg_trade_count": ends - starts,
            "classified_tick_count": counted(classified),
            "unavailable_tick_count": counted(~classified),
            "state_reset_count": counted(reset),
            "quote_notional": quote_notional,
            "classified_quote_notional": classified_notional,
            "buy_quote_notional": buy_notional,
            "sell_quote_notional": sell_notional,
            "signed_quote_notional": buy_notional - sell_notional,
            "up_tick_notional": up_notional,
            "down_tick_notional": down_notional,
            "carried_zero_up_notional": weighted(equal_carried & (tick > 0)),
            "carried_zero_down_notional": weighted(equal_carried & (tick < 0)),
            "strict_buy_frustrated_notional": strict_buy,
            "strict_sell_frustrated_notional": strict_sell,
            "carried_buy_frustrated_notional": carried_buy,
            "carried_sell_frustrated_notional": carried_sell,
            "buy_frustrated_notional": buy_frustrated,
            "sell_frustrated_notional": sell_frustrated,
            "frustrated_notional_share": np.divide(
                total_frustrated,
                quote_notional,
                out=np.zeros_like(total_frustrated),
                where=quote_notional > 0.0,
            ),
            "frustration_score": np.divide(
                sell_frustrated - buy_frustrated,
                quote_notional,
                out=np.zeros_like(total_frustrated),
                where=quote_notional > 0.0,
            ),
            "tick_notional_imbalance": np.divide(
                up_notional - down_notional,
                classified_notional,
                out=np.zeros_like(classified_notional),
                where=classified_notional > 0.0,
            ),
        }
    )
    output = output.loc[:, BAR_COLUMNS]
    numeric = output.drop(columns="date").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite BAFR aggregate")
    if output["date"].duplicated().any() or not output["date"].is_monotonic_increasing:
        raise ValueError("five-minute BAFR timestamps are duplicate or unordered")
    return output, state
