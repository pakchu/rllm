"""Strict BCTP target-position economic accounting on caller-supplied frames.

This module intentionally contains no loader for the frozen market or funding
payloads.  Callers must pass already-materialized synthetic or stage-local
frames; the functions below only validate and account those frames.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd

from training import freeze_block_clearing_target_position_evaluator as freeze

FIVE_MINUTES = pd.Timedelta(minutes=5)
SECONDS_PER_YEAR = 365.2425 * 86400.0
ACTIONS = frozenset(freeze.ACTIONS)


@dataclass(frozen=True)
class Bar:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Funding:
    timestamp: pd.Timestamp
    mark: float
    rate: float


def _utc_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"{name} must be timezone aware")
    ts = ts.tz_convert("UTC")
    if pd.isna(ts):
        raise ValueError(f"{name} is NaT")
    return ts


def _column(frame: pd.DataFrame, names: tuple[str, ...], *, kind: str) -> str:
    lowered = {str(col).lower(): str(col) for col in frame.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise ValueError(f"{kind} frame missing one of columns {names}")


def _time_series(frame: pd.DataFrame, *, kind: str) -> pd.Series:
    if isinstance(frame.index, pd.DatetimeIndex):
        if frame.index.tz is None:
            raise ValueError(f"{kind} timestamps must be timezone aware")
        raw = list(frame.index)
    else:
        aliases = {
            "market": ("timestamp", "open_time", "time", "date"),
            "funding": (
                "timestamp",
                "funding_time",
                "funding_time_utc",
                "time",
                "date",
            ),
            "schedule": ("timestamp", "entry_time", "time", "date"),
        }
        col = _column(frame, aliases.get(kind, ("timestamp", "time", "date")), kind=kind)
        raw = frame[col].tolist()
    parsed = [_utc_timestamp(value, name=f"{kind} timestamp") for value in raw]
    return pd.Series(parsed, dtype="datetime64[ns, UTC]")


def _prepare_market(market: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[pd.Timestamp, Bar]:
    if market.empty:
        raise ValueError("market frame is empty")
    times = _time_series(market, kind="market")
    cols = {
        field: _column(market, (field,), kind="market")
        for field in ("open", "high", "low", "close")
    }
    prepared = pd.DataFrame(
        {
            "timestamp": times.to_numpy(),
            "open": pd.to_numeric(market[cols["open"]], errors="raise").to_numpy(),
            "high": pd.to_numeric(market[cols["high"]], errors="raise").to_numpy(),
            "low": pd.to_numeric(market[cols["low"]], errors="raise").to_numpy(),
            "close": pd.to_numeric(market[cols["close"]], errors="raise").to_numpy(),
        }
    )
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True)
    if prepared["timestamp"].duplicated().any():
        raise ValueError("market timestamps are duplicated")
    expected = pd.date_range(start=start, end=end - FIVE_MINUTES, freq=FIVE_MINUTES, tz="UTC")
    actual = pd.DatetimeIndex(prepared["timestamp"])
    if not actual.equals(expected):
        raise ValueError("market must be the exact half-open 5m grid [start,end)")
    bars: dict[pd.Timestamp, Bar] = {}
    for row in prepared.itertuples(index=False):
        values = (float(row.open), float(row.high), float(row.low), float(row.close))
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("market OHLC values must be finite and positive")
        low = float(row.low)
        high = float(row.high)
        open_ = float(row.open)
        close = float(row.close)
        if low > min(open_, close, high) or high < max(open_, close, low):
            raise ValueError("market OHLC ordering is invalid")
        bars[pd.Timestamp(row.timestamp)] = Bar(pd.Timestamp(row.timestamp), open_, high, low, close)
    return bars


def _prepare_funding(funding: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[Funding]:
    if funding.empty:
        return []
    times = _time_series(funding, kind="funding")
    mark_col = _column(
        funding,
        (
            "settlement_mark_price",
            "settlement_mark",
            "mark",
            "price",
            "mark_price",
        ),
        kind="funding",
    )
    rate_col = _column(funding, ("funding_rate", "rate"), kind="funding")
    if "symbol" in {str(column).lower() for column in funding.columns}:
        symbol_col = _column(funding, ("symbol",), kind="funding")
        if not funding[symbol_col].astype(str).eq("BTCUSDT").all():
            raise ValueError("funding symbol must be BTCUSDT")
    rows = pd.DataFrame(
        {
            "timestamp": times,
            "mark": pd.to_numeric(funding[mark_col], errors="raise").to_numpy(),
            "rate": pd.to_numeric(funding[rate_col], errors="raise").to_numpy(),
        }
    )
    if not rows["timestamp"].between(start, end, inclusive="left").all():
        raise ValueError("funding rows escape the authorized interval")
    if not rows["timestamp"].is_monotonic_increasing:
        raise ValueError("funding timestamps are not sorted")
    if rows["timestamp"].duplicated().any():
        raise ValueError("funding timestamps are duplicated")
    out: list[Funding] = []
    for row in rows.itertuples(index=False):
        mark = float(row.mark)
        rate = float(row.rate)
        if not math.isfinite(mark) or mark <= 0.0 or not math.isfinite(rate):
            raise ValueError("funding mark/rate values are invalid")
        out.append(Funding(pd.Timestamp(row.timestamp), mark, rate))
    return out


def _target(value: Any) -> float:
    if isinstance(value, str):
        normalized = value.strip().upper()
        mapping = {
            "TARGET_SHORT": -0.5,
            "SHORT": -0.5,
            "-0.5": -0.5,
            "TARGET_FLAT": 0.0,
            "FLAT": 0.0,
            "0": 0.0,
            "0.0": 0.0,
            "TARGET_LONG": 0.5,
            "LONG": 0.5,
            "+0.5": 0.5,
            "0.5": 0.5,
        }
        return mapping.get(normalized, 0.0)
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return 0.0
    return candidate if math.isfinite(candidate) and candidate in ACTIONS else 0.0


def _prepare_schedule(schedule: pd.DataFrame, start: pd.Timestamp, terminal_flat_time: pd.Timestamp) -> list[tuple[pd.Timestamp, float]]:
    if schedule.empty:
        return []
    times = _time_series(schedule, kind="schedule")
    target_col = _column(schedule, ("target", "target_position", "action"), kind="schedule")
    rows = pd.DataFrame(
        {
            "timestamp": times,
            "target": schedule[target_col].to_numpy(),
        }
    )
    if not rows["timestamp"].between(
        start,
        terminal_flat_time,
        inclusive="left",
    ).all():
        raise ValueError("schedule rows escape the executable stage")
    if not rows["timestamp"].is_monotonic_increasing:
        raise ValueError("schedule timestamps are not sorted")
    if rows["timestamp"].duplicated().any():
        raise ValueError("schedule timestamps are duplicated")
    out: list[tuple[pd.Timestamp, float]] = []
    for row in rows.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        if (ts - start) % FIVE_MINUTES != pd.Timedelta(0):
            raise ValueError("schedule decisions must be on the exact 5m grid")
        out.append((ts, _target(row.target)))
    return out


def _boundary_funding(fundings: list[Funding], boundary: pd.Timestamp) -> Funding | None:
    for event in fundings:
        if event.timestamp == boundary:
            return event
    return None


def _event(timestamp: pd.Timestamp, kind: str, equity: float, **extra: Any) -> dict[str, Any]:
    return {"timestamp": timestamp.isoformat(), "kind": kind, "equity": float(equity), **extra}


def _max_drawdown(events: list[dict[str, Any]], initial_hwm: float = 1.0) -> tuple[float, float]:
    hwm = float(initial_hwm)
    mdd = 0.0
    for event in events:
        equity = float(event["equity"])
        if equity > hwm:
            hwm = equity
        if hwm > 0.0:
            mdd = max(mdd, (hwm - equity) / hwm)
    return hwm, mdd


def _infer_target_or_none(pre_equity: float, quantity: float, price: float) -> float | None:
    if pre_equity <= 0.0 or not all(math.isfinite(v) for v in (pre_equity, quantity, price)):
        return None
    ratio = quantity * price / pre_equity
    for action in freeze.ACTIONS:
        if math.isclose(ratio, action, rel_tol=1e-9, abs_tol=1e-9):
            return float(action)
    return None


def simulate_counterfactual_interval(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pre_equity: float,
    old_quantity: float,
    target: float,
    cost_rate: float,
    terminal_flatten: bool,
) -> dict[str, Any]:
    """Simulate one half-open target interval on caller-supplied data.

    The interval rebalances once at ``start`` using the frozen quantity solver,
    owns bars ``[start,end)``, and (unless ``terminal_flatten``) marks the next
    boundary open at ``end``.  Funding at the rebalance boundary uses the frozen
    conservative boundary rule; interior funding is exact and counted once.
    """

    start = _utc_timestamp(start, name="start")
    end = _utc_timestamp(end, name="end")
    if end <= start or (end - start) % FIVE_MINUTES != pd.Timedelta(0):
        raise ValueError("interval must be a positive exact 5m half-open range")
    requested_target = _target(target)
    if not all(math.isfinite(v) for v in (pre_equity, old_quantity, cost_rate)) or pre_equity <= 0.0:
        raise ValueError("account inputs are invalid")
    if not 0.0 <= cost_rate < 0.1:
        raise ValueError("cost_rate is invalid")

    market_end = end + FIVE_MINUTES
    bars = _prepare_market(market, start, market_end)
    funding_end = end + pd.Timedelta(nanoseconds=1) if terminal_flatten else end
    fundings = _prepare_funding(funding, start, funding_end)
    first = bars[start]
    cash = float(pre_equity) - float(old_quantity) * first.open
    events: list[dict[str, Any]] = [_event(start, "boundary_open_old", pre_equity, quantity=float(old_quantity), price=first.open)]

    new_quantity = freeze.solve_target_quantity(
        pre_equity,
        old_quantity,
        first.open,
        cost_rate,
        requested_target,
    )
    effective_target = requested_target
    if requested_target == 0.0:
        new_quantity = 0.0
    elif new_quantity == 0.0:
        effective_target = 0.0
    changed_notional = abs(new_quantity - old_quantity) * first.open
    cost = cost_rate * changed_notional
    cash -= (new_quantity - old_quantity) * first.open
    cash -= cost

    boundary_cash = 0.0
    interior_funding_cash = 0.0
    boundary = _boundary_funding(fundings, start)
    if boundary is not None:
        boundary_cash = freeze.conservative_boundary_funding_cash(
            old_quantity,
            new_quantity,
            boundary.mark,
            boundary.rate,
        )
        cash += boundary_cash
    post_equity = cash + new_quantity * first.open
    events.append(
        _event(
            start,
            "rebalance",
            post_equity,
            old_quantity=float(old_quantity),
            new_quantity=float(new_quantity),
            requested_target=float(requested_target),
            target=float(effective_target),
            cost=float(cost),
            boundary_funding_cash=float(boundary_cash),
        )
    )

    by_bar: dict[pd.Timestamp, list[Funding]] = {}
    for event in fundings:
        if event.timestamp == start:
            continue
        bar_time = start + ((event.timestamp - start) // FIVE_MINUTES) * FIVE_MINUTES
        if start <= bar_time < end:
            by_bar.setdefault(bar_time, []).append(event)

    for bar_time in pd.date_range(start=start, end=end - FIVE_MINUTES, freq=FIVE_MINUTES, tz="UTC"):
        bar = bars[bar_time]
        open_equity = cash + new_quantity * bar.open
        events.append(_event(bar_time, "bar_open", open_equity, quantity=float(new_quantity), price=bar.open))
        for funding_event in by_bar.get(bar_time, []):
            exact_cash = -new_quantity * funding_event.mark * funding_event.rate
            interior_funding_cash += exact_cash
            cash += exact_cash
            marked = cash + new_quantity * funding_event.mark
            liquidating = marked - cost_rate * abs(new_quantity) * funding_event.mark
            events.append(
                _event(
                    funding_event.timestamp,
                    "interior_funding",
                    liquidating,
                    cash_flow=float(exact_cash),
                    mark=funding_event.mark,
                    rate=funding_event.rate,
                    quantity=float(new_quantity),
                    virtual_flatten_cost=float(cost_rate * abs(new_quantity) * funding_event.mark),
                )
            )
        if new_quantity > 0.0:
            favorable_price, adverse_price = bar.high, bar.low
        elif new_quantity < 0.0:
            favorable_price, adverse_price = bar.low, bar.high
        else:
            favorable_price = adverse_price = bar.open
        favorable_equity = cash + new_quantity * favorable_price
        adverse_marked = cash + new_quantity * adverse_price
        virtual_fee = cost_rate * abs(new_quantity) * adverse_price
        adverse_equity = adverse_marked - virtual_fee
        events.append(_event(bar_time, "favorable_ohlc", favorable_equity, price=favorable_price))
        events.append(
            _event(
                bar_time,
                "adverse_ohlc_virtual_flat",
                adverse_equity,
                price=adverse_price,
                virtual_flatten_cost=float(virtual_fee),
            )
        )

    next_bar = bars[end]
    boundary_equity = cash + new_quantity * next_bar.open
    events.append(
        _event(
            end,
            "terminal_boundary_open_old"
            if terminal_flatten
            else "next_boundary_open",
            boundary_equity,
            quantity=float(new_quantity),
            price=next_bar.open,
        )
    )
    terminal_cost = 0.0
    terminal_boundary_funding_cash = 0.0
    ending_quantity = new_quantity
    if terminal_flatten:
        terminal_cost = cost_rate * abs(new_quantity) * next_bar.open
        cash += new_quantity * next_bar.open
        cash -= terminal_cost
        terminal_boundary = _boundary_funding(fundings, end)
        if terminal_boundary is not None:
            terminal_boundary_funding_cash = freeze.conservative_boundary_funding_cash(
                new_quantity,
                0.0,
                terminal_boundary.mark,
                terminal_boundary.rate,
            )
            cash += terminal_boundary_funding_cash
        ending_quantity = 0.0
        end_equity = cash
        events.append(
            _event(
                end,
                "terminal_flatten",
                end_equity,
                old_quantity=float(new_quantity),
                new_quantity=0.0,
                price=next_bar.open,
                terminal_cost=float(terminal_cost),
                boundary_funding_cash=float(terminal_boundary_funding_cash),
            )
        )
    else:
        end_equity = boundary_equity

    _, interval_mdd = _max_drawdown(events, initial_hwm=float(pre_equity))
    downside = max(0.0, max((pre_equity - float(event["equity"])) / pre_equity for event in events))
    funding_cash = boundary_cash + interior_funding_cash + terminal_boundary_funding_cash
    bars_held = int((end - start) / FIVE_MINUTES)
    changed_notional_fraction = changed_notional / pre_equity
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "requested_target": float(requested_target),
        "target": float(effective_target),
        "new_target": float(effective_target),
        "old_target_inferred_or_none": _infer_target_or_none(pre_equity, old_quantity, first.open),
        "old_quantity": float(old_quantity),
        "new_quantity": float(new_quantity),
        "ending_quantity": float(ending_quantity),
        "cost": float(cost),
        "entry_cost": float(cost),
        "terminal_cost": float(terminal_cost),
        "changed_notional_fraction": float(changed_notional_fraction),
        "boundary_funding_cash": float(boundary_cash),
        "interior_funding_cash": float(interior_funding_cash),
        "terminal_boundary_funding_cash": float(terminal_boundary_funding_cash),
        "funding_cash": float(funding_cash),
        "bars_held": bars_held,
        "pre_equity": float(pre_equity),
        "post_rebalance_equity": float(post_equity),
        "end_equity": float(end_equity),
        "ending_equity": float(end_equity),
        "multiplier": float(end_equity / pre_equity),
        "held_path_downside_fraction": float(downside),
        "max_drawdown": float(interval_mdd),
        "events": events,
    }


def _stage_bars_frame(all_bars: dict[pd.Timestamp, Bar], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = [all_bars[ts] for ts in pd.date_range(start=start, end=end - FIVE_MINUTES, freq=FIVE_MINUTES, tz="UTC")]
    return pd.DataFrame(
        {
            "timestamp": [row.timestamp for row in rows],
            "open": [row.open for row in rows],
            "high": [row.high for row in rows],
            "low": [row.low for row in rows],
            "close": [row.close for row in rows],
        }
    )


def _funding_frame(events: list[Funding], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = [event for event in events if start <= event.timestamp < end]
    return pd.DataFrame(
        {
            "timestamp": [event.timestamp for event in rows],
            "settlement_mark": [event.mark for event in rows],
            "funding_rate": [event.rate for event in rows],
        }
    )


def _monday_floor(ts: pd.Timestamp) -> pd.Timestamp:
    ts = _utc_timestamp(ts, name="weekly timestamp")
    return (ts - pd.Timedelta(days=ts.weekday())).normalize()


def _iso_z(ts: pd.Timestamp) -> str:
    return _utc_timestamp(ts, name="output timestamp").isoformat().replace(
        "+00:00",
        "Z",
    )


def _weekly_log_returns(events: list[dict[str, Any]], start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    actual_point_priority = {
        "bar_open": 10,
        "next_boundary_open": 20,
        "terminal_boundary_open_old": 20,
        "boundary_open_old": 30,
        "final_flat_bar_open": 40,
        "split_end_flat_no_row_loaded": 50,
    }
    points: dict[pd.Timestamp, tuple[int, float]] = {start: (0, 1.0)}
    for event in events:
        priority = actual_point_priority.get(str(event["kind"]))
        if priority is None:
            continue
        ts = _utc_timestamp(event["timestamp"], name="event timestamp")
        current = points.get(ts)
        if current is None or priority > current[0]:
            points[ts] = (priority, float(event["equity"]))
    if end not in points:
        points[end] = (100, float(events[-1]["equity"]))
    boundaries = [start]
    first_monday = _monday_floor(start)
    if first_monday <= start:
        first_monday += pd.Timedelta(days=7)
    current = first_monday
    while current < end:
        if current in points:
            boundaries.append(current)
        current += pd.Timedelta(days=7)
    boundaries.append(end)
    out: list[dict[str, Any]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if left == right:
            continue
        left_equity = points[left][1]
        right_equity = points[right][1]
        out.append(
            {
                "week_start": _iso_z(_monday_floor(left)),
                "start": _iso_z(left),
                "end": _iso_z(right),
                "log_return": float(math.log(max(right_equity / left_equity, 1e-300))),
            }
        )
    return out


def simulate_target_schedule(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_rate: float,
) -> dict[str, Any]:
    """Simulate a strict target-position schedule over a full split calendar."""

    start = _utc_timestamp(start, name="start")
    end = _utc_timestamp(end, name="end")
    if end <= start + FIVE_MINUTES or (end - start) % FIVE_MINUTES != pd.Timedelta(0):
        raise ValueError("schedule simulation requires an exact 5m calendar with a final flat bar")
    if not math.isfinite(cost_rate) or not 0.0 <= cost_rate < 0.1:
        raise ValueError("cost_rate is invalid")
    terminal_flat_time = end - FIVE_MINUTES
    all_bars = _prepare_market(market, start, end)
    all_funding = _prepare_funding(funding, start, end)
    decisions = _prepare_schedule(schedule, start, terminal_flat_time)

    boundaries: list[tuple[pd.Timestamp, float]] = []
    current = start
    if decisions and decisions[0][0] > start:
        boundaries.append((start, 0.0))
    for decision in decisions:
        if decision[0] < current:
            raise ValueError("schedule decisions are not sorted")
        boundaries.append(decision)
        current = decision[0]
    if not boundaries:
        boundaries.append((start, 0.0))

    cash = 1.0
    quantity = 0.0
    equity = 1.0
    intervals: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []

    decision_times = [ts for ts, _ in boundaries]
    for index, (decision_time, target) in enumerate(boundaries):
        next_time = decision_times[index + 1] if index + 1 < len(decision_times) else terminal_flat_time
        if next_time <= decision_time:
            continue
        interval_market = _stage_bars_frame(all_bars, decision_time, min(next_time + FIVE_MINUTES, end))
        funding_end = next_time + pd.Timedelta(nanoseconds=1) if next_time == terminal_flat_time else next_time
        interval_funding = _funding_frame(all_funding, decision_time, funding_end)
        result = simulate_counterfactual_interval(
            interval_market,
            interval_funding,
            start=decision_time,
            end=next_time,
            pre_equity=equity,
            old_quantity=quantity,
            target=target,
            cost_rate=cost_rate,
            terminal_flatten=(next_time == terminal_flat_time),
        )
        all_events.extend(result["events"])
        intervals.append(
            {
                key: value
                for key, value in result.items()
                if key != "events"
            }
        )
        quantity = float(result["ending_quantity"])
        equity = float(result["ending_equity"])
        boundary_open = all_bars[next_time].open
        cash = equity - quantity * boundary_open

    final_equity = equity
    all_events.append(_event(terminal_flat_time, "final_flat_bar_open", final_equity, quantity=0.0))
    all_events.append(_event(terminal_flat_time, "final_flat_bar_favorable", final_equity, quantity=0.0))
    all_events.append(_event(terminal_flat_time, "final_flat_bar_adverse", final_equity, quantity=0.0))
    all_events.append(_event(end, "split_end_flat_no_row_loaded", final_equity, quantity=0.0))
    hwm, mdd = _max_drawdown(all_events, initial_hwm=1.0)
    years = (end - start).total_seconds() / SECONDS_PER_YEAR
    if years > 0.0 and final_equity > 0.0:
        try:
            cagr = math.expm1(math.log(final_equity) / years)
        except OverflowError:
            cagr = float("inf")
    else:
        cagr = float("nan")
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "terminal_flat_time": terminal_flat_time.isoformat(),
        "cost_rate": float(cost_rate),
        "initial_equity": 1.0,
        "final_equity": final_equity,
        "total_return": final_equity - 1.0,
        "cagr": float(cagr),
        "high_water_mark": float(hwm),
        "max_drawdown": float(mdd),
        "intervals": intervals,
        "weekly_log_returns": _weekly_log_returns(all_events, start, end),
    }
