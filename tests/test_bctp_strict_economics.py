import math

import pandas as pd
import pytest

from training.bctp_strict_economics import (
    FIVE_MINUTES,
    _weekly_log_returns,
    simulate_counterfactual_interval,
    simulate_target_schedule,
)


def market(
    start: pd.Timestamp,
    opens: list[float],
    highs=None,
    lows=None,
) -> pd.DataFrame:
    highs = opens if highs is None else highs
    lows = opens if lows is None else lows
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(opens), freq="5min", tz="UTC"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": opens,
        }
    )


def empty_funding() -> pd.DataFrame:
    return pd.DataFrame({"timestamp": [], "settlement_mark": [], "funding_rate": []})


def solved_notional(
    pre_equity: float,
    old_notional: float,
    target: float,
    cost_rate: float,
) -> float:
    if target == 0.0:
        return 0.0
    ge = target * (pre_equity + cost_rate * old_notional) / (1.0 + target * cost_rate)
    le = target * (pre_equity - cost_rate * old_notional) / (1.0 - target * cost_rate)
    if ge >= old_notional - 1e-12:
        return ge
    return le


def test_flat_schedule_stays_at_one_and_uses_no_post_split_row() -> None:
    start = pd.Timestamp("2026-01-05T00:00:00Z")
    end = start + 3 * FIVE_MINUTES
    result = simulate_target_schedule(
        market(start, [100.0, 101.0, 102.0]),
        empty_funding(),
        pd.DataFrame({"timestamp": [start], "target": [0.0]}),
        start=start,
        end=end,
        cost_rate=0.001,
    )
    assert result["terminal_flat_time"] == (end - FIVE_MINUTES).isoformat()
    assert result["final_equity"] == pytest.approx(1.0)
    assert result["max_drawdown"] == pytest.approx(0.0)
    assert result["intervals"][-1]["ending_quantity"] == 0.0
    assert result["intervals"][-1]["terminal_cost"] == 0.0
    assert result["intervals"][-1]["bars_held"] == 2
    assert result["weekly_log_returns"][0]["week_start"].endswith("00:00:00Z")


def test_long_short_and_reversal_charge_one_changed_quantity_cost() -> None:
    start = pd.Timestamp("2026-01-05T00:00:00Z")
    prices = market(start, [100.0, 100.0, 100.0])
    long_result = simulate_counterfactual_interval(
        prices,
        empty_funding(),
        start=start,
        end=start + 2 * FIVE_MINUTES,
        pre_equity=1.0,
        old_quantity=0.0,
        target=0.5,
        cost_rate=0.001,
        terminal_flatten=False,
    )
    long_notional = solved_notional(1.0, 0.0, 0.5, 0.001)
    assert long_result["new_quantity"] == pytest.approx(long_notional / 100.0)
    assert long_result["entry_cost"] == pytest.approx(0.001 * long_notional)
    assert long_result["ending_quantity"] == pytest.approx(long_result["new_quantity"])

    short_result = simulate_counterfactual_interval(
        prices,
        empty_funding(),
        start=start,
        end=start + 2 * FIVE_MINUTES,
        pre_equity=1.0,
        old_quantity=0.0,
        target=-0.5,
        cost_rate=0.001,
        terminal_flatten=False,
    )
    short_notional = solved_notional(1.0, 0.0, -0.5, 0.001)
    assert short_result["new_quantity"] == pytest.approx(short_notional / 100.0)
    assert short_result["entry_cost"] == pytest.approx(0.001 * abs(short_notional))

    reversal = simulate_counterfactual_interval(
        prices,
        empty_funding(),
        start=start,
        end=start + 2 * FIVE_MINUTES,
        pre_equity=1.0,
        old_quantity=long_result["new_quantity"],
        target=-0.5,
        cost_rate=0.001,
        terminal_flatten=False,
    )
    new_notional = solved_notional(1.0, long_notional, -0.5, 0.001)
    changed = abs(new_notional - long_notional)
    assert reversal["entry_cost"] == pytest.approx(0.001 * changed)
    assert reversal["changed_notional_fraction"] == pytest.approx(changed)


def test_same_target_resize_pays_actual_quantity_change() -> None:
    start = pd.Timestamp("2026-01-05T00:00:00Z")
    old_quantity = solved_notional(1.0, 0.0, 0.5, 0.001) / 100.0
    result = simulate_counterfactual_interval(
        market(start, [120.0, 120.0, 120.0]),
        empty_funding(),
        start=start,
        end=start + 2 * FIVE_MINUTES,
        pre_equity=1.0,
        old_quantity=old_quantity,
        target=0.5,
        cost_rate=0.001,
        terminal_flatten=False,
    )
    old_notional = old_quantity * 120.0
    new_notional = solved_notional(1.0, old_notional, 0.5, 0.001)
    assert result["new_quantity"] == pytest.approx(new_notional / 120.0)
    assert result["entry_cost"] == pytest.approx(0.001 * abs(new_notional - old_notional))
    assert result["old_target_inferred_or_none"] is None


def test_boundary_funding_signs_are_conservative_and_interior_is_exact_once() -> None:
    start = pd.Timestamp("2026-01-05T00:00:00Z")
    prices = market(start, [100.0, 100.0, 100.0])
    boundary = pd.DataFrame(
        {"timestamp": [start], "settlement_mark": [100.0], "funding_rate": [0.01]}
    )
    from_short_to_long = simulate_counterfactual_interval(
        prices,
        boundary,
        start=start,
        end=start + 2 * FIVE_MINUTES,
        pre_equity=1.0,
        old_quantity=-0.005,
        target=0.5,
        cost_rate=0.001,
        terminal_flatten=False,
    )
    # Old short would receive +0.005, new long pays a debit; keep only the debit.
    assert from_short_to_long["boundary_funding_cash"] < 0.0
    assert from_short_to_long["funding_cash"] == pytest.approx(
        from_short_to_long["boundary_funding_cash"]
    )

    interior_time = start + pd.Timedelta(minutes=3)
    interior = pd.DataFrame(
        {"timestamp": [interior_time], "settlement_mark": [100.0], "funding_rate": [0.01]}
    )
    held = simulate_counterfactual_interval(
        prices,
        interior,
        start=start,
        end=start + 2 * FIVE_MINUTES,
        pre_equity=1.0,
        old_quantity=0.0,
        target=0.5,
        cost_rate=0.001,
        terminal_flatten=False,
    )
    expected_cash = -held["new_quantity"] * 100.0 * 0.01
    assert held["boundary_funding_cash"] == 0.0
    assert held["interior_funding_cash"] == pytest.approx(expected_cash)
    assert held["funding_cash"] == pytest.approx(expected_cash)
    assert len([e for e in held["events"] if e["kind"] == "interior_funding"]) == 1


def test_favorable_then_adverse_path_gap_and_terminal_flatten_cost() -> None:
    start = pd.Timestamp("2026-01-05T00:00:00Z")
    prices = market(start, [100.0, 110.0], highs=[120.0, 110.0], lows=[90.0, 110.0])
    terminal_funding = pd.DataFrame(
        {
            "timestamp": [start + FIVE_MINUTES],
            "settlement_mark": [110.0],
            "funding_rate": [0.01],
        }
    )
    result = simulate_counterfactual_interval(
        prices,
        terminal_funding,
        start=start,
        end=start + FIVE_MINUTES,
        pre_equity=1.0,
        old_quantity=0.0,
        target=0.5,
        cost_rate=0.001,
        terminal_flatten=True,
    )
    q = result["new_quantity"]
    entry_cost = 0.001 * q * 100.0
    cash_after_entry = 1.0 - q * 100.0 - entry_cost
    adverse = cash_after_entry + q * 90.0 - 0.001 * q * 90.0
    terminal_funding_cash = -q * 110.0 * 0.01
    ending = cash_after_entry + q * 110.0 - 0.001 * q * 110.0 + terminal_funding_cash
    kinds = [event["kind"] for event in result["events"]]
    assert kinds.index("favorable_ohlc") < kinds.index("adverse_ohlc_virtual_flat")
    assert result["held_path_downside_fraction"] == pytest.approx(1.0 - adverse)
    assert result["terminal_cost"] == pytest.approx(0.001 * q * 110.0)
    assert result["terminal_boundary_funding_cash"] == pytest.approx(terminal_funding_cash)
    assert result["funding_cash"] == pytest.approx(terminal_funding_cash)
    assert result["ending_equity"] == pytest.approx(ending)
    assert result["ending_quantity"] == 0.0
    assert result["multiplier"] == pytest.approx(ending)


def test_schedule_global_hwm_mdd_full_calendar_cagr_and_terminal_flatten() -> None:
    start = pd.Timestamp("2026-01-05T00:00:00Z")
    end = start + 3 * FIVE_MINUTES
    result = simulate_target_schedule(
        market(
            start,
            [100.0, 100.0, 90.0],
            highs=[120.0, 100.0, 90.0],
            lows=[80.0, 100.0, 90.0],
        ),
        empty_funding(),
        pd.DataFrame({"timestamp": [start], "target": [0.5]}),
        start=start,
        end=end,
        cost_rate=0.001,
    )
    assert result["intervals"][-1]["end"] == (end - FIVE_MINUTES).isoformat()
    assert result["intervals"][-1]["terminal_cost"] > 0.0
    assert result["intervals"][-1]["ending_quantity"] == 0.0
    years = (end - start).total_seconds() / (365.2425 * 86400.0)
    assert result["cagr"] == pytest.approx(result["final_equity"] ** (1.0 / years) - 1.0)
    assert result["max_drawdown"] > 0.0


def test_market_grid_rejects_post_split_row_and_missing_row() -> None:
    start = pd.Timestamp("2026-01-05T00:00:00Z")
    end = start + 2 * FIVE_MINUTES
    too_many = market(start, [100.0, 100.0, 100.0])
    with pytest.raises(ValueError, match="exact half-open 5m grid"):
        simulate_target_schedule(
            too_many,
            empty_funding(),
            pd.DataFrame({"timestamp": [start], "target": [0.0]}),
            start=start,
            end=end,
            cost_rate=0.001,
        )
    with pytest.raises(ValueError, match="exact half-open 5m grid"):
        simulate_target_schedule(
            too_many.iloc[:1],
            empty_funding(),
            pd.DataFrame({"timestamp": [start], "target": [0.0]}),
            start=start,
            end=end,
            cost_rate=0.001,
        )


def test_naive_timestamps_and_out_of_stage_schedule_fail_closed() -> None:
    start = pd.Timestamp("2026-01-05T00:00:00Z")
    end = start + 3 * FIVE_MINUTES
    naive = market(start, [100.0, 100.0, 100.0])
    naive["timestamp"] = naive["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone aware"):
        simulate_target_schedule(
            naive,
            empty_funding(),
            pd.DataFrame({"timestamp": [start], "target": [0.0]}),
            start=start,
            end=end,
            cost_rate=0.001,
        )
    with pytest.raises(ValueError, match="escape"):
        simulate_target_schedule(
            market(start, [100.0, 100.0, 100.0]),
            empty_funding(),
            pd.DataFrame(
                {
                    "timestamp": [start, end],
                    "target": [0.0, 0.5],
                }
            ),
            start=start,
            end=end,
            cost_rate=0.001,
        )


def test_real_funding_aliases_and_symbol_validation() -> None:
    start = pd.Timestamp("2026-01-05T00:00:00Z")
    prices = market(start, [100.0, 100.0, 100.0])
    real_schema = pd.DataFrame(
        {
            "funding_time_utc": [start],
            "symbol": ["BTCUSDT"],
            "funding_rate": [0.01],
            "settlement_mark_price": [100.0],
        }
    )
    result = simulate_counterfactual_interval(
        prices,
        real_schema,
        start=start,
        end=start + 2 * FIVE_MINUTES,
        pre_equity=1.0,
        old_quantity=0.0,
        target=0.5,
        cost_rate=0.001,
        terminal_flatten=False,
    )
    assert result["boundary_funding_cash"] < 0.0
    wrong_symbol = real_schema.copy()
    wrong_symbol["symbol"] = "ETHUSDT"
    with pytest.raises(ValueError, match="BTCUSDT"):
        simulate_counterfactual_interval(
            prices,
            wrong_symbol,
            start=start,
            end=start + 2 * FIVE_MINUTES,
            pre_equity=1.0,
            old_quantity=0.0,
            target=0.5,
            cost_rate=0.001,
            terminal_flatten=False,
        )


def test_weekly_returns_ignore_virtual_adverse_marks() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    monday = pd.Timestamp("2026-01-05T00:00:00Z")
    end = pd.Timestamp("2026-01-12T00:00:00Z")
    events = [
        {
            "timestamp": start.isoformat(),
            "kind": "boundary_open_old",
            "equity": 1.0,
        },
        {
            "timestamp": monday.isoformat(),
            "kind": "bar_open",
            "equity": 1.1,
        },
        {
            "timestamp": monday.isoformat(),
            "kind": "adverse_ohlc_virtual_flat",
            "equity": 0.2,
        },
        {
            "timestamp": end.isoformat(),
            "kind": "split_end_flat_no_row_loaded",
            "equity": 1.21,
        },
    ]
    weekly = _weekly_log_returns(events, start, end)
    assert weekly[0]["log_return"] == pytest.approx(math.log(1.1))
    assert weekly[1]["log_return"] == pytest.approx(math.log(1.1))
