from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import training.evaluate_rv20_low_participation_shock_reversal as evaluator


def _cell(**changes):
    base = evaluator.Cell(0.8, 0.42, 0.75, 6, 0.5, 0.6, 0.02, 0.04)
    return replace(base, **changes)


def _market(count=81, *, high=101.0, low=99.0, close=100.0):
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=count, freq="5min"),
        "open": np.full(count, 100.0),
        "high": np.full(count, high),
        "low": np.full(count, low),
        "close": np.full(count, close),
    })


def test_causal_quantile_is_shifted_and_uses_only_preceding_finite_values():
    values = pd.Series([1.0, 2.0, np.nan, 100.0, 4.0])
    result = evaluator.shifted_rolling_quantile(values, 0.5, 2)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert np.isnan(result.iloc[2])
    assert result.iloc[3] == pytest.approx(1.5)  # excludes current 100
    assert result.iloc[4] == pytest.approx(51.0)  # preceding finite 2,100


def test_adverse_barrier_wins_when_take_and_stop_touch_same_bar():
    market = _market(high=106.0, low=97.0)
    trade = evaluator.execute_trade(
        market, pd.DataFrame(columns=["date", "funding_rate"]), 0, 1,
        _cell(max_hold_hours=1),
    )
    assert trade is not None
    assert trade.exit_kind == "stop"
    assert trade.exit_position == 1
    assert trade.price_factor == pytest.approx(0.98)


@pytest.mark.parametrize(
    ("side", "gap_open", "expected"),
    [(1, 95.0, 0.95), (-1, 105.0, 0.95)],
)
def test_gap_through_stop_exits_at_worse_bar_open(side, gap_open, expected):
    market = _market(count=20)
    market.loc[2:, "open"] = gap_open
    trade = evaluator.execute_trade(
        market, pd.DataFrame(columns=["date", "funding_rate"]), 0, side,
        _cell(max_hold_hours=1),
    )
    assert trade is not None
    assert trade.exit_kind == "gap_stop"
    assert trade.exit_position == 2
    assert trade.price_factor == pytest.approx(expected)


def test_cagr_uses_full_calendar_including_idle_time():
    trade = evaluator.Trade(
        0, 1, 2, 1, "take", "2024-06-01", "2024-06-01T00:05:00",
        1.10, 1.0, 1.10, 1.0,
    )
    result = evaluator.strict_metrics(
        [trade], start="2024-01-01", end="2025-01-01", cost_rate=0.0
    )
    years = evaluator.years_between("2024-01-01", "2025-01-01")
    assert result["cagr_pct"] == pytest.approx((1.1 ** (1 / years) - 1) * 100)


def test_split_metrics_exclude_trade_that_exits_on_next_split_boundary():
    trade = evaluator.Trade(
        0, 1, 2, 1, "time", "2024-12-31T23:55:00",
        "2025-01-01T00:00:00", 1.10, 1.0, 1.10, 1.0,
    )
    result = evaluator.strict_metrics(
        [trade], start="2024-01-01", end="2025-01-01", cost_rate=0.0
    )
    assert result["trades"] == 0
    assert result["absolute_return_pct"] == 0.0
    assert evaluator._split_trades(
        [trade], "2024-01-01", "2025-01-01"
    ) == []


def test_strict_mdd_retains_global_hwm_and_orders_favorable_before_adverse():
    trade = evaluator.Trade(
        0, 1, 2, 1, "time", "2024-01-02", "2024-01-02T00:05:00",
        1.0, 1.0, 1.10, 0.90,
    )
    result = evaluator.strict_metrics(
        [trade], start="2024-01-01", end="2025-01-01", cost_rate=0.0
    )
    assert result["strict_mdd_pct"] == pytest.approx((1 - 0.9 / 1.1) * 100)


def test_frozen_grid_cardinality_and_canonical_uniqueness():
    prereg = evaluator.load_preregistration()
    grid = evaluator.frozen_grid(prereg)
    assert len(grid) == 384
    assert len(set(grid)) == 384


def test_hourly_features_use_completed_hour_and_one_hour_oi_change():
    dates = pd.date_range("2024-01-01", periods=24, freq="5min")
    market = pd.DataFrame({
        "date": dates,
        "open": np.r_[np.full(12, 100.0), np.full(12, 110.0)],
        "high": np.r_[np.full(12, 102.0), np.full(12, 113.0)],
        "low": np.r_[np.full(12, 99.0), np.full(12, 109.0)],
        "close": np.r_[np.full(11, 100.0), 101.0, np.full(11, 110.0), 112.0],
        "quote_asset_volume": np.ones(24),
        "taker_buy_quote": np.r_[np.full(12, 0.6), np.full(12, 0.4)],
        "open_interest": np.r_[np.full(12, 100.0), np.full(12, 110.0)],
    })
    hourly = evaluator.build_hourly_features(market)
    assert len(hourly) == 2
    assert np.isnan(hourly.loc[0, "oi_change_1h"])
    assert hourly.loc[1, "oi_change_1h"] == pytest.approx(np.log(1.1))
    assert hourly.loc[0, "taker_fraction"] == pytest.approx(0.6)
    assert hourly.loc[1, "hour_return"] == pytest.approx(np.log(1.12 / 1.10))


@pytest.mark.parametrize("column", ["quote_asset_volume", "taker_buy_quote"])
def test_hour_with_missing_flow_bar_is_inactive(column):
    dates = pd.date_range("2024-01-01", periods=12, freq="5min")
    market = pd.DataFrame({
        "date": dates, "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "quote_asset_volume": 1.0,
        "taker_buy_quote": 0.6, "open_interest": 100.0,
    })
    market.loc[5, column] = np.nan
    assert evaluator.build_hourly_features(market).empty


def test_missing_reindexed_hour_is_inactive_without_signal_position_crash(monkeypatch):
    market = _market(count=300)
    hourly = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "signal_position": [np.nan]})
    monkeypatch.setattr(
        evaluator, "cell_signal_sides", lambda _hourly, _cell: np.array([0])
    )
    assert evaluator.execute_schedule_window(
        market, pd.DataFrame(), hourly, _cell(),
        start="2024-01-01", end="2024-01-02",
    ) == []


def test_daily_rv20_regime_quantiles_are_shifted_and_cell_selectable():
    days = pd.date_range("2020-01-01 23:55", periods=800, freq="D")
    returns = 0.001 + np.arange(800) * 0.000001
    close = 100.0 * np.exp(np.cumsum(returns))
    clock = evaluator.build_rv20_daily_clock(
        pd.DataFrame({"date": days, "close": close})
    )
    first = clock["rv20_q60"].first_valid_index()
    assert first == pd.Timestamp("2022-02-16")
    preceding = clock.loc[: first - pd.Timedelta(days=1), "rv20"].dropna().iloc[-756:]
    assert clock.loc[first, "rv20_q60"] == pytest.approx(
        preceding.quantile(0.6, interpolation="linear")
    )
    assert clock.loc[first, "rv20_q75"] == pytest.approx(
        preceding.quantile(0.75, interpolation="linear")
    )


def test_reversal_signals_require_selected_regime_low_oi_and_extreme_taker_flow():
    hourly = pd.DataFrame({
        "rv20": [0.70] * 6,
        "rv20_q60": [0.60] * 6,
        "rv20_q75": [0.80] * 6,
        "hour_return": [-0.03, 0.03, -0.03, 0.03, -0.03, 0.03],
        "hour_range": [0.04] * 6,
        "oi_change_1h": [-0.01, -0.01, 0.02, -0.01, -0.01, -0.01],
        "taker_fraction": [0.40, 0.60, 0.40, 0.40, 0.50, 0.60],
        "q_abs_return_0.8": [0.02] * 6,
        "q_range_0.75": [0.03] * 6,
        "q_oi_0.5": [0.0] * 6,
    })
    assert evaluator.cell_signal_sides(hourly, _cell()).tolist() == [1, -1, 0, 0, 0, -1]
    assert evaluator.cell_signal_sides(
        hourly, _cell(rv20_regime_quantile=0.75)
    ).tolist() == [0] * 6


def test_train_gate_uses_preregistered_40_trade_20_week_minimums():
    normal = {
        "trades": 40, "active_weeks": 20, "absolute_return_pct": 1.0,
        "strict_mdd_pct": 20.0, "cagr_to_strict_mdd": 1.5,
    }
    stress = {"absolute_return_pct": 0.01}
    assert evaluator.train_cell_passes(normal, stress)
    assert not evaluator.train_cell_passes({**normal, "trades": 39}, stress)
    assert not evaluator.train_cell_passes({**normal, "active_weeks": 19}, stress)


def test_persistent_long_vol_control_uses_selected_cell_regime_quantile():
    market = _market(count=24)
    market["date"] = pd.date_range("2024-01-01", periods=24, freq="5min")
    daily = pd.DataFrame(
        {"rv20": [0.70], "rv20_q60": [0.60], "rv20_q75": [0.80]},
        index=[pd.Timestamp("2024-01-01")],
    )
    funding = pd.DataFrame(columns=["date", "funding_rate"])
    q60 = evaluator.persistent_long_vol_control(
        market, funding, daily, regime_quantile=0.6, leverage=1.0,
        cost_rate=evaluator.ENTRY_COST,
    )
    q75 = evaluator.persistent_long_vol_control(
        market, funding, daily, regime_quantile=0.75, leverage=1.0,
        cost_rate=evaluator.ENTRY_COST,
    )
    assert len(q60) == 1
    assert q75 == []


def test_structural_rank_tie_break_is_deterministic():
    cell_a = _cell(stop_loss_pct=0.02)
    cell_b = _cell(stop_loss_pct=0.04)
    metric = {
        "cagr_to_strict_mdd": 2.0,
        "strict_mdd_pct": 10.0,
        "absolute_return_pct": 20.0,
    }
    rows = [
        {"cell": evaluator.asdict(cell_b), "train": metric, "train_pass": True},
        {"cell": evaluator.asdict(cell_a), "train": metric, "train_pass": True},
    ]
    assert evaluator.select_structural_top1(rows)["cell"] == evaluator.asdict(cell_a)


def test_preregistration_hash_rejects_even_semantically_equivalent_rewrite(tmp_path):
    payload = json.loads(evaluator.PREREGISTRATION.read_text())
    rewritten = tmp_path / "prereg.json"
    rewritten.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash drifted"):
        evaluator.load_preregistration(rewritten)


def test_input_hash_rejection_happens_before_any_csv_decode(monkeypatch):
    prereg = {"inputs": {name: f"/{name}" for name in evaluator.INPUT_SHA256}}
    monkeypatch.setattr(evaluator, "sha256_file", lambda _path: "wrong")
    monkeypatch.setattr(
        evaluator.pd, "read_csv",
        lambda *args, **kwargs: pytest.fail("decoded before authentication"),
    )
    with pytest.raises(RuntimeError, match="input hash drifted"):
        evaluator.load_inputs(prereg)


def test_gross9_prefix_is_zero_padded_to_complete_clock():
    complete = pd.DataFrame({
        "date": pd.date_range("2025-12-31 23:50", periods=4, freq="5min"),
        "open": [1.0] * 4, "high": [2.0] * 4, "low": [0.5] * 4,
        "close": [1.0] * 4,
    })
    gross = complete.iloc[:2].copy()
    masks = {}
    dates = pd.DatetimeIndex(gross["date"])
    for split, (start, end) in evaluator.portfolio.SPLIT_BOUNDS.items():
        masks[split] = np.asarray(
            (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)),
            dtype=bool,
        )
    event = {
        "split": "eval2025", "sleeve": "x", "ret": np.array([0.1, 0.2]),
        "adv": np.array([-0.1, -0.2]), "fav": np.array([0.2, 0.3]),
        "low": np.array([-0.1, -0.2]), "high": np.array([0.2, 0.3]),
    }
    _, padded_masks, events = evaluator.extend_gross9_context(
        gross, masks, [event], complete
    )
    assert events[0]["ret"].tolist() == [0.1, 0.2, 0.0, 0.0]
    assert padded_masks["eval2025"].tolist() == [True, True, False, False]
    assert padded_masks["ytd2026"].tolist() == [False, False, True, True]


def test_candidate_bar_path_adapter_preserves_exact_trade_final_equity():
    market = _market(count=6, close=100.0)
    funding = pd.DataFrame(columns=["date", "funding_rate"])
    trade = evaluator.Trade(
        signal_position=0, entry_position=1, exit_position=2, side=1,
        exit_kind="take", entry_time=str(market.iloc[1]["date"]),
        exit_time=str(market.iloc[2]["date"]), price_factor=1.02,
        funding_factor=1.0, favorable_price_factor=1.02,
        adverse_price_factor=1.0,
    )
    prereg = {"inputs": {
        "market": "market", "open_interest": "oi", "funding": "funding",
        "gross9_anchor": "anchor",
    }}
    events = evaluator.candidate_events(
        market, funding, [trade], prereg, leverage=1.0,
        cost_rate=evaluator.ENTRY_COST, hold_bars=12,
    )
    event = next(row for row in events if row["split"] == "test2024")
    expected = (1.0 - evaluator.ENTRY_COST) ** 2 * 1.02 - 1.0
    assert np.prod(1.0 + event["ret"]) - 1.0 == pytest.approx(expected)
    assert event["entry_positions"] == [1]


def test_cross_boundary_trade_is_excluded_without_suppressing_next_window(monkeypatch):
    market = _market(count=30)
    market["date"] = pd.date_range("2023-12-31 23:00", periods=30, freq="5min")
    hourly = pd.DataFrame({
        "signal_position": [10, 13],
        "date": [market.iloc[10]["date"], market.iloc[13]["date"]],
    })
    monkeypatch.setattr(
        evaluator, "cell_signal_sides", lambda _hourly, _cell: np.array([1, 1])
    )

    def fake_trade(_market, _funding, signal_position, side, _cell, *, leverage=1.0):
        entry = signal_position + 1
        exit_ = entry + (4 if signal_position == 10 else 1)
        return evaluator.Trade(
            signal_position, entry, exit_, side, "time",
            str(market.iloc[entry]["date"]), str(market.iloc[exit_]["date"]),
            1.01, 1.0, 1.01, 1.0,
        )

    monkeypatch.setattr(evaluator, "execute_trade", fake_trade)
    rows = evaluator.execute_schedule_window(
        market, pd.DataFrame(), hourly, _cell(),
        start="2024-01-01", end="2024-01-02",
    )
    assert [trade.signal_position for trade in rows] == [13]


def test_late_crossing_trade_suppresses_all_later_window_signals(monkeypatch):
    market = _market(count=30)
    market["date"] = pd.date_range("2024-01-01 22:00", periods=30, freq="5min")
    hourly = pd.DataFrame({
        "signal_position": [20, 22],
        "date": [market.iloc[20]["date"], market.iloc[22]["date"]],
    })
    monkeypatch.setattr(
        evaluator, "cell_signal_sides", lambda _hourly, _cell: np.array([1, 1])
    )

    def fake_trade(_market, _funding, signal_position, side, _cell, *, leverage=1.0):
        entry = signal_position + 1
        exit_ = 29 if signal_position == 20 else entry + 1
        return evaluator.Trade(
            signal_position, entry, exit_, side, "time",
            str(market.iloc[entry]["date"]), str(market.iloc[exit_]["date"]),
            1.01, 1.0, 1.01, 1.0,
        )

    monkeypatch.setattr(evaluator, "execute_trade", fake_trade)
    rows = evaluator.execute_schedule_window(
        market, pd.DataFrame(), hourly, _cell(),
        start="2024-01-01", end="2024-01-02",
    )
    assert rows == []


def test_one_shot_marker_and_outputs_fail_closed(tmp_path):
    marker = tmp_path / "attempt.json"
    output = tmp_path / "result.json"
    report = tmp_path / "report.md"
    evaluator.consume_one_shot(marker, output, report)
    assert marker.stat().st_mode & 0o777 == 0o400
    with pytest.raises(RuntimeError, match="already consumed"):
        evaluator.consume_one_shot(marker, output, report)


def test_candidate_adapter_does_not_leak_cross_boundary_trade_outcome():
    market = _market(count=4)
    market["date"] = pd.date_range(
        "2024-12-31 23:50", periods=4, freq="5min"
    )
    trade = evaluator.Trade(
        signal_position=0, entry_position=1, exit_position=2, side=1,
        exit_kind="time", entry_time=str(market.iloc[1]["date"]),
        exit_time=str(market.iloc[2]["date"]), price_factor=1.10,
        funding_factor=1.0, favorable_price_factor=1.10,
        adverse_price_factor=1.0,
    )
    prereg = {"inputs": {
        "market": "market", "open_interest": "oi", "funding": "funding",
        "gross9_anchor": "anchor",
    }}
    events = evaluator.candidate_events(
        market, pd.DataFrame(columns=["date", "funding_rate"]), [trade],
        prereg, leverage=1.0, cost_rate=evaluator.ENTRY_COST, hold_bars=1,
    )
    assert all(event["trade_count"] == 0 for event in events)
    assert all(np.count_nonzero(event["ret"]) == 0 for event in events)


def test_persistent_long_vol_gate_requires_normal_and_stress_residuals():
    passing = {
        "full_calendar": {"log_return_residual": 0.01},
        "rv20_stress": {"log_return_residual": 0.001},
    }
    assert evaluator.residual_report_passes(passing)
    failing = {
        **passing,
        "rv20_stress": {"log_return_residual": -0.001},
    }
    assert not evaluator.residual_report_passes(failing)
