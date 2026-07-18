from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_price_memory_cage_escape_pre2024 as evaluator
from training.preregister_price_memory_cage_escape_alpha import Candidate
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, Trade


def _stats(**overrides):
    output = {
        "absolute_return_pct": 5.0,
        "cagr_to_strict_mdd": 4.0,
        "strict_mdd_pct": 10.0,
        "trades": 60,
        "mean_gross_bps": 35.0,
        "mean_net_bps": 23.0,
        "weekly_cluster_sign_flip": {"p_value_one_sided": 0.05},
    }
    output.update(overrides)
    return output


def _windows():
    return {name: _stats(trades=12) for name in evaluator.WINDOWS} | {
        "fit": _stats(trades=60),
        "select_2023": _stats(trades=40),
    }


def test_selection_gates_require_fit_select_stress_direction_and_timing() -> None:
    windows = _windows()
    stress = {name: _stats() for name in evaluator.FULL_WINDOWS}
    flip = {
        name: _stats(absolute_return_pct=-5.0)
        for name in evaluator.FULL_WINDOWS
    }
    delay = {
        name: _stats(absolute_return_pct=1.0)
        for name in evaluator.FULL_WINDOWS
    }
    assert all(evaluator.selection_gates(windows, stress, flip, delay).values())

    stress["select_2023"] = _stats(absolute_return_pct=-0.1)
    failed = evaluator.selection_gates(windows, stress, flip, delay)
    assert not failed["select_10bp_per_side_stress_positive"]
    flip["fit"] = _stats(absolute_return_pct=1.0)
    failed = evaluator.selection_gates(windows, {name: _stats() for name in evaluator.FULL_WINDOWS}, flip, delay)
    assert not failed["fit_direction_flip_negative"]
    delay["select_2023"] = _stats(absolute_return_pct=6.0)
    failed = evaluator.selection_gates(
        windows,
        {name: _stats() for name in evaluator.FULL_WINDOWS},
        {name: _stats(absolute_return_pct=-5.0) for name in evaluator.FULL_WINDOWS},
        delay,
    )
    assert not failed["select_beats_24h_delay"]


def test_winner_sort_prefers_minimum_fit_select_ratio() -> None:
    balanced = {
        "name": "b",
        "windows": _windows(),
    }
    balanced["windows"]["fit"] = _stats(cagr_to_strict_mdd=3.5)
    balanced["windows"]["select_2023"] = _stats(cagr_to_strict_mdd=3.6)
    lopsided = {
        "name": "a",
        "windows": _windows(),
    }
    lopsided["windows"]["fit"] = _stats(cagr_to_strict_mdd=3.1)
    lopsided["windows"]["select_2023"] = _stats(cagr_to_strict_mdd=10.0)
    assert sorted([lopsided, balanced], key=evaluator.winner_sort_key)[0] is balanced


def test_funding_grid_rejects_missing_or_duplicate_events() -> None:
    expected = pd.date_range(
        "2023-01-01", "2023-01-02", freq="8h", inclusive="left", tz="UTC"
    )
    raw = pd.DataFrame(
        {
            "funding_time_utc": expected + pd.to_timedelta([0, 1, -1], unit="ms"),
            "symbol": "BTCUSDT",
            "funding_rate": [0.0001, -0.0001, 0.0],
        }
    )
    parsed = evaluator._validated_funding_frame(
        raw, start="2023-01-01", end="2023-01-02"
    )
    assert parsed["date"].tolist() == list(expected.tz_convert(None))
    with pytest.raises(ValueError, match="cover every expected"):
        evaluator._validated_funding_frame(
            raw.iloc[:-1], start="2023-01-01", end="2023-01-02"
        )
    duplicate = raw.copy()
    duplicate.loc[1, "funding_time_utc"] = duplicate.loc[0, "funding_time_utc"]
    with pytest.raises(ValueError, match="duplicate"):
        evaluator._validated_funding_frame(
            duplicate, start="2023-01-01", end="2023-01-02"
        )


def test_delay_clock_has_no_wraparound_and_preserves_side() -> None:
    active = np.array([True, False, True, False])
    side = np.array([1, 0, -1, 0], dtype=np.int8)
    delayed_active, delayed_side = evaluator.delay_clock(active, side, 2)
    assert delayed_active.tolist() == [False, False, True, False]
    assert delayed_side.tolist() == [0, 0, 1, 0]


def test_delayed_window_clock_rejects_signal_from_previous_split() -> None:
    dates = pd.Series(pd.date_range("2022-12-31 12:00", periods=300, freq="5min"))
    active = np.zeros(len(dates), dtype=bool)
    side = np.zeros(len(dates), dtype=np.int8)
    active[0] = True
    side[0] = 1
    delayed_active, delayed_side = evaluator.delayed_window_clock(
        active,
        side,
        dates,
        bars=288,
        start="2023-01-01",
        end="2024-01-01",
    )
    assert not delayed_active.any()
    assert not delayed_side.any()


def test_delayed_window_clock_keeps_original_and_delay_inside_split() -> None:
    dates = pd.Series(pd.date_range("2023-01-01", periods=400, freq="5min"))
    active = np.zeros(len(dates), dtype=bool)
    side = np.zeros(len(dates), dtype=np.int8)
    active[12] = True
    side[12] = -1
    delayed_active, delayed_side = evaluator.delayed_window_clock(
        active,
        side,
        dates,
        bars=12,
        start="2023-01-01",
        end="2024-01-01",
    )
    assert np.flatnonzero(delayed_active).tolist() == [24]
    assert delayed_side[24] == -1


def test_strict_mdd_charges_both_cost_legs_at_adverse_mark() -> None:
    trade = Trade(
        signal_position=0,
        entry_position=1,
        exit_position=2,
        side=1,
        gross_return=0.0,
        price_factor=1.0,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=1.0,
        adverse_price_factor=1.0,
        entry_date="2023-01-01 00:05:00",
    )
    cfg = evaluator.EvaluationConfig(leverage=0.5)
    stats = evaluator.strict_equity_stats(
        [trade],
        start="2023-01-01",
        end="2024-01-01",
        cfg=cfg,
        cost_rate=0.01,
    )
    expected_mdd = (1.0 - (1.0 - 0.5 * 0.01) ** 2) * 100.0
    assert np.isclose(stats["strict_mdd_pct"], expected_mdd)
    assert np.isclose(stats["absolute_return_pct"], -expected_mdd)
    assert np.isclose(stats["wall_clock_years"], 365.0 / 365.25)


def test_strict_mdd_places_funding_credit_before_favorable_hwm() -> None:
    trade = Trade(
        signal_position=0,
        entry_position=1,
        exit_position=2,
        side=-1,
        gross_return=0.0,
        price_factor=1.0,
        funding_factor=1.02,
        funding_debit_factor=1.0,
        favorable_price_factor=1.0,
        adverse_price_factor=1.0,
        entry_date="2023-01-01 00:05:00",
    )
    stats = evaluator.strict_equity_stats(
        [trade],
        start="2023-01-01",
        end="2024-01-01",
        cfg=evaluator.EvaluationConfig(leverage=0.5),
        cost_rate=0.0,
    )
    assert np.isclose(stats["strict_mdd_pct"], (1.0 - 1.0 / 1.02) * 100.0)


def test_build_trades_enforces_completed_feature_entry_and_elapsed_exit() -> None:
    dates = pd.date_range("2023-01-01", periods=40, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(30_000.0, 30_100.0, len(dates)),
            "high": np.linspace(30_010.0, 30_110.0, len(dates)),
            "low": np.linspace(29_990.0, 30_090.0, len(dates)),
            "close": np.linspace(30_001.0, 30_101.0, len(dates)),
        }
    )
    funding = pd.DataFrame(
        {"date": pd.Series(dtype="datetime64[ns]"), "funding_rate": pd.Series(dtype=float)}
    )
    cfg = evaluator.EvaluationConfig()
    engine = ExecutionEngine(market, funding, cfg)
    candidate = Candidate(576, 2)
    schedule = pd.DataFrame(
        {
            "signal_bar_open": [str(dates[0])],
            "feature_available": [str(dates[1])],
            "entry_time": [str(dates[1])],
            "exit_time": [str(dates[25])],
            "side": [1],
        }
    )
    trades = evaluator._build_trades(
        engine,
        {timestamp: index for index, timestamp in enumerate(dates)},
        schedule,
        candidate,
    )
    assert trades[0].entry_position == 1
    assert trades[0].exit_position == 25
    schedule.loc[0, "feature_available"] = str(dates[0])
    with pytest.raises(ValueError, match="feature availability"):
        evaluator._build_trades(
            engine,
            {timestamp: index for index, timestamp in enumerate(dates)},
            schedule,
            candidate,
        )


def test_json_artifacts_are_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    evaluator._write_json_exclusive(output, {"value": 1})
    with pytest.raises(FileExistsError):
        evaluator._write_json_exclusive(output, {"value": 2})
