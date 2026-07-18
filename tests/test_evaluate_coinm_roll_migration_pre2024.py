from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_coinm_roll_migration_pre2024 as evaluator
from training.preregister_coinm_roll_migration_alpha import Candidate


def _trade(**overrides) -> evaluator.InverseTrade:
    values = {
        "signal_position": 0,
        "entry_position": 1,
        "exit_position": 2,
        "side": 1,
        "traded_leg": "front",
        "symbol": "BTCUSD_230331",
        "entry_price": 100.0,
        "exit_price": 100.0,
        "favorable_price": 100.0,
        "adverse_price": 100.0,
        "entry_date": "2023-01-01 00:05:00",
    }
    values.update(overrides)
    return evaluator.InverseTrade(**values)


def _stats(**overrides):
    output = {
        "absolute_return_pct": 5.0,
        "cagr_to_strict_mdd": 4.0,
        "strict_mdd_pct": 10.0,
        "trades": 500,
        "mean_net_bps": 2.0,
        "weekly_cluster_sign_flip": {"p_value_one_sided": 0.05},
    }
    output.update(overrides)
    return output


def _windows():
    return {name: _stats(trades=80) for name in evaluator.WINDOWS} | {
        "fit": _stats(trades=500),
        "select_2023": _stats(trades=200),
    }


def test_inverse_coin_pnl_converts_exactly_to_fixed_face_usd_return() -> None:
    coin = evaluator.inverse_coin_pnl(5.0, 100.0, 100.0, 110.0, 1)
    assert np.isclose(coin, 5.0 * 100.0 * (1.0 / 100.0 - 1.0 / 110.0))
    assert np.isclose(coin * 110.0, 50.0)
    assert np.isclose(evaluator.inverse_usd_return(100.0, 110.0, 1), 0.10)
    assert np.isclose(evaluator.inverse_usd_return(100.0, 110.0, -1), -0.10)


def test_strict_mdd_charges_both_fixed_face_cost_legs() -> None:
    cfg = evaluator.EvaluationConfig(leverage=0.5)
    stats = evaluator.strict_equity_stats(
        [_trade()],
        start="2023-01-01",
        end="2024-01-01",
        cfg=cfg,
        cost_rate=0.01,
    )
    assert np.isclose(stats["strict_mdd_pct"], 1.0)
    assert np.isclose(stats["absolute_return_pct"], -1.0)
    assert np.isclose(stats["wall_clock_years"], 365.0 / 365.25)


def test_strict_mdd_uses_favorable_before_adverse_held_path() -> None:
    stats = evaluator.strict_equity_stats(
        [_trade(favorable_price=120.0, adverse_price=90.0)],
        start="2023-01-01",
        end="2024-01-01",
        cfg=evaluator.EvaluationConfig(leverage=0.5),
        cost_rate=0.0,
    )
    expected = (1.0 - 0.95 / 1.10) * 100.0
    assert np.isclose(stats["strict_mdd_pct"], expected)


def _outcome_frame(rows: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=rows, freq="5min")
    frame = pd.DataFrame(
        {
            "signal_bar_open_utc": dates,
            "front_symbol": ["BTCUSD_230331"] * rows,
            "next_symbol": ["BTCUSD_230630"] * rows,
        }
    )
    for leg, offset in (("front", 0.0), ("next", 100.0)):
        opens = np.arange(rows, dtype=float) + 1_000.0 + offset
        frame[f"{leg}_open"] = opens
        frame[f"{leg}_high"] = opens + 10.0
        frame[f"{leg}_low"] = opens - 10.0
        frame[f"{leg}_close"] = opens + 1.0
    return frame


def test_build_trades_uses_next_open_and_held_bar_extremes() -> None:
    outcome = _outcome_frame()
    candidate = Candidate("test", "test", "next", 2)
    dates = outcome["signal_bar_open_utc"]
    schedule = pd.DataFrame(
        {
            "signal_bar_open": [str(dates.iloc[0])],
            "feature_available": [str(dates.iloc[1])],
            "entry_time": [str(dates.iloc[1])],
            "exit_time": [str(dates.iloc[3])],
            "side": [1],
            "traded_leg": ["next"],
            "symbol": ["BTCUSD_230630"],
        }
    )
    trades = evaluator._build_trades(
        outcome,
        {timestamp: position for position, timestamp in enumerate(dates)},
        schedule,
        candidate,
    )
    trade = trades[0]
    assert trade.entry_price == outcome.iloc[1]["next_open"]
    assert trade.exit_price == outcome.iloc[3]["next_open"]
    assert trade.favorable_price == outcome.iloc[1:3]["next_high"].max()
    assert trade.adverse_price == outcome.iloc[1:3]["next_low"].min()


def test_build_trades_rejects_contract_transition() -> None:
    outcome = _outcome_frame()
    outcome.loc[2, "next_symbol"] = "BTCUSD_230929"
    candidate = Candidate("test", "test", "next", 2)
    dates = outcome["signal_bar_open_utc"]
    schedule = pd.DataFrame(
        {
            "signal_bar_open": [str(dates.iloc[0])],
            "feature_available": [str(dates.iloc[1])],
            "entry_time": [str(dates.iloc[1])],
            "exit_time": [str(dates.iloc[3])],
            "side": [1],
            "traded_leg": ["next"],
            "symbol": ["BTCUSD_230630"],
        }
    )
    with pytest.raises(ValueError, match="contract-symbol transition"):
        evaluator._build_trades(
            outcome,
            {timestamp: position for position, timestamp in enumerate(dates)},
            schedule,
            candidate,
        )


def _schedule(dates: pd.Series, positions: list[int], sides: list[int], symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_bar_open": str(dates.iloc[position]),
                "feature_available": str(dates.iloc[position + 1]),
                "entry_time": str(dates.iloc[position + 1]),
                "exit_time": str(dates.iloc[position + 3]),
                "side": side,
                "traded_leg": "next",
                "symbol": symbol,
            }
            for position, side in zip(positions, sides)
        ]
    )


def test_delayed_schedule_preserves_frozen_set_and_same_contract() -> None:
    rows = 10
    dates = pd.date_range("2023-01-01", periods=rows, freq="5min")
    source = pd.DataFrame(
        {
            "signal_bar_open_utc": dates,
            "feature_available_time_utc": dates + pd.Timedelta("5min"),
            "trade_earliest_time_utc": dates + pd.Timedelta("5min"),
            "feature_valid": [True] * rows,
            "next_symbol": ["a"] * 5 + ["b"] * 5,
            "front_symbol": ["f"] * rows,
            "front_hours_to_delivery": [30 * 24.0] * rows,
            "next_hours_to_delivery": [120 * 24.0] * rows,
        }
    )
    candidate = Candidate("test", "test", "next", 2)
    delayed = evaluator.delayed_schedule(
        _schedule(pd.Series(dates), [1, 4], [1, -1], "a"),
        source,
        candidate,
        bars=2,
        start="2023-01-01",
        end="2023-01-02",
    )
    assert len(delayed) == 1
    assert pd.Timestamp(delayed.iloc[0]["signal_bar_open"]) == dates[3]
    assert delayed.iloc[0]["side"] == 1


def test_delayed_clock_reapplies_delivery_safety_at_destination() -> None:
    rows = 5
    dates = pd.date_range("2023-01-01", periods=rows, freq="5min")
    source = pd.DataFrame(
        {
            "signal_bar_open_utc": dates,
            "feature_available_time_utc": dates + pd.Timedelta("5min"),
            "trade_earliest_time_utc": dates + pd.Timedelta("5min"),
            "feature_valid": [True] * rows,
            "next_symbol": ["n"] * rows,
            "front_symbol": ["f"] * rows,
            "front_hours_to_delivery": [30 * 24.0, 30 * 24.0, 12.5, 12.4, 12.3],
            "next_hours_to_delivery": [120 * 24.0] * rows,
        }
    )
    candidate = Candidate("test", "test", "next", 12)
    base = pd.DataFrame(
        {
            "signal_bar_open": [str(dates[0])],
            "feature_available": [str(dates[1])],
            "entry_time": [str(dates[1])],
            "exit_time": [str(dates[1] + pd.Timedelta("1h"))],
            "side": [1],
            "traded_leg": ["next"],
            "symbol": ["n"],
        }
    )
    delayed = evaluator.delayed_schedule(
        base,
        source,
        candidate,
        bars=2,
        start="2023-01-01",
        end="2023-01-02",
    )
    assert delayed.empty


def test_delayed_schedule_does_not_admit_event_skipped_by_base_nonoverlap() -> None:
    rows = 30
    dates = pd.Series(pd.date_range("2023-01-01", periods=rows, freq="5min"))
    source = pd.DataFrame(
        {
            "signal_bar_open_utc": dates,
            "feature_available_time_utc": dates + pd.Timedelta("5min"),
            "trade_earliest_time_utc": dates + pd.Timedelta("5min"),
            "feature_valid": [True] * rows,
            "next_symbol": ["n"] * rows,
            "front_symbol": ["f"] * rows,
            "front_hours_to_delivery": [30 * 24.0] * rows,
            "next_hours_to_delivery": [120 * 24.0] * rows,
        }
    )
    candidate = Candidate("test", "test", "next", 2)
    base = _schedule(dates, [0, 10], [1, -1], "n")
    delayed = evaluator.delayed_schedule(
        base,
        source,
        candidate,
        bars=2,
        start="2023-01-01",
        end="2023-01-02",
    )
    assert len(delayed) == len(base)
    assert delayed["side"].tolist() == base["side"].tolist()


def test_stable_artifact_hash_excludes_timestamp_only() -> None:
    first = {"created_at": "a", "value": 1}
    second = {"created_at": "b", "value": 1}
    assert evaluator._stable_artifact_hash(first) == evaluator._stable_artifact_hash(second)
    assert evaluator._stable_artifact_hash(first) != evaluator._stable_artifact_hash(
        {"created_at": "a", "value": 2}
    )


def test_canonical_config_rejects_protocol_mutation() -> None:
    with pytest.raises(ValueError, match="protocol parameters are frozen"):
        evaluator._require_canonical_artifact_paths(
            evaluator.EvaluationConfig(leverage=1.0)
        )


def test_build_trades_rejects_missing_held_path_ohlc() -> None:
    outcome = _outcome_frame()
    outcome.loc[2, "next_high"] = np.nan
    candidate = Candidate("test", "test", "next", 2)
    dates = outcome["signal_bar_open_utc"]
    schedule = pd.DataFrame(
        {
            "signal_bar_open": [str(dates.iloc[0])],
            "feature_available": [str(dates.iloc[1])],
            "entry_time": [str(dates.iloc[1])],
            "exit_time": [str(dates.iloc[3])],
            "side": [1],
            "traded_leg": ["next"],
            "symbol": ["BTCUSD_230630"],
        }
    )
    with pytest.raises(ValueError, match="missing outcome prices"):
        evaluator._build_trades(
            outcome,
            {timestamp: position for position, timestamp in enumerate(dates)},
            schedule,
            candidate,
        )


def test_selection_gates_require_timing_stress_direction_and_halves() -> None:
    windows = _windows()
    stress = {name: _stats() for name in evaluator.FULL_WINDOWS}
    flip = {
        name: _stats(absolute_return_pct=-5.0)
        for name in evaluator.FULL_WINDOWS
    }
    delay_1h = {
        name: _stats(absolute_return_pct=1.0)
        for name in evaluator.FULL_WINDOWS
    }
    delay_24h = {
        name: _stats(absolute_return_pct=0.5)
        for name in evaluator.FULL_WINDOWS
    }
    assert all(
        evaluator.selection_gates(
            windows, stress, flip, delay_1h, delay_24h
        ).values()
    )
    delay_1h["select_2023"] = _stats(absolute_return_pct=6.0)
    assert not evaluator.selection_gates(
        windows, stress, flip, delay_1h, delay_24h
    )["select_beats_1h_delay"]


def test_winner_sort_prefers_balanced_fit_and_selection_ratio() -> None:
    balanced = {"name": "b", "windows": _windows()}
    balanced["windows"]["fit"] = _stats(cagr_to_strict_mdd=3.5)
    balanced["windows"]["select_2023"] = _stats(cagr_to_strict_mdd=3.6)
    lopsided = {"name": "a", "windows": _windows()}
    lopsided["windows"]["fit"] = _stats(cagr_to_strict_mdd=3.1)
    lopsided["windows"]["select_2023"] = _stats(cagr_to_strict_mdd=10.0)
    assert sorted([lopsided, balanced], key=evaluator.winner_sort_key)[0] is balanced


def test_static_support_and_source_bytes_are_sealed_pre2024() -> None:
    support = evaluator._verify_static_dependencies()
    assert support["protocol"]["2024_plus_opened"] is False
    assert evaluator._sha256(evaluator.EvaluationConfig.source_csv) == evaluator.SOURCE_SHA256


def test_json_artifacts_are_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    evaluator._write_json_exclusive(output, {"value": 1})
    with pytest.raises(FileExistsError):
        evaluator._write_json_exclusive(output, {"value": 2})
