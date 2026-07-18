from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_cross_venue_vol_disagreement_pre2024 as evaluator
from training.preregister_cross_venue_vol_disagreement_alpha import Candidate
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine


def _stats(**overrides: float | int | dict[str, float]) -> dict[str, object]:
    output: dict[str, object] = {
        "absolute_return_pct": 5.0,
        "cagr_to_strict_mdd": 4.0,
        "strict_mdd_pct": 10.0,
        "trades": 30,
        "mean_gross_bps": 30.0,
        "mean_net_bps": 20.0,
        "weekly_cluster_sign_flip": {"p_value_one_sided": 0.05},
    }
    output.update(overrides)
    return output


def test_selection_gates_require_directional_and_stress_evidence() -> None:
    gates = evaluator.selection_gates(
        _stats(), _stats(trades=12), _stats(trades=12), _stats(), _stats(absolute_return_pct=-4.0)
    )
    assert all(gates.values())

    failed = evaluator.selection_gates(
        _stats(), _stats(trades=12), _stats(trades=12), _stats(absolute_return_pct=-0.1), _stats()
    )
    assert not failed["ten_bp_per_side_stress_positive"]
    assert not failed["direction_flip_absolute_return_negative"]


def test_winner_sort_prefers_worst_quarter_before_full_ratio() -> None:
    stronger_floor = {
        "name": "b",
        "selection": _stats(cagr_to_strict_mdd=3.1, mean_net_bps=10.0),
        "q3": _stats(absolute_return_pct=3.0),
        "q4": _stats(absolute_return_pct=3.0),
    }
    higher_full_ratio = {
        "name": "a",
        "selection": _stats(cagr_to_strict_mdd=8.0, mean_net_bps=30.0),
        "q3": _stats(absolute_return_pct=1.0),
        "q4": _stats(absolute_return_pct=8.0),
    }
    assert sorted([higher_full_ratio, stronger_floor], key=evaluator.winner_sort_key)[0] is stronger_floor


def test_funding_grid_rejects_missing_or_duplicate_events() -> None:
    expected = pd.date_range("2023-01-01", "2023-01-02", freq="8h", inclusive="left", tz="UTC")
    raw = pd.DataFrame(
        {
            "funding_time_utc": expected + pd.to_timedelta([0, 1, -1], unit="ms"),
            "symbol": "BTCUSDT",
            "funding_rate": [0.0001, -0.0001, 0.0],
        }
    )
    parsed = evaluator._validated_funding_frame(raw, start="2023-01-01", end="2023-01-02")
    assert parsed["date"].tolist() == list(expected.tz_convert(None))

    with pytest.raises(ValueError, match="cover every expected"):
        evaluator._validated_funding_frame(raw.iloc[:-1], start="2023-01-01", end="2023-01-02")
    duplicate = raw.copy()
    duplicate.loc[1, "funding_time_utc"] = duplicate.loc[0, "funding_time_utc"]
    with pytest.raises(ValueError, match="duplicate"):
        evaluator._validated_funding_frame(duplicate, start="2023-01-01", end="2023-01-02")


def test_build_trades_enforces_next_five_minute_entry_and_elapsed_exit() -> None:
    dates = pd.date_range("2023-07-01", periods=40, freq="5min")
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
    candidate = Candidate("bvol_rich_move_fade", 0.8, 0.8, 2)
    schedule = pd.DataFrame(
        {
            "signal_time": [str(dates[0])],
            "entry_time": [str(dates[1])],
            "exit_time": [str(dates[25])],
            "side": [1],
        }
    )
    trades = evaluator._build_trades(
        engine, {timestamp: i for i, timestamp in enumerate(dates)}, schedule, candidate
    )
    assert trades[0].entry_position == 1
    assert trades[0].exit_position == 25

    schedule.loc[0, "entry_time"] = str(dates[0])
    with pytest.raises(ValueError, match="entry differs"):
        evaluator._build_trades(
            engine, {timestamp: i for i, timestamp in enumerate(dates)}, schedule, candidate
        )


def test_json_artifacts_are_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    evaluator._write_json_exclusive(output, {"value": 1})
    with pytest.raises(FileExistsError):
        evaluator._write_json_exclusive(output, {"value": 2})
