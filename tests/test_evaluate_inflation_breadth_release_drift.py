from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_inflation_breadth_release_drift as evaluator


def test_static_inputs_and_schedule_family_are_frozen() -> None:
    registration, support = evaluator._verify_static_inputs()
    assert registration["policy_id"] == "IBRD-7"
    assert support["support_passed"] is True
    schedules = evaluator.load_schedules()
    assert tuple(schedules) == evaluator.ALL_CLOCK_NAMES
    assert len(evaluator._window_schedule(schedules["primary"], evaluator.STAGE1)) == 20
    assert len(evaluator._window_schedule(schedules["primary"], evaluator.STAGE2)) == 7
    for schedule in schedules.values():
        assert schedule["entry_time"].eq(
            schedule["signal_day"] + pd.Timedelta(minutes=5)
        ).all()
        assert schedule["exit_time"].eq(
            schedule["entry_time"] + pd.Timedelta(days=7)
        ).all()


def test_freeze_opens_no_outcome_and_replays(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    report = evaluator.freeze_evaluator(path)
    assert report["opened_windows"] == []
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["simulation_run_during_freeze"] is False
    assert evaluator.verify_evaluator_freeze(path) == report


def test_two_sided_signflip_is_invariant_to_exact_direction_flip() -> None:
    trades = pd.DataFrame(
        {
            "entry_time": [
                "2021-01-04T22:05:00+00:00",
                "2021-01-11T22:05:00+00:00",
                "2021-01-18T22:05:00+00:00",
            ],
            "net_return": [0.02, -0.01, 0.03],
        }
    )
    first = evaluator.weekly_cluster_signflip_two_sided(trades, draws=1000, seed=7)
    flipped = trades.copy()
    flipped["net_return"] = -flipped["net_return"]
    second = evaluator.weekly_cluster_signflip_two_sided(flipped, draws=1000, seed=7)
    assert first["method"] == "exact"
    assert first["p_value_two_sided"] == second["p_value_two_sided"]


def test_exact_signflip_handles_twenty_clusters_without_sampling() -> None:
    trades = pd.DataFrame(
        {
            "entry_time": pd.date_range("2020-01-06", periods=20, freq="7D", tz="UTC"),
            "net_return": [0.01] * 20,
        }
    )
    result = evaluator.weekly_cluster_signflip_two_sided(
        trades, draws=10, seed=11
    )
    assert result["method"] == "exact"
    assert result["draws"] == 2**20
    assert result["p_value_two_sided"] == 2 / 2**20


def test_frozen_evaluator_artifact_replays() -> None:
    stored = json.loads(Path(evaluator.EVALUATOR_FREEZE).read_text())
    assert evaluator.verify_evaluator_freeze() == stored


def test_stage2_refuses_a_failed_or_missing_stage1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", missing)
    with pytest.raises(ValueError, match="has not been run"):
        evaluator._verified_passing_stage1("irrelevant")
