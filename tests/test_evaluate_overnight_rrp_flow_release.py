from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_overnight_rrp_flow_release as evaluator


def test_freeze_opens_no_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    report = evaluator.freeze_evaluator(path)
    assert report["opened_windows"] == []
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["simulation_run_during_freeze"] is False
    assert evaluator.verify_evaluator_freeze(path) == report


def test_schedules_are_causal_nonoverlapping_and_variable_horizon() -> None:
    schedules = evaluator.load_schedules()
    horizons: set[pd.Timedelta] = set()
    for schedule in schedules.values():
        assert schedule["entry_time"].eq(
            schedule["signal_day"] + pd.Timedelta(minutes=5)
        ).all()
        assert schedule["exit_time"].gt(schedule["entry_time"]).all()
        assert schedule["entry_time"].iloc[1:].reset_index(drop=True).ge(
            schedule["exit_time"].iloc[:-1].reset_index(drop=True)
        ).all()
        horizons.update(schedule["exit_time"] - schedule["entry_time"])
    assert pd.Timedelta(days=1) in horizons
    assert any(value > pd.Timedelta(days=1) for value in horizons)


def test_primary_stage_counts_are_frozen() -> None:
    primary = evaluator.load_schedules()["primary"]
    stage1 = evaluator._entry_distribution(primary, evaluator.STAGE1)
    stage2 = evaluator._entry_distribution(primary, evaluator.STAGE2)
    assert stage1["trades"] == 111
    assert (stage1["longs"], stage1["shorts"]) == (63, 48)
    assert stage2["trades"] == 74
    assert (stage2["longs"], stage2["shorts"]) == (50, 24)


def test_failed_stage1_blocks_2023_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = evaluator.freeze_evaluator(tmp_path / "freeze.json")
    fake = {
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": evaluator._sha256(evaluator.EVALUATOR_SOURCE),
        "gate_passed": False,
    }
    core = dict(fake)
    fake["manifest_hash"] = evaluator._canonical_hash(core)
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(json.dumps(fake))
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", stage1)
    called = False

    def forbidden_loader(window: evaluator.TimeWindow):
        nonlocal called
        called = True
        raise AssertionError(window)

    monkeypatch.setattr(evaluator, "load_execution_window", forbidden_loader)
    with pytest.raises(ValueError, match="2023 remains sealed"):
        evaluator._verified_passing_stage1(freeze["manifest_hash"])
    assert called is False


def test_two_sided_signflip_detects_symmetric_null() -> None:
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2021-01-04", "2021-01-11", "2021-01-18", "2021-01-25"],
                utc=True,
            ),
            "net_return": [0.01, -0.01, 0.01, -0.01],
        }
    )
    result = evaluator.weekly_cluster_signflip_two_sided(
        trades, draws=20_000, seed=20_260_717
    )
    assert result["method"] == "exact"
    assert result["p_value_two_sided"] == 1.0
