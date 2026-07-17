from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_eia_petroleum_stock_breadth as evaluator


def test_static_inputs_and_schedule_family_are_frozen() -> None:
    registration, support = evaluator._verify_static_inputs()
    assert registration["policy_id"] == "EPSB-1"
    assert support["support_passed"] is True
    schedules = evaluator.load_schedules()
    assert tuple(schedules) == evaluator.ALL_CLOCK_NAMES
    assert len(evaluator._window_schedule(schedules["primary"], evaluator.STAGE1)) == 37
    assert len(evaluator._window_schedule(schedules["primary"], evaluator.STAGE2)) == 13
    for schedule in schedules.values():
        assert schedule["entry_time"].eq(
            schedule["signal_day"] + pd.Timedelta(minutes=5)
        ).all()
        assert schedule["exit_time"].eq(
            schedule["entry_time"] + pd.Timedelta(hours=72)
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
                "2021-01-04T13:05:00+00:00",
                "2021-01-11T13:05:00+00:00",
                "2021-01-18T13:05:00+00:00",
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


def test_stage2_refuses_a_missing_stage1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", missing)
    with pytest.raises(ValueError, match="has not been run"):
        evaluator._verified_passing_stage1("irrelevant")


def test_frozen_stage1_is_rejected_and_keeps_2023_sealed() -> None:
    stored = json.loads(Path(evaluator.STAGE1_OUTPUT).read_text())
    assert stored["manifest_hash"] == (
        "9e0e0e5eabe6eeeddbd46ff88bbf45453f08d76ca6488192d8f309f84c4b9869"
    )
    assert stored["gate_passed"] is False
    assert stored["disposition"] == "REJECT_KEEP_2023_SEALED"
    assert stored["opened_windows"] == ["stage1_2020_2022"]
    assert stored["sealed_windows"] == ["stage2_2023", "2024_plus"]
    primary = stored["headline_by_clock"]["primary"]
    assert primary["absolute_return_pct"] == pytest.approx(29.211626364928133)
    assert primary["cagr_pct"] == pytest.approx(8.916096553527764)
    assert primary["strict_mdd_pct"] == pytest.approx(12.034489573323137)
    assert primary["cagr_to_strict_mdd"] == pytest.approx(0.7408786637110129)
    assert primary["trades"] == 37
    with pytest.raises(ValueError, match="Stage1 failed; 2023 remains sealed"):
        evaluator._verified_passing_stage1(
            stored["evaluator_freeze_manifest_hash"]
        )
