from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_cleveland_fed_cpi_surprise as evaluator


def test_static_inputs_and_schedule_family_are_frozen() -> None:
    registration, support = evaluator._verify_static_inputs()
    assert registration["policy_id"] == "CFCS-1"
    assert support["support_passed"] is True
    schedules = evaluator.load_schedules()
    assert tuple(schedules) == evaluator.ALL_CLOCK_NAMES
    assert len(evaluator._window_schedule(schedules["primary"], evaluator.STAGE1)) == 26
    assert len(evaluator._window_schedule(schedules["primary"], evaluator.STAGE2)) == 8
    for schedule in schedules.values():
        record = evaluator._schedule_record(schedule)
        assert record["execution_clock_exact"] is True
        assert record["globally_nonoverlapping"] is True


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


def test_stage2_refuses_a_missing_stage1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", missing)
    with pytest.raises(ValueError, match="has not been run"):
        evaluator._verified_passing_stage1("irrelevant")


def test_frozen_evaluator_artifact_replays() -> None:
    stored = json.loads(Path(evaluator.EVALUATOR_FREEZE).read_text())
    assert evaluator.verify_evaluator_freeze() == stored
    assert evaluator._sha256(evaluator.EVALUATOR_SOURCE) == (
        "92aba5e648ee4a0ac7119d37a271edd86df99f4177b8a533d4338d0e88bb5ff2"
    )
    assert evaluator._sha256(evaluator.EVALUATOR_FREEZE) == (
        "4e53c9b5d890ee9f8f15b0f993340401b26c3b29b0f35ccdd74c701505b2b381"
    )
    assert stored["manifest_hash"] == (
        "76f91543c284535dcf46f01c38e2bbb47f7192a57422c2d247198822f442feae"
    )
