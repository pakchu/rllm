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


def test_frozen_stage1_is_rejected_and_keeps_2023_sealed() -> None:
    stored = json.loads(Path(evaluator.STAGE1_OUTPUT).read_text())
    assert evaluator._sha256(evaluator.STAGE1_OUTPUT) == (
        "57ba5710033d7d816cbf9dac04d34f7e3ee4441b38024a0c23f49a2a3ace413e"
    )
    assert stored["manifest_hash"] == (
        "bc47514ad06ad3e4a422d078a7436f13077c3634fe3234fc9b4ba04c416a08d6"
    )
    assert stored["gate_passed"] is False
    assert stored["disposition"] == "REJECT_KEEP_2023_SEALED"
    assert stored["opened_windows"] == ["stage1_2020_2022"]
    assert stored["sealed_windows"] == ["stage2_2023", "2024_plus"]
    primary = stored["headline_by_clock"]["primary"]
    assert primary["absolute_return_pct"] == pytest.approx(6.503480502844838)
    assert primary["cagr_pct"] == pytest.approx(2.1219705301286362)
    assert primary["strict_mdd_pct"] == pytest.approx(4.35091163184399)
    assert primary["cagr_to_strict_mdd"] == pytest.approx(0.48770710822947905)
    assert primary["trades"] == 26
    diagnostics = stored["execution_diagnostics"]
    assert diagnostics["market"]["last_timestamp"] == "2022-12-31T23:55:00+00:00"
    assert diagnostics["funding"]["last_timestamp"] == "2022-12-31T16:00:00+00:00"
    assert diagnostics["market"]["stopped_before_parsing_end_boundary"] is True
    assert diagnostics["funding"]["stopped_before_parsing_end_boundary"] is True
    with pytest.raises(ValueError, match="Stage1 failed; 2023 remains sealed"):
        evaluator._verified_passing_stage1(stored["evaluator_freeze_manifest_hash"])
