from __future__ import annotations

import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/bitmex_insurance_fund_absorption_support_2026-07-20.json"
)
EVENT_CLOCK = Path(
    "results/bitmex_insurance_fund_absorption_event_clock_2026-07-20.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ifar_support_rejection_is_outcome_blind_and_hash_bound() -> None:
    assert _sha256(RESULT) == (
        "5b32c92bf79abcf031698e87f155c19f8b0398fb4d85df591943a1f156143a02"
    )
    result = json.loads(RESULT.read_text())
    assert result["result_hash"] == (
        "085fffd6addbd0e450bc70605e5c04e844d7b05b4c44cb916bff5b5a21959630"
    )
    assert result["protocol_hash"] == (
        "a68ade9d78a86f57d9873eb5050d75c046340c9752e7419d21fa8720cec919e2"
    )
    assert result["outcomes_opened"] is False
    assert result["protocol"]["outcomes_opened"] is False
    assert result["source_audit"] == {
        "insurance_rows_parsed": 1826,
        "market_snapshot_rows_parsed": 1096,
        "market_outside_snapshot_non_date_rows_parsed": 0,
        "funding_rows_loaded": 0,
        "post_decision_execution_or_outcome_rows_loaded": 0,
        "rows_at_or_after_2023_loaded": 0,
    }
    assert result["event_clock_written"] is False
    assert not EVENT_CLOCK.exists()
    assert result["failure_action"] == (
        "reject before outcomes; no threshold, embargo, side, or hold repair"
    )


def test_ifar_frozen_support_counts_and_failures_match() -> None:
    result = json.loads(RESULT.read_text())
    assert result["window_support"] == {
        "insurance_days": 1826,
        "market_days": 1096,
        "eligible_days": 914,
        "threshold_ready_eligible_days": 805,
        "candidate_days": 25,
    }
    gate = result["support_gate"]
    assert gate["passed"] is False
    assert gate["counts"] == {
        "total_2020h2_2022": 25,
        "train_2020h2_2021": 17,
        "train_2020h2": 0,
        "train_2021": 17,
        "test_2022": 8,
        "test_2022_h1": 2,
        "test_2022_h2": 6,
    }
    assert gate["quarter_counts"] == {
        "2021Q1": 7,
        "2021Q2": 5,
        "2021Q3": 2,
        "2021Q4": 3,
        "2022Q1": 1,
        "2022Q2": 1,
        "2022Q3": 2,
        "2022Q4": 4,
    }
    assert gate["maximum_quarter_share"] == 0.28
    assert [name for name, passed in gate["checks"].items() if passed] == [
        "side_all",
        "side_train",
        "side_test",
    ]


def test_ifar_frozen_artifact_hashes_still_match() -> None:
    artifacts = json.loads(RESULT.read_text())["protocol"]["frozen_artifacts"]
    for key in (
        "source_decision",
        "source_downloader",
        "preregistration_document",
        "preregistration_source",
    ):
        assert _sha256(Path(artifacts[key])) == artifacts[f"{key}_sha256"]
