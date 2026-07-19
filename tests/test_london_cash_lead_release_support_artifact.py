from __future__ import annotations

import hashlib
import json
from pathlib import Path


RESULT = Path("results/london_cash_lead_release_support_2026-07-20.json")
EVENT_CLOCK = Path(
    "results/london_cash_lead_release_event_clock_2026-07-20.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lclr_support_rejection_is_outcome_blind_and_hash_bound() -> None:
    assert _sha256(RESULT) == (
        "9445daa781a6b798242f3166ffe5cbcb03e8d93d0bba35d9740294a93bba3ea1"
    )
    result = json.loads(RESULT.read_text())

    assert result["result_hash"] == (
        "8673c93acc8a3624895069298dc1073046ffbc85b909344b96f14c0d443f466b"
    )
    assert result["outcomes_opened"] is False
    assert result["protocol"]["outcomes_opened"] is False
    assert result["source_audit"] == {
        "coinbase_rows_parsed": 9396,
        "binance_rows_parsed": 9396,
        "coinbase_outside_window_non_date_rows_parsed": 0,
        "binance_outside_window_non_date_rows_parsed": 0,
        "funding_rows_loaded": 0,
        "post_window_execution_or_outcome_rows_loaded": 0,
        "rows_at_or_after_2023_loaded": 0,
    }
    assert result["event_clock_written"] is False
    assert not EVENT_CLOCK.exists()
    assert result["failure_action"] == (
        "reject before outcomes; no parameter, vote, latency, or hold repair"
    )


def test_lclr_fails_only_the_frozen_quarter_support_check() -> None:
    result = json.loads(RESULT.read_text())
    gate = result["support_gate"]

    assert gate["passed"] is False
    assert gate["counts"] == {
        "total_2020_2022": 325,
        "train_2020_2021": 209,
        "train_2020": 81,
        "train_2021": 128,
        "test_2022": 116,
        "test_2022_h1": 60,
        "test_2022_h2": 56,
    }
    assert gate["quarter_counts"] == {
        "2020Q1": 1,
        "2020Q2": 26,
        "2020Q3": 29,
        "2020Q4": 25,
        "2021Q1": 30,
        "2021Q2": 29,
        "2021Q3": 31,
        "2021Q4": 38,
        "2022Q1": 33,
        "2022Q2": 27,
        "2022Q3": 24,
        "2022Q4": 32,
    }
    assert [name for name, passed in gate["checks"].items() if not passed] == [
        "each_quarter"
    ]
    assert result["window_support"] == {
        "windows": 783,
        "complete_windows": 783,
        "incomplete_windows": 0,
        "threshold_ready_windows": 720,
        "candidate_windows": 325,
    }


def test_lclr_frozen_artifact_hashes_still_match() -> None:
    artifacts = json.loads(RESULT.read_text())["protocol"]["frozen_artifacts"]
    for key in (
        "source_decision",
        "preregistration_document",
        "preregistration_source",
    ):
        assert _sha256(Path(artifacts[key])) == artifacts[f"{key}_sha256"]
