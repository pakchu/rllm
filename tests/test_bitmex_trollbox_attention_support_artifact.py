from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


RESULT = Path("results/bitmex_trollbox_attention_support_2026-07-20.json")
ATTENTION_CLOCK = Path(
    "results/bitmex_trollbox_attention_clock_2026-07-20.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tbasr_attention_support_pass_is_text_and_outcome_blind() -> None:
    assert _sha256(RESULT) == (
        "c42713c6ef60588c18cf97cf2d84bd19ec89c8c0657a8f7eba02f345cd4046d9"
    )
    result = json.loads(RESULT.read_text())
    assert result["result_hash"] == (
        "fe36490c82fba474ae0a8cf0cd5a59e5dfd2af6d192263eca8ba393618773387"
    )
    assert result["protocol_hash"] == (
        "9740a6f1acc96fd6aa3f6474866053e2a26880c2c8f4770498827d7ea73a6f91"
    )
    assert result["outcomes_opened"] is False
    assert result["message_semantics_opened"] is False
    assert result["source_audit"] == {
        "aggregate_rows_parsed": 294807,
        "character_count_loaded": False,
        "message_text_rows_loaded": 0,
        "market_rows_loaded": 0,
        "rows_at_or_after_2023_loaded": 0,
    }
    assert result["attention_clock_written"] is True
    assert result["failure_action"] is None


def test_tbasr_attention_support_passes_every_frozen_calendar_gate() -> None:
    result = json.loads(RESULT.read_text())
    assert result["window_support"] == {
        "eligible_bars": 263232,
        "threshold_ready_eligible_bars": 263232,
        "raw_candidate_bars": 16143,
        "cooldown_selected_bars": 5417,
    }
    gate = result["support_gate"]
    assert gate["counts"] == {
        "total_2020h2_2022": 5417,
        "train_2020h2_2021": 3132,
        "train_2020h2": 849,
        "train_2021": 2283,
        "test_2022": 2285,
        "test_2022_h1": 1087,
        "test_2022_h2": 1198,
    }
    assert gate["quarter_counts"] == {
        "2020Q3": 428,
        "2020Q4": 421,
        "2021Q1": 637,
        "2021Q2": 341,
        "2021Q3": 600,
        "2021Q4": 705,
        "2022Q1": 473,
        "2022Q2": 614,
        "2022Q3": 584,
        "2022Q4": 614,
    }
    assert gate["active_weeks"] == {"all": 131, "train": 79, "test": 53}
    assert gate["maximum_quarter_share"] == 0.13014583717925052
    assert all(gate["checks"].values())
    assert gate["passed"] is True


def test_tbasr_attention_clock_is_hash_bound_causal_and_semantics_free() -> None:
    assert _sha256(ATTENTION_CLOCK) == (
        "5b60016a3d612f8cd29ea4548241daea76b6a6b60759837ab7bfcd60b8727f73"
    )
    clock = json.loads(ATTENTION_CLOCK.read_text())
    assert clock["manifest_hash"] == (
        "8d1eebc60906942f5900454f956c41f8e1ccb2f00d8e97ad426669e983abdb7e"
    )
    assert clock["attention_clock_hash"] == (
        "d9eafbc0ff55893abf05166b902c4153e09720c06496ed4886dec7f3efbd44f6"
    )
    assert clock["outcomes_opened"] is False
    assert clock["message_semantics_opened"] is False
    events = clock["events"]
    assert len(events) == 5417
    expected_keys = {
        "observation_start",
        "observation_end",
        "entry_earliest",
        "exit_time",
    }
    assert all(set(event) == expected_keys for event in events)
    assert events[0] == {
        "observation_start": "2020-07-01 13:35:00",
        "observation_end": "2020-07-01 13:40:00",
        "entry_earliest": "2020-07-01 13:45:00",
        "exit_time": "2020-07-01 15:45:00",
    }
    assert events[-1] == {
        "observation_start": "2022-12-31 07:35:00",
        "observation_end": "2022-12-31 07:40:00",
        "entry_earliest": "2022-12-31 07:45:00",
        "exit_time": "2022-12-31 09:45:00",
    }
    starts = pd.to_datetime([event["observation_start"] for event in events])
    assert starts.is_monotonic_increasing
    assert starts.to_series().diff().dropna().ge(pd.Timedelta(hours=1)).all()


def test_tbasr_attention_frozen_artifact_hashes_still_match() -> None:
    artifacts = json.loads(RESULT.read_text())["protocol"]["frozen_artifacts"]
    for key in (
        "source_decision",
        "source_downloader",
        "preregistration_document",
        "preregistration_source",
    ):
        assert _sha256(Path(artifacts[key])) == artifacts[f"{key}_sha256"]
