from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from training.evaluate_trollbox_semantic_disagreement_resolution_novelty import (
    ClockRow,
    build_outputs,
    canonical_hash,
    exact_entry_jaccard,
    maximum_tolerant_matches,
    publish_outputs,
)


BASE = datetime(2021, 1, 1, tzinfo=timezone.utc)


def _clock(
    hours: int,
    *,
    candidate_id: str = "candidate",
    side: int = 1,
    hold_hours: int = 1,
) -> ClockRow:
    entry = BASE + timedelta(hours=hours)
    return ClockRow(
        candidate_id=candidate_id,
        split="train",
        causal_origin=entry - timedelta(minutes=10),
        decision_time=entry - timedelta(minutes=5),
        entry_time=entry,
        exit_time=entry + timedelta(hours=hold_hours),
        side=side,
    )


def test_maximum_tolerant_matching_is_one_to_one() -> None:
    primary = [_clock(0), _clock(10), _clock(20)]
    comparator = [_clock(5, candidate_id="other"), _clock(15, candidate_id="other")]

    matches = maximum_tolerant_matches(
        primary,
        comparator,
        tolerance=timedelta(hours=6),
    )

    assert matches == 2


def test_exact_entry_jaccard_uses_unique_entry_sets() -> None:
    primary = [_clock(0), _clock(10)]
    comparator = [_clock(10, candidate_id="other"), _clock(20, candidate_id="other")]

    jaccard, matches = exact_entry_jaccard(primary, comparator)

    assert matches == 1
    assert jaccard == pytest.approx(1 / 3)


def test_real_novelty_gate_retires_only_for_tbasr_near_overlap() -> None:
    report, clock_bytes = build_outputs()

    assert report["novelty_gate"]["passed"] is False
    failed = [
        name for name, passed in report["novelty_gate"]["checks"].items() if not passed
    ]
    assert failed == ["tbasr_tolerant_coverage_at_most_0_35"]
    tbasr = report["novelty_metrics"]["tbasr:primary"]
    assert tbasr["tsdr_events"] == 163
    assert tbasr["comparator_events"] == 358
    assert tbasr["exact_entry_matches"] == 26
    assert tbasr["exact_entry_jaccard"] == pytest.approx(
        0.052525252525252523
    )
    assert tbasr["maximum_one_to_one_tolerant_matches"] == 66
    assert tbasr["tsdr_tolerant_match_coverage"] == pytest.approx(
        0.4049079754601227
    )
    assert tbasr["signed_occupied_exposure_correlation"] == pytest.approx(
        -0.07648823196555221
    )
    assert report["failure_action"] == "retire_before_economic_evaluation"
    assert report["pure_clock"]["rows_by_candidate"] == {
        "tbasr:primary": 358,
        "tsdr:primary": 238,
    }
    assert report["pure_clock"]["sha256"] == hashlib.sha256(clock_bytes).hexdigest()
    assert report["outcome_boundary"] == {
        "market_rows_loaded_for_frozen_tbasr_causal_feature": 210528,
        "funding_rows_loaded": 0,
        "performance_artifacts_parsed": 0,
        "return_or_pnl_fields_read": 0,
        "strict_simulation_calls": 0,
        "tbasr_test_or_later_market_rows_loaded": 0,
        "post_2022_semantic_rows_loaded": 0,
        "raw_private_text_opened": False,
        "network_calls": 0,
        "economic_outcomes_computed": False,
    }
    core = {k: v for k, v in report.items() if k not in {"created_at", "result_hash"}}
    assert report["result_hash"] == canonical_hash(core)


def test_clock_compression_is_deterministic() -> None:
    first_report, first_clock = build_outputs()
    second_report, second_clock = build_outputs()

    assert first_clock == second_clock
    assert first_report["pure_clock"]["sha256"] == second_report["pure_clock"][
        "sha256"
    ]
    assert first_report["result_hash"] == second_report["result_hash"]


def test_publication_is_create_only_and_rolls_back(tmp_path: Path) -> None:
    report, clock_bytes = build_outputs()
    clock_path = tmp_path / "clock.csv.gz"
    report_path = tmp_path / "report.json"
    mutable_report = copy.deepcopy(report)
    mutable_report["pure_clock"]["path"] = str(clock_path)

    publish_outputs(report_path, clock_path, mutable_report, clock_bytes)
    assert clock_path.read_bytes() == clock_bytes
    assert json.loads(report_path.read_text(encoding="utf-8"))["policy_id"] == (
        "TSDR-72"
    )

    with pytest.raises(FileExistsError):
        publish_outputs(report_path, clock_path, mutable_report, clock_bytes)
    assert clock_path.exists()
    assert report_path.exists()
