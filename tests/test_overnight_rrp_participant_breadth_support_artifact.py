from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training import evaluate_overnight_rrp_participant_breadth_support as orpb


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "results/overnight_rrp_participant_breadth_source_support_2026-07-21.json"
)
CLOCK = (
    ROOT / "results/overnight_rrp_participant_breadth_support_clocks_2026-07-21.csv.gz"
)
ARTIFACT_SHA256 = "cb341310436e5de2cc578dd8232f99f0e78efd50a6c8110a9e5c549dc60c5d0b"
CLOCK_SHA256 = "ef21323229801f11557e0c2d9d4465f7d58b13569552d656d64fdb7d440622ed"
MANIFEST_HASH = "62cc70768d70964ce46f65a9d6025b589bf7b7937e18d88fa79ef0271ea8804d"


def _report() -> dict[str, Any]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_committed_support_artifact_is_hash_bound_and_reproducible(
    tmp_path: Path,
) -> None:
    called = False

    def forbidden_loader() -> orpb.ComparatorBundle:
        nonlocal called
        called = True
        raise AssertionError("support failure must keep comparator clocks sealed")

    assert orpb.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    assert orpb.sha256_file(CLOCK) == CLOCK_SHA256
    reproduced = orpb.build_report(
        clock_output=tmp_path / "clocks.csv.gz",
        comparator_loader=forbidden_loader,
    )
    assert called is False
    assert reproduced == _report()
    assert orpb.sha256_file(tmp_path / "clocks.csv.gz") == CLOCK_SHA256
    assert reproduced["manifest_hash"] == MANIFEST_HASH


def test_train_density_failure_retires_orpb_before_novelty() -> None:
    report = _report()
    support = report["support"]

    assert support["passed"] is False
    assert support["failed_checks"] == ["train_events_max"]
    assert support["summaries"]["primary"]["train"]["events"] == 137
    assert support["summaries"]["primary"]["selection"]["events"] == 73
    assert support["integrity"]["prior_only_ols_replay_exact"] is True
    assert support["integrity"]["prior_only_rank_replay_exact"] is True
    assert support["integrity"]["primary_clock_score_rank_replay_exact"] is True
    assert report["novelty"]["evaluated"] is False
    assert report["decision"]["status"] == "retired_before_novelty"
    assert report["decision"]["outcome_evaluator_authorized"] is False
    assert report["decision"]["repair_authorized"] is False


def test_artifact_preserves_the_outcome_boundary() -> None:
    boundary = _report()["outcome_boundary"]

    assert boundary["source_rows_read_for_support"] == 1498
    assert boundary["candidate_clock_rows_created"] == 1588
    assert boundary["comparator_rows_read_for_novelty"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["return_or_pnl_fields_read"] == 0
    assert boundary["post_2023_source_rows_read"] == 0
    assert boundary["economic_outcomes_opened"] is False
