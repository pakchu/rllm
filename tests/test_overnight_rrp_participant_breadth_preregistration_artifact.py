from __future__ import annotations

import json
from pathlib import Path

from training import preregister_overnight_rrp_participant_breadth as orpb


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "results/overnight_rrp_participant_breadth_preregistration_2026-07-21.json"
)
ARTIFACT_SHA256 = "62855414b6926ff3e0f2bc37fe3c4c5c6f46f78803c66d6da564ec65de937b30"
MANIFEST_HASH = "cdc0a7297df71417fe6a00198c296ddf8899b4e9d581e5a2b5f8c55b3b8ba1dd"


def test_committed_preregistration_is_hash_bound_and_reproducible() -> None:
    assert orpb.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    committed = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert orpb.build_registration() == committed
    assert committed["manifest_hash"] == MANIFEST_HASH


def test_artifact_authorizes_only_support_and_novelty() -> None:
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert report["authorization"] == {
        "candidate_count": 1,
        "current_action": "source support and novelty only",
        "economic_evaluator_authorized": False,
    }
    assert (
        report["evidence_boundary"][
            "prospective_comparator_headers_inspected_before_review"
        ]
        == 8
    )
    assert report["evidence_boundary"]["bound_comparator_artifacts"] == 7
    assert (
        report["evidence_boundary"][
            "comparator_identifier_rows_projected_for_cohort_freeze"
        ]
        == 7104
    )
    assert (
        report["evidence_boundary"]["comparator_entry_exit_or_side_fields_materialized"]
        == 0
    )
    assert report["evidence_boundary"]["economic_outcomes_opened"] is False
    assert report["later_outcome_contract"]["authorized"] is False
