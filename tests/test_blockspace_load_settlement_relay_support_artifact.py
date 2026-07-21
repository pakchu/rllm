from __future__ import annotations

import json
from typing import Any

from training import evaluate_blockspace_load_settlement_relay_support as evaluate


EXPECTED_ARTIFACT_SHA256 = (
    "82cec44fe766a406272678721b0ff5ec997dda0bae4092701a30e28b8c27f672"
)
EXPECTED_MANIFEST_HASH = (
    "e5fe85d141c7c1191dbe5e0ea4d0be00b486b0b0312627b7a6db71b982fa5ed2"
)


def _artifact() -> dict[str, Any]:
    return json.loads(
        evaluate._repository_path(evaluate.DEFAULT_OUTPUT_REPORT).read_text(
            encoding="utf-8"
        )
    )


def test_frozen_blsr288_support_artifact_is_canonical_and_rejected() -> None:
    assert evaluate.sha256_file(evaluate.DEFAULT_OUTPUT_REPORT) == (
        EXPECTED_ARTIFACT_SHA256
    )
    artifact = _artifact()
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    assert artifact["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert artifact["manifest_hash"] == evaluate.canonical_hash(core)
    assert artifact["candidate"] == "BLSR-288"
    assert artifact["verdict"] == {
        "passed": False,
        "status": "REJECT",
        "failed_stages": ["support", "novelty"],
        "strict_economic_train_authorized": False,
        "repair_allowed_under_candidate_identity": False,
    }


def test_frozen_blsr288_support_failure_counts_are_exact() -> None:
    artifact = _artifact()
    support = artifact["support"]
    assert support["passed"] is False
    assert support["counts"] == {
        "total": 54,
        "train": 34,
        "selection": 20,
        "2021": 15,
        "2022": 19,
        "2021H1": 7,
        "2021H2": 8,
        "2022H1": 13,
        "2022H2": 6,
        "2023H1": 9,
        "2023H2": 11,
        "2023Q1": 8,
        "2023Q2": 1,
        "2023Q3": 7,
        "2023Q4": 4,
    }
    assert support["side_counts"] == {
        "train": {"long": 14, "short": 20},
        "selection": {"long": 11, "short": 9},
    }
    assert support["concentration"]["train"]["maximum_weekday_share"] == (
        0.29411764705882354
    )
    assert support["concentration"]["selection"]["maximum_month_share"] == 0.3
    assert artifact["sealed_clock_commitments"]["primary"]["rows"] == 54
    assert artifact["sealed_clock_commitments"]["primary"]["frame_hash"] == (
        "643fe4d3acc03f5f29ec58ffd8611bc693c3ff906fd3adf1f8f2f9d5e05634aa"
    )


def test_frozen_blsr288_stopped_before_comparators_and_outcomes() -> None:
    artifact = _artifact()
    assert artifact["novelty"] == {
        "evaluated": False,
        "passed": False,
        "skip_reason": "primary support or control structure failed",
        "limits": evaluate.NOVELTY_LIMITS,
        "comparators": {},
    }
    boundary = artifact["outcome_boundary"]
    assert boundary["source_value_rows_read"] == 213_095
    assert boundary["source_feature_rows_derived"] == 2_959
    assert boundary["primary_event_incidence_rows_derived"] == 54
    assert boundary["comparator_event_rows_read"] == 0
    assert boundary["btc_market_rows_loaded"] == 0
    assert boundary["funding_rows_loaded"] == 0
    assert boundary["return_rows_loaded"] == 0
    assert boundary["return_or_pnl_fields_read"] == 0
    assert artifact["event_rows_published"] == 0
    assert artifact["feature_values_published"] == 0
