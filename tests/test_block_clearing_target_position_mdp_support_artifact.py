from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training import build_block_clearing_target_position_mdp_support as s
from training import preregister_block_clearing_target_position_mdp as prereg


REPORT = Path(
    "results/block_clearing_target_position_mdp_support_2026-07-25.json"
)
SEQUENCES = Path(
    "data/block_clearing_target_position_mdp_sequences_2020_2023.csv.gz"
)
REPORT_SHA256 = (
    "00166aac18bd59e2b8c56ac285072fe5151c77159eca2e8c8e446ae0ed134ef8"
)
SEQUENCE_SHA256 = (
    "00fd5a0fb5c238ca27109e49d6b3c7f11d16d6edbd37788f76a1bcaeeb86dd56"
)
MANIFEST_HASH = (
    "b13d5429ec11191232b8b4af40c3d85e003a17f58e6a6f6fa4eed31b27b9b85c"
)
FRAME_HASH = (
    "d0a5fce4ab8dc1e26e1309a53c36cb3e875123dfcd7c783589a6e59b919c15b0"
)


def _report() -> dict[str, object]:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_support_artifacts_are_hash_bound_and_canonical() -> None:
    report = _report()
    assert s.sha256_file(REPORT) == REPORT_SHA256
    assert s.sha256_file(SEQUENCES) == SEQUENCE_SHA256
    core = {
        key: value
        for key, value in report.items()
        if key != "manifest_hash"
    }
    assert report["manifest_hash"] == MANIFEST_HASH
    assert s.canonical_hash(core) == MANIFEST_HASH
    assert report["sequence_artifact"] == {
        "path": str(s.DEFAULT_SEQUENCE_OUTPUT),
        "sha256": SEQUENCE_SHA256,
        "frame_hash": FRAME_HASH,
        "rows": 2_784,
        "columns": list(prereg.SOURCE_SEQUENCE_COLUMNS),
    }

    sequences = pd.read_csv(
        SEQUENCES,
        usecols=list(prereg.SOURCE_SEQUENCE_COLUMNS),
    )
    assert sequences.columns.tolist() == list(prereg.SOURCE_SEQUENCE_COLUMNS)
    assert len(sequences) == 2_784
    assert s._frame_hash(sequences) == FRAME_HASH


def test_support_passed_without_any_outcome_access() -> None:
    report = _report()
    assert report["artifact_eligible"] is True
    assert report["source_support_passed"] is True
    assert all(report["source_support_checks"].values())
    assert report["first_failing_stage"] == "none"
    assert report["first_failing_check"] is None
    assert report["decision"] == (
        "advance_to_frozen_economic_and_cheap_policy_evaluator"
    )
    assert report["authorized_next_stage"] == (
        "freeze_economic_evaluator_and_cheap_policy_family"
    )
    assert not any(
        "2023" in name or "gap" in name
        for name in report["source_support_checks"]
    )

    boundary = report["outcome_boundary"]
    for field in (
        "market_rows_decoded",
        "funding_rows_decoded",
        "comparator_rows_decoded",
        "future_return_rows_decoded",
        "return_or_PnL_fields_decoded",
        "PnL_CAGR_MDD_values_decoded",
        "post_2023_rows_decoded",
        "actions_or_labels_created",
        "model_training_runs",
        "network_calls",
    ):
        assert boundary[field] == 0


def test_source_sequence_funnel_and_report_only_2023_are_exact() -> None:
    report = _report()
    assert report["feature_funnel"] == {
        "formed_buckets": 2_918,
        "rank_complete_states": 2_792,
        "first_rank_complete_predecessor_only": 1,
        "token_ready_states": 2_791,
    }
    replay = report["bcrt_replay_audit"]
    assert replay["development_gate"] == {
        "formed_buckets": 2_184,
        "rank_complete_states": 2_058,
        "token_ready_states": 2_057,
        "rows_compared": 2_057,
        "common_projection_identical": True,
        "common_projection_sha256": (
            "b985db849d855f9046c93ad817ba2d49415e6aedf341d3296e77e2db8ecbb03b"
        ),
        "prefix_replay_passed": True,
    }
    assert replay["full_source_report_only"][
        "expected_replay_counts_exact"
    ] is True
    assert replay["train_2022_gate"] == {
        "reports_exact": True,
        "checks_exact": True,
        "checks_all_true": True,
    }
    assert replay["legacy_first_failure_unchanged"] == {
        "stage": "source_support",
        "check": "max_entry_gap_days_2020_2022",
    }

    batching = report["batching_audit"]
    assert batching["development_boolean"]["actionable_releases"] == 2_057
    assert batching["development_boolean"]["actual_sequence_rows"] == 2_055
    assert batching["development_boolean"]["warmup_exact"] is True
    assert batching["full_source_report_only"][
        "post_2023_release_states_omitted"
    ] == 5
    assert batching["full_source_report_only"][
        "unknown_2023_vocabulary_present"
    ] is False
    assert batching["full_source_report_only"][
        "strict_full_replay_when_vocabulary_known"
    ] is True

    reports = report["development_sequence_reports"]
    assert reports["2020"]["events"] == 595
    assert reports["2021"]["events"] == 729
    assert reports["2022"]["events"] == 731
    assert reports["development"]["events"] == 2_055
    assert reports["2021"]["active_months"] == 12
    assert reports["2022"]["active_months"] == 12
    assert reports["development"][
        "maximum_exact_source_signature_share"
    ] == 0.00048661800486618007

    evaluation = report["report_only_2023"]
    assert evaluation["events"] == 729
    assert evaluation["active_months"] == 12
    assert evaluation["boolean_gate"] is False
    assert evaluation[
        "may_authorize_continue_retire_repair_or_selection"
    ] is False
    assert evaluation["unknown_vocabulary"] == {}
    assert evaluation["unknown_vocabulary_operational_action"] == "TARGET_FLAT"
