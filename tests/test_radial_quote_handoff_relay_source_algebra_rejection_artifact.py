from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import build_radial_quote_handoff_relay_support as rqhr


ARTIFACT = Path(
    "results/radial_quote_handoff_relay_source_algebra_rejection_2026-07-23.json"
)
ARTIFACT_SHA256 = (
    "92aa49128906e007beae1e7f65120741bc7942ee49371cbb264aee6313a63167"
)
MANIFEST_HASH = (
    "bb20b4fb60f0f9f166eedbf00d4102d8f5f9787d8c844842114967d96d83cc55"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_rejection_artifact_is_immutable_and_self_consistent() -> None:
    report = payload()
    assert sha256(ARTIFACT) == ARTIFACT_SHA256
    assert report["manifest_hash"] == MANIFEST_HASH
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == rqhr.canonical_hash(core)
    assert report["candidate"] == "RQHR-72"
    assert report["failure_stage"] == "frozen_source_algebra"
    assert report["advance_to_evaluator_freeze"] is False


def test_rejection_counts_and_serialization_root_cause_are_frozen() -> None:
    report = payload()
    audit = report["source_audit"]
    assert audit["rows_read"] == 105_120
    assert audit["complete_rows"] == 101_956
    assert audit["incomplete_rows"] == 3_164
    assert audit["algebra_observations"] == 407_824
    assert audit["fixed_absolute_tolerance"] == "5E-12"
    assert audit["failure_counts_by_reason"] == {"efficiency_ratio": 307}
    assert audit["failure_counts_by_radius"] == {
        "2": 41,
        "3": 79,
        "4": 80,
        "5": 107,
    }
    assert audit["total_failed_observations"] == 307
    assert audit["maximum_absolute_efficiency_error"] == (
        "8.6537675544275273E-12"
    )
    assert report["frozen_bindings"]["gzip_csv_writer"]["float_format"] == (
        "%.12g"
    )


def test_rejection_kept_event_comparator_and_outcome_boundaries_closed() -> None:
    report = payload()
    boundary = report["boundary"]
    assert boundary["rqhr_net_path_efficiency_values_opened"] is True
    assert boundary["rqhr_features_derived"] == 0
    assert boundary["rqhr_arms_derived"] == 0
    assert boundary["rqhr_confirmations_derived"] == 0
    assert boundary["rqhr_events_derived"] == 0
    assert boundary["comparator_rows_read"] == 0
    assert boundary["comparator_overlap_opened"] is False
    assert boundary["market_or_outcome_rows_read"] == 0
    assert boundary["price_funding_return_pnl_cagr_mdd_opened"] is False
    assert report["parameter_changes_after_failure"] == []
    assert report["support_or_novelty_evaluated"] is False
