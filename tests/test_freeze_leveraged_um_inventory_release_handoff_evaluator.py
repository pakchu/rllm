from __future__ import annotations

import pytest

from training import evaluate_leveraged_um_inventory_release_handoff as evaluator
from training import freeze_leveraged_um_inventory_release_handoff_evaluator as freeze


def test_build_freeze_manifest_is_outcome_blind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    monkeypatch.setattr(
        freeze,
        "_committed_blob_sha256",
        lambda _commit, _path: evaluator._sha256(evaluator.EVALUATION_SOURCE),
    )
    monkeypatch.setattr(evaluator, "verify_preregistration", lambda: ({}, {}, {}))
    manifest = freeze.build_freeze_manifest(commit)
    assert manifest["outcomes_opened_for_luri48"] is False
    assert manifest["returns_prices_or_funding_loaded_during_freeze"] is False
    assert manifest["opened_windows"] == []
    assert manifest["mutable_parameters"] == []
    assert manifest["evaluation_source_commit"] == commit
    assert manifest["sealed_windows"] == [
        *evaluator.WINDOWS,
        "test2024",
        "eval2025",
        "ytd2026",
    ]
    assert manifest["support_result_sha256"] == evaluator.SUPPORT_RESULT_SHA256
    assert manifest["event_clock_sha256"] == evaluator.EVENT_CLOCK_SHA256
    assert manifest["market_manifest_sha256"] == evaluator.MARKET_MANIFEST_SHA256
    assert manifest["market_data_sha256"] == evaluator.MARKET_DATA_SHA256
    assert manifest["funding_manifest_sha256"] == evaluator.FUNDING_MANIFEST_SHA256
    assert manifest["funding_data_sha256"] == evaluator.FUNDING_DATA_SHA256
    assert manifest["evaluator_document_sha256"] == (
        evaluator.EVALUATOR_DOCUMENT_SHA256
    )


def test_build_freeze_manifest_rejects_uncommitted_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        freeze,
        "_committed_blob_sha256",
        lambda _commit, _path: "0" * 64,
    )
    with pytest.raises(ValueError, match="differs from source commit"):
        freeze.build_freeze_manifest("a" * 40)


def test_build_freeze_manifest_requires_full_commit() -> None:
    with pytest.raises(ValueError, match="full length"):
        freeze.build_freeze_manifest("short")
