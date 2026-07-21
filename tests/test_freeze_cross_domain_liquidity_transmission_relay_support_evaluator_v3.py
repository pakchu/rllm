from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from training import (
    freeze_cross_domain_liquidity_transmission_relay_support_evaluator_v3 as freeze,
)


def _temporary_output() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    directory = tempfile.TemporaryDirectory(
        dir=freeze.evaluate.REPOSITORY_ROOT / "results"
    )
    output = Path(directory.name) / "evaluator-freeze-v3.json"
    return directory, output.relative_to(freeze.evaluate.REPOSITORY_ROOT)


def _hide_completed_support_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        freeze.evaluate,
        "DEFAULT_OUTPUT_REPORT",
        Path("results/.cdltr-v3-freeze-test-missing-report.json"),
    )
    monkeypatch.setattr(
        freeze.evaluate,
        "DEFAULT_OUTPUT_CLOCK",
        Path("results/.cdltr-v3-freeze-test-missing-clock.csv.gz"),
    )


def test_v3_freeze_preserves_both_failures_but_opens_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_completed_support_run(monkeypatch)
    monkeypatch.setattr(
        freeze.evaluate.pd,
        "read_csv",
        lambda *_args, **_kwargs: pytest.fail("v3 freeze may not reopen source rows"),
    )
    directory, output = _temporary_output()
    try:
        status, payload = freeze.freeze(output)
        assert status == "created"
        assert payload["evaluator_protocol_version"].endswith("_v3")
        assert payload["evaluator_source_commit"] == freeze.EXPECTED_EVALUATOR_COMMIT
        assert payload["evaluator_source_sha256"] == freeze.EXPECTED_EVALUATOR_SHA256
        failures = payload["prior_failed_run_boundaries"]
        assert failures["v1_date_representation_failure"] == (
            freeze.V1_FAILED_RUN_BOUNDARY
        )
        assert failures["v2_duplicate_network_availability_failure"] == (
            freeze.V2_FAILED_RUN_BOUNDARY
        )
        assert all(value == 0 for value in payload["v3_freeze_boundary"].values())
        assert payload["mutable_parameters"] == []
        assert payload["manifest_hash"] == freeze.evaluate.canonical_hash(
            {key: value for key, value in payload.items() if key != "manifest_hash"}
        )

        status_again, same = freeze.freeze(output)
        assert status_again == "verified_existing"
        assert same == payload
    finally:
        directory.cleanup()


def test_v3_freeze_binds_v2_audit_and_simultaneous_batch_rule() -> None:
    commit, source_sha = freeze._clean_committed_evaluator()
    payload = freeze.build_manifest(commit, source_sha)
    assert payload["predecessor_freeze"] == {
        "path": str(freeze.PREDECESSOR_FREEZE),
        "sha256": freeze.PREDECESSOR_FREEZE_SHA256,
    }
    assert payload["v2_failure_and_v3_correction"] == {
        "path": str(freeze.CORRECTION_DOCUMENT),
        "sha256": freeze.CORRECTION_DOCUMENT_SHA256,
        "correction_scope": "simultaneous network publication batching only",
    }
    assert payload["frozen_contract"]["network_publication_batch_rule"] == (
        "raw available_at monotonic non-decreasing; exact ties collapse to latest "
        "observation_date; final timestamps unique increasing"
    )


def test_tampered_v3_freeze_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_completed_support_run(monkeypatch)
    directory, output = _temporary_output()
    try:
        freeze.freeze(output)
        target = freeze.evaluate._repository_path(output)
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["v3_freeze_boundary"]["source_value_rows_read_during_freeze"] = 1
        target.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match="manifest hash mismatch"):
            freeze.freeze(output)
    finally:
        directory.cleanup()
