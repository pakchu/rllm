from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from training import (
    freeze_cross_domain_liquidity_transmission_relay_support_evaluator_v2 as freeze,
)


def _temporary_output() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    directory = tempfile.TemporaryDirectory(
        dir=freeze.evaluate.REPOSITORY_ROOT / "results"
    )
    output = Path(directory.name) / "evaluator-freeze-v2.json"
    return directory, output.relative_to(freeze.evaluate.REPOSITORY_ROOT)


def _hide_completed_support_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        freeze.evaluate,
        "DEFAULT_OUTPUT_REPORT",
        Path("results/.cdltr-v2-freeze-test-missing-report.json"),
    )
    monkeypatch.setattr(
        freeze.evaluate,
        "DEFAULT_OUTPUT_CLOCK",
        Path("results/.cdltr-v2-freeze-test-missing-clock.csv.gz"),
    )


def test_v2_freeze_is_non_pristine_but_opens_no_rows_during_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_completed_support_run(monkeypatch)
    monkeypatch.setattr(
        freeze.evaluate.pd,
        "read_csv",
        lambda *_args, **_kwargs: pytest.fail("v2 freeze may not reopen source rows"),
    )
    directory, output = _temporary_output()
    try:
        status, payload = freeze.freeze(output)
        assert status == "created"
        assert payload["evaluator_protocol_version"].endswith("_v2")
        assert payload["evaluator_source_commit"] == freeze.EXPECTED_EVALUATOR_COMMIT
        assert payload["evaluator_source_sha256"] == freeze.EXPECTED_EVALUATOR_SHA256
        assert payload["prior_failed_run_boundary"] == {
            "source_value_rows_loaded": 4_468,
            "comparator_event_rows_loaded": 9_985,
            "rrp_vote_rows_derived_in_memory": 1_498,
            "cboe_vote_rows_derived_in_memory": 1_508,
            "network_vote_rows_derived": 0,
            "candidate_event_rows_derived": 0,
            "support_or_novelty_verdicts_produced": 0,
            "btc_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "return_pnl_or_equity_rows_loaded": 0,
            "economic_simulations_run": 0,
            "output_artifacts_written": 0,
        }
        assert all(value == 0 for value in payload["v2_freeze_boundary"].values())
        assert payload["mutable_parameters"] == []
        assert payload["manifest_hash"] == freeze.evaluate.canonical_hash(
            {key: value for key, value in payload.items() if key != "manifest_hash"}
        )

        status_again, same = freeze.freeze(output)
        assert status_again == "verified_existing"
        assert same == payload
    finally:
        directory.cleanup()


def test_v2_freeze_binds_v1_audit_and_representation_only_correction() -> None:
    commit, source_sha = freeze._clean_committed_evaluator()
    payload = freeze.build_manifest(commit, source_sha)
    assert payload["predecessor_freeze"] == {
        "path": str(freeze.PREDECESSOR_FREEZE),
        "sha256": freeze.PREDECESSOR_FREEZE_SHA256,
    }
    assert payload["v1_failure_and_v2_correction"] == {
        "path": str(freeze.CORRECTION_DOCUMENT),
        "sha256": freeze.CORRECTION_DOCUMENT_SHA256,
        "correction_scope": "date representation normalization only",
    }
    assert payload["frozen_contract"]["accepted_date_representations"] == [
        "YYYY-MM-DD",
        "YYYY-MM-DD 00:00:00",
    ]


def test_tampered_v2_freeze_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_completed_support_run(monkeypatch)
    directory, output = _temporary_output()
    try:
        freeze.freeze(output)
        target = freeze.evaluate._repository_path(output)
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["v2_freeze_boundary"]["source_value_rows_read_during_freeze"] = 1
        target.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match="manifest hash mismatch"):
            freeze.freeze(output)
    finally:
        directory.cleanup()
