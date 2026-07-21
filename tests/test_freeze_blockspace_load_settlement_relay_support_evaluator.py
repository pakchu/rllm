from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from training import (
    freeze_blockspace_load_settlement_relay_support_evaluator as freeze,
)


def _temporary_output() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    directory = tempfile.TemporaryDirectory(
        dir=freeze.evaluate.REPOSITORY_ROOT / "results"
    )
    output = Path(directory.name) / "blsr-evaluator-freeze.json"
    return directory, output.relative_to(freeze.evaluate.REPOSITORY_ROOT)


def _hide_support_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        freeze.evaluate,
        "DEFAULT_OUTPUT_REPORT",
        Path("results/.blsr-freeze-test-missing-support.json"),
    )


def test_freeze_opens_no_source_comparator_or_outcome_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_support_output(monkeypatch)
    monkeypatch.setattr(
        freeze.evaluate.pd,
        "read_csv",
        lambda *_args, **_kwargs: pytest.fail(
            "BLSR freeze may not parse source or comparator rows"
        ),
    )
    monkeypatch.setattr(
        freeze.evaluate,
        "load_source_frame",
        lambda *_args, **_kwargs: pytest.fail("BLSR freeze may not load source values"),
    )
    directory, output = _temporary_output()
    try:
        status, payload = freeze.freeze(output)
        assert status == "created"
        assert payload["candidate"] == "BLSR-288"
        assert payload["evaluator_source_commit"] == freeze.EXPECTED_EVALUATOR_COMMIT
        assert payload["evaluator_source_sha256"] == freeze.EXPECTED_EVALUATOR_SHA256
        assert payload["mutable_parameters"] == []
        assert payload["freeze_boundary"] == freeze.FREEZE_BOUNDARY
        for field in (
            "source_value_rows_read",
            "source_feature_rows_derived",
            "candidate_event_rows_derived",
            "comparator_event_rows_read",
            "support_or_novelty_verdicts_produced",
            "btc_market_rows_loaded",
            "funding_rows_loaded",
            "return_pnl_or_equity_rows_loaded",
            "economic_simulations_run",
            "network_calls",
        ):
            assert payload["freeze_boundary"][field] == 0

        status_again, same = freeze.freeze(output)
        assert status_again == "verified_existing"
        assert same == payload
    finally:
        directory.cleanup()


def test_freeze_binds_complete_blsr_contract() -> None:
    commit, source_sha = freeze._clean_committed_evaluator()
    payload = freeze.build_manifest(commit, source_sha)
    contract = payload["frozen_contract"]

    assert contract["allowed_source_columns"] == list(
        freeze.evaluate.BLSR_SOURCE_COLUMNS
    )
    assert contract["forbidden_source_value_columns"] == list(
        freeze.evaluate.FORBIDDEN_SOURCE_VALUE_COLUMNS
    )
    assert contract["controls"] == list(freeze.evaluate.CONTROL_NAMES)
    assert contract["support_limits"] == freeze.evaluate.SUPPORT_LIMITS
    assert contract["novelty_limits"] == freeze.evaluate.NOVELTY_LIMITS
    assert contract["comparator_capabilities"] == (
        freeze.evaluate.EXPECTED_COMPARATOR_CAPABILITIES
    )
    assert contract["relay_deadline_packets"] == 3
    assert (
        "same-onset-packet endpoint state is ineligible"
        in contract["stale_response_contract"]
    )
    dependency = payload["ledger_builder_dependency"]
    assert dependency["source_loader_or_packetizer_reuse"] is False
    assert "BLSR-local allowed-column" in dependency["authorized_role"]


def test_tampered_freeze_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_support_output(monkeypatch)
    directory, output = _temporary_output()
    try:
        freeze.freeze(output)
        target = freeze.evaluate._repository_path(output)
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["freeze_boundary"]["candidate_event_rows_derived"] = 1
        target.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match="manifest hash mismatch"):
            freeze.freeze(output)
    finally:
        directory.cleanup()


def test_freeze_rejects_support_artifact_that_predates_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "support.json"
    existing.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(freeze.evaluate, "DEFAULT_OUTPUT_REPORT", existing)
    directory, output = _temporary_output()
    try:
        with pytest.raises(RuntimeError, match="predates evaluator freeze"):
            freeze.freeze(output)
    finally:
        directory.cleanup()
