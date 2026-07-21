from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from training import (
    freeze_cross_domain_liquidity_transmission_relay_support_evaluator as freeze,
)


def _temporary_output() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    directory = tempfile.TemporaryDirectory(
        dir=freeze.evaluate.REPOSITORY_ROOT / "results"
    )
    output = Path(directory.name) / "evaluator-freeze.json"
    return directory, output.relative_to(freeze.evaluate.REPOSITORY_ROOT)


def _hide_completed_support_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        freeze.evaluate,
        "DEFAULT_OUTPUT_REPORT",
        Path("results/.cdltr-v1-freeze-test-missing-report.json"),
    )
    monkeypatch.setattr(
        freeze.evaluate,
        "DEFAULT_OUTPUT_CLOCK",
        Path("results/.cdltr-v1-freeze-test-missing-clock.csv.gz"),
    )


def test_freeze_reproduces_pinned_v1_evaluator_without_opening_source_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_completed_support_run(monkeypatch)
    monkeypatch.setattr(
        freeze.evaluate.pd,
        "read_csv",
        lambda *_args, **_kwargs: pytest.fail("source rows must remain unopened"),
    )
    directory, output = _temporary_output()
    try:
        status, payload = freeze.freeze(output)
        assert status == "created"
        assert payload["candidate"] == "CDLTR-72A"
        assert payload["evaluator_source_commit"] == (
            "6900b42ecc7d64c708218fcf048290e52ceb7a46"
        )
        assert payload["evaluator_source_sha256"] == (
            "649a4d4da64df32c3acb66ccedc6ad607bc8abef6b247235ff42e837ab3992e1"
        )
        assert payload["opened_source_value_rows"] == 0
        assert payload["opened_comparator_event_rows"] == 0
        assert payload["opened_btc_market_rows"] == 0
        assert payload["economic_simulation_run"] is False
        assert payload["mutable_parameters"] == []
        assert payload["manifest_hash"] == freeze.evaluate.canonical_hash(
            {key: value for key, value in payload.items() if key != "manifest_hash"}
        )

        status_again, same = freeze.freeze(output)
        assert status_again == "verified_existing"
        assert same == payload
    finally:
        directory.cleanup()


def test_tampered_existing_freeze_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_completed_support_run(monkeypatch)
    directory, output = _temporary_output()
    try:
        freeze.freeze(output)
        target = freeze.evaluate._repository_path(output)
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["opened_source_value_rows"] = 1
        target.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match="manifest hash mismatch"):
            freeze.freeze(output)
    finally:
        directory.cleanup()


def test_freeze_output_path_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_completed_support_run(monkeypatch)
    for unsafe in ("/tmp/freeze.json", "~/freeze.json", "../freeze.json"):
        with pytest.raises(RuntimeError, match="repository-relative"):
            freeze.freeze(unsafe)
