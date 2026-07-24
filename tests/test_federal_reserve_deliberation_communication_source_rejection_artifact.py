from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import (
    audit_federal_reserve_deliberation_communication_source as audit,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "results/federal_reserve_deliberation_communication_source_2026-07-25.json"
)
EXPECTED_FILE_SHA256 = (
    "7d9ee7a007dc1a066dc60ca27090c4f1f9fb68511a4439150d3e9d28c2a73801"
)
EXPECTED_REPORT_HASH = (
    "d36f44b01da6357edf980f77e873276ec4d7229415667175bd28280887506b8d"
)


def test_frdcl_rejection_report_hash_and_failure_are_frozen() -> None:
    raw = REPORT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_FILE_SHA256
    payload = json.loads(raw)
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "report_hash_without_self"
    }
    assert payload["report_hash_without_self"] == EXPECTED_REPORT_HASH
    assert (
        audit.sha256_bytes(audit.canonical_json_bytes(unsigned, newline=False))
        == EXPECTED_REPORT_HASH
    )
    assert payload["decision"] == "TERMINAL_REJECT"
    assert payload["source_audit_authoritative"] is True
    assert payload["retry_or_resume_authorized"] is False
    assert payload["mechanism_preregistration_authorized"] is False
    assert payload["failure"] == {
        "exception_class": "IndexError",
        "stage": "historical_indexes",
    }


def test_frdcl_rejection_kept_model_market_and_later_bodies_closed() -> None:
    payload = json.loads(REPORT.read_text())
    assert payload["outcome_boundary"] == {
        "article_text_emitted": False,
        "database_opened": False,
        "market_price_return_or_funding_opened": False,
        "model_tokenizer_adapter_prompt_or_checkpoint_opened": False,
        "portfolio_reward_or_performance_opened": False,
        "post_2020_document_body_opened": False,
        "semantic_label_embedding_or_inference_called": False,
    }
    assert "support" not in payload
    assert "source_hashes" not in payload
    assert payload["bindings"]["verifier_commit"] == (
        "01aaebae240bb98bc1a9685b96d6715684aa76ec"
    )
    assert payload["bindings"]["boundary_sha256"] == audit.BOUNDARY_SHA256
    assert payload["bindings"]["ledger_sha256"] == audit.LEDGER_SHA256
