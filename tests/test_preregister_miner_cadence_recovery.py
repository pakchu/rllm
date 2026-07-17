from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from training import preregister_miner_cadence_recovery as prereg


def test_protocol_is_single_policy_outcome_blind_and_seals_2024() -> None:
    payload = prereg.manifest()
    prereg.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["policy"] == asdict(prereg.Policy())
    assert payload["selection_protocol"]["candidate_count"] == 1
    assert payload["selection_protocol"]["sealed"] == ["2024", "2025", "2026_ytd"]
    assert payload["source_contract"]["columns_allowed"] == [
        "observation_date",
        "available_at",
        "HashRate",
        "BlkCnt",
    ]
    assert payload["causal_feature_contract"]["price_or_derivative_feature_columns_loaded"] == []


def test_manifest_hash_detects_policy_drift() -> None:
    payload = prereg.manifest()
    payload["policy"]["hold_bars"] += 1
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        prereg.validate_manifest(payload)


def test_run_writes_matching_artifacts(tmp_path) -> None:
    output = tmp_path / "mcr.json"
    docs = tmp_path / "mcr.md"
    payload = prereg.run(str(output), str(docs))
    assert json.loads(output.read_text())["manifest_hash"] == payload["manifest_hash"]
    assert payload["manifest_hash"] in docs.read_text()
    assert "2024+" in docs.read_text()
