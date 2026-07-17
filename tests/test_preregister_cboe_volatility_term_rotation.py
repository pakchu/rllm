from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_cboe_volatility_term_rotation as prereg


def test_manifest_is_singleton_outcome_blind_and_source_only() -> None:
    payload = prereg.build_manifest()
    prereg.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["selection_protocol"]["candidate_count"] == 1
    assert payload["causal_feature_contract"]["price_or_derivative_feature_columns_loaded"] == []
    disclosure = payload["research_history_boundary"]["disclosure"]
    assert disclosure["outcomes_joined"] is False
    assert disclosure["clocks"]["primary"]["stage1"]["events"] >= 200


def test_write_once_refuses_policy_change(tmp_path: Path) -> None:
    path = tmp_path / "prereg.json"
    payload = prereg.build_manifest()
    assert prereg.write_once(path, payload) == "created"
    assert prereg.write_once(path, payload) == "verified_existing"
    changed = json.loads(json.dumps(payload))
    changed["policy"]["lower_tail_rank"] = 0.20
    with pytest.raises(RuntimeError):
        prereg.write_once(path, changed)


def test_manifest_hash_detects_mutation() -> None:
    payload = prereg.build_manifest()
    payload["controls"]["constant_long"] = "mutated"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        prereg.validate_manifest(payload, verify_sources=False)
