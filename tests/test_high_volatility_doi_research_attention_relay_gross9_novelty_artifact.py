import hashlib
import json

from training import evaluate_high_volatility_doi_research_attention_relay_gross9_novelty as novelty


def test_novelty_artifact_passes_and_keeps_economics_closed():
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == (
        "7e5f9f2f4fdcc78bb8cb4d963230b036a98d87b6f692eae19f5d611a21dcb8d9"
    )
    payload = json.loads(novelty.OUTPUT.read_text())
    assert payload["policy_id"] == "HVDRA-24"
    assert payload["source_support_passed"] is True
    assert payload["every_gross9_sleeve_passed"] is True
    assert payload["advance_to_economic_outcomes"] is True
    assert payload["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    assert payload["evidence_boundary"]["outcomes_opened"] is False
    digest = payload.pop("manifest_hash")
    assert novelty.chash(payload) == digest


def test_every_sleeve_passes_every_limit():
    payload = json.loads(novelty.OUTPUT.read_text())
    assert len(payload["gross9_sleeves"]) == 5
    for sleeve in payload["gross9_sleeves"].values():
        assert sleeve["passed"] is True
        assert all(sleeve["checks"].values())
