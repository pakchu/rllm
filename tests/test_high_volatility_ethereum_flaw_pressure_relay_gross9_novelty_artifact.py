import hashlib
import json

from training import evaluate_high_volatility_ethereum_flaw_pressure_relay_gross9_novelty as novelty


def test_novelty_artifact_passes_and_keeps_economics_closed():
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == (
        "836d3a517bebb62783f3735410da4463a35cdd2fdb56020844c996787dd1fd03"
    )
    payload = json.loads(novelty.OUTPUT.read_text())
    assert payload["policy_id"] == "HVEFPR-24"
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
