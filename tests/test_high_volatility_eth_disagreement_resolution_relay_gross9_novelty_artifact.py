import hashlib
import json

from training import evaluate_high_volatility_eth_disagreement_resolution_relay_gross9_novelty as novelty


def test_hvedr_novelty_artifact_passes_without_outcomes():
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == "8463bb61c08741c07a4e01882fc122b668a1a78cd3cc8ac9c017ac1380fb4d08"
    payload = json.loads(novelty.OUTPUT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == novelty.chash(core)
    assert payload["gross9_novelty_status"] == "passed"
    assert payload["every_gross9_sleeve_passed"] is True
    assert payload["advance_to_economic_outcomes"] is True
    assert payload["evidence_boundary"]["outcomes_opened"] is False
    assert payload["evidence_boundary"]["btc_price_or_return_rows_opened"] == 0
