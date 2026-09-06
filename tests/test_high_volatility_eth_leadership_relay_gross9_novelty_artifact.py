import hashlib
import json

from training import evaluate_high_volatility_eth_leadership_relay_gross9_novelty as novelty


def test_hvelr_novelty_artifact_passes_without_outcomes():
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == "5d1ce1ddfdeaaab45c1a575f4852a3f0d3f0c51ca63d7652ccdb618b1c8d796b"
    payload = json.loads(novelty.OUTPUT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == novelty.chash(core)
    assert payload["gross9_novelty_status"] == "passed"
    assert payload["every_gross9_sleeve_passed"] is True
    assert payload["advance_to_economic_outcomes"] is True
    assert payload["evidence_boundary"]["outcomes_opened"] is False
    assert payload["evidence_boundary"]["btc_price_or_return_rows_opened"] == 0
