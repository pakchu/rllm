import hashlib, json
from training import evaluate_daily_taker_flow_acceleration_relay_gross9_novelty as novelty


def test_dtfar_novelty_artifact_passes_without_outcomes():
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == "4bafb1b1eb2402624444c7d9f199786d1088df9bab405ae51b6342586ee9903f"
    payload = json.loads(novelty.OUTPUT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == novelty.chash(core)
    assert payload["gross9_novelty_status"] == "passed"
    assert payload["every_gross9_sleeve_passed"] is True
    assert payload["advance_to_economic_outcomes"] is True
    assert payload["evidence_boundary"]["outcomes_opened"] is False
    assert payload["evidence_boundary"]["btc_price_or_return_rows_opened"] == 0
