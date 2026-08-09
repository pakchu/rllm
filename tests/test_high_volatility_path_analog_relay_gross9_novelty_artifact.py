import hashlib
import json

from training import evaluate_high_volatility_path_analog_relay_gross9_novelty as novelty


def test_hvpar_novelty_artifact_passes_without_oos_outcomes() -> None:
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == "d20c9a74d9d2ec21b3b5141844f5a033af8acb3ff09ed0c867afe881d26add00"
    payload = json.loads(novelty.OUTPUT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == novelty.chash(core)
    assert payload["gross9_novelty_status"] == "passed"
    assert payload["every_gross9_sleeve_passed"] is True
    assert payload["advance_to_economic_outcomes"] is True
    assert payload["evidence_boundary"]["outcomes_opened"] is False
    assert payload["evidence_boundary"]["btc_price_or_return_rows_opened"] == 0

