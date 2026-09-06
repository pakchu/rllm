import hashlib
import json

from training import evaluate_high_volatility_intrinsic_topology_ridge_relay_gross9_novelty as novelty


def test_hvitr_novelty_artifact_passes_without_oos_outcomes() -> None:
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == "3001c955f196b158f6417fdeed88c8f1dbba3e5b6f50d095e839807e9ef4473e"
    payload = json.loads(novelty.OUTPUT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == novelty.chash(core)
    assert payload["gross9_novelty_status"] == "passed"
    assert payload["every_gross9_sleeve_passed"] is True
    assert payload["advance_to_economic_outcomes"] is True
    assert payload["evidence_boundary"]["outcomes_opened"] is False
    assert payload["evidence_boundary"]["btc_price_or_return_rows_opened"] == 0
    assert payload["evidence_boundary"]["economic_outcome_rows_opened"] == 0

