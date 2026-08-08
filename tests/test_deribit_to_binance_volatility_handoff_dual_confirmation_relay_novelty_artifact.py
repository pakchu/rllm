import hashlib
import json

from training import evaluate_deribit_to_binance_volatility_handoff_dual_confirmation_relay_gross9_novelty as novelty


def test_dbvhdr_novelty_is_frozen_pass_before_economics():
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == "8cbe053a89b19b8bcc97cca9acdad6b3474d2a0277f8a0c9c0b77050c186ca0b"
    data = json.loads(novelty.OUTPUT.read_text())
    core = {key: value for key, value in data.items() if key != "manifest_hash"}
    assert data["manifest_hash"] == novelty.canonical_hash(core) == "4a6482c7a9f948c9f4b454bf3e3eb0ff9af61347291bc1fbf79e3562ea26496e"
    assert data["gross9_novelty_status"] == "passed"
    assert data["every_gross9_sleeve_passed"] is True
    assert data["advance_to_economic_outcomes"] is True
    assert data["evidence_boundary"]["outcomes_opened"] is False
    assert max(item["metrics"]["one_to_one_6h_max_matched_share"] for item in data["gross9_sleeves"].values()) < 0.21
