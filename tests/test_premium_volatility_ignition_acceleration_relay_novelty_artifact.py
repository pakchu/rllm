import hashlib
import json

from training import evaluate_premium_volatility_ignition_acceleration_relay_gross9_novelty as novelty


def test_pviar_novelty_frozen_pass_without_outcomes():
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == "71c6ec6cf33729ef8e513452441eec707fbf7b46e040b5163e598249d3b0461b"
    report = json.loads(novelty.OUTPUT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == novelty.canonical_hash(core)
    assert report["every_gross9_sleeve_passed"] is True
    assert report["advance_to_economic_outcomes"] is True
    assert report["evidence_boundary"]["outcomes_opened"] is False
    assert max(
        result["metrics"]["one_to_one_6h_max_matched_share"]
        for result in report["gross9_sleeves"].values()
    ) <= 0.16049382716049382
