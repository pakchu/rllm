import hashlib
import json

from training import evaluate_high_volatility_online_expert_rotation_relay_gross9_novelty as n


def test_hvoer_novelty_is_frozen_pass_before_economics():
    assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest() == "ee560bed5d29b0927f46a3e5957d91aa1cbd92cc7b7fb42644c4cda09290a4e9"
    result = json.loads(n.OUTPUT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == n.canonical_hash(core) == "f33213cb83ebc6e1df2d4e26753588b64427a1878d05a8f5e1eba36641843bdd"
    assert result["every_gross9_sleeve_passed"] is True
    assert result["advance_to_economic_outcomes"] is True
    assert result["evidence_boundary"]["outcomes_opened"] is False
