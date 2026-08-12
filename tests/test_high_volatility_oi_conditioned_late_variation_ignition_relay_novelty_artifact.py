import hashlib
import json

from training import evaluate_high_volatility_oi_conditioned_late_variation_ignition_relay_gross9_novelty as n


def test_hvoilvi_novelty_is_frozen_pass_before_economics():
    assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest() == "7775869b4ce76e84b8f01d9c557feeea4261953ed36a474770d7dd036caf7d12"
    result = json.loads(n.OUTPUT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == n.canonical_hash(core) == "d2a282db895d929869b3b1657c74e7524969ce9b4d0a5785d7007ed1a944a029"
    assert result["every_gross9_sleeve_passed"] is True
    assert result["advance_to_economic_outcomes"] is True
    assert result["evidence_boundary"]["outcomes_opened"] is False
