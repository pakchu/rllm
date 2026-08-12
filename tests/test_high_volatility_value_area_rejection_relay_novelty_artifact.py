import hashlib
import json

from training import evaluate_high_volatility_value_area_rejection_relay_gross9_novelty as novelty


def test_hvvar_novelty_is_frozen_pass_before_economics() -> None:
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == (
        "69ce9e93bd853abfd52e36621e27b60cc530a2c5ac6e9e6c558e226ad3d72a3f"
    )
    result = json.loads(novelty.OUTPUT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == novelty.canonical_hash(core) == (
        "693015ee7428aebe13edc2a7891adc23a0b132c30ac108f53f6b0af70f553f20"
    )
    assert result["every_gross9_sleeve_passed"] is True
    assert result["advance_to_economic_outcomes"] is True
    assert result["evidence_boundary"]["outcomes_opened"] is False
