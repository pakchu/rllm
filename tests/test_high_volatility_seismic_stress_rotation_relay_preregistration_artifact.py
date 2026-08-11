import hashlib
import json
from pathlib import Path

from training import preregister_high_volatility_seismic_stress_rotation_relay as prereg


ARTIFACT = Path("results/high_volatility_seismic_stress_rotation_relay_preregistration_2026-08-12.json")


def test_hvssr_preregistration_artifact_is_sealed():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "92dc3d2086e7197311e386f4c62da3b2cdf247256d6b5ac02482415dcf0b079d"
    )
    value = json.loads(ARTIFACT.read_text()); prereg.validate(value)
    assert value["policy_id"] == "HVSSR-24"
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["research_boundary"]["earthquake_event_values_or_candidate_incidence_opened"] is False
