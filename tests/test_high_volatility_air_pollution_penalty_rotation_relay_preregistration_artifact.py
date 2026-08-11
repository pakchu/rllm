import hashlib
import json
from pathlib import Path

from training import preregister_high_volatility_air_pollution_penalty_rotation_relay as prereg


ARTIFACT = Path("results/high_volatility_air_pollution_penalty_rotation_relay_preregistration_2026-08-12.json")


def test_hvappr_preregistration_artifact_is_sealed():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "d886f157c5c8b56d78736a63749fb0ed6c96fc753f9d6b385ae0597eb5ea1383"
    )
    value = json.loads(ARTIFACT.read_text())
    prereg.validate(value)
    assert value["policy_id"] == "HVAPPR-24"
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["research_boundary"]["multi_day_air_quality_values_or_candidate_incidence_opened"] is False
