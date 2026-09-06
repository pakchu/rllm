import hashlib
import json
from pathlib import Path

from training import preregister_high_volatility_fan_token_result_rotation_relay as prereg


ARTIFACT = Path("results/high_volatility_fan_token_result_rotation_relay_preregistration_2026-08-12.json")


def test_hvftrr_preregistration_artifact_is_sealed():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "fd04741bca1211a27812d7a26ec4cabb6741b0b468f701d6886db4c6e9ed7ac1"
    )
    value = json.loads(ARTIFACT.read_text()); prereg.validate(value)
    assert value["policy_id"] == "HVFTRR-12"
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["research_boundary"]["multi_year_match_results_or_candidate_incidence_opened"] is False
