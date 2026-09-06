import hashlib
import json
from pathlib import Path

from training import preregister_high_volatility_doi_research_attention_relay as prereg


ARTIFACT = Path("results/high_volatility_doi_research_attention_relay_preregistration_2026-08-12.json")


def test_hvdra_preregistration_artifact_is_sealed():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "edab5cf88d7cdd2b90b58aa938a4e09e5c58aeae27e3dd646b42bf2c129ef3a4"
    )
    value = json.loads(ARTIFACT.read_text())
    prereg.validate(value)
    assert value["policy_id"] == "HVDRA-24"
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["research_boundary"]["full_historical_candidate_incidence_opened"] is False
