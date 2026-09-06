import hashlib
import json
from pathlib import Path

from training import preregister_high_volatility_lottery_sales_risk_appetite_relay as prereg


ARTIFACT = Path("results/high_volatility_lottery_sales_risk_appetite_relay_preregistration_2026-08-12.json")


def test_hvlsra_preregistration_artifact_is_sealed():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "80b12477f0d4676da43d83abd0883216a081ce993613c4cd3f76fea8a51379f7"
    )
    value = json.loads(ARTIFACT.read_text())
    prereg.validate(value)
    assert value["policy_id"] == "HVLSRA-24"
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["research_boundary"]["numeric_sales_or_candidate_incidence_opened"] is False
