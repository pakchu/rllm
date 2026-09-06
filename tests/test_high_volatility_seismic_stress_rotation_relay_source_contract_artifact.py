import hashlib
import json
from pathlib import Path

from training import audit_high_volatility_seismic_stress_rotation_source_contract as audit


ARTIFACT = Path("results/high_volatility_seismic_stress_rotation_relay_source_contract_2026-08-12.json")


def test_hvssr_source_contract_pass_is_sealed():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "28dfbe32f2d3705cdc2a270435892564de247b59ca9ca6bbe2bdbbaa7636e8b6"
    )
    result = json.loads(ARTIFACT.read_text())
    assert result["policy_id"] == "HVSSR-24"
    assert result["probe"]["geojson_events"] == result["probe"]["quakeml_events"] == 372
    assert result["probe"]["events_with_multiple_origins"] >= 1
    assert result["probe"]["events_with_multiple_magnitudes"] >= 1
    assert result["source_contract_passed"] is True
    assert result["advance_to_source_incidence"] is True
    assert result["candidate_incidence_opened"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["decision"] == "pass_to_source_incidence"


def test_hvssr_source_contract_manifest_is_hash_bound():
    result = json.loads(ARTIFACT.read_text())
    assert result["manifest_hash"] == audit.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
    assert result["source_evaluator"]["sha256"] == audit.sha(Path(result["source_evaluator"]["path"]))
