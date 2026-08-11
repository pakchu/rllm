import hashlib
import json
from pathlib import Path

from training import audit_high_volatility_geomagnetic_activity_rotation_source_contract as audit


ARTIFACT = Path(
    "results/high_volatility_geomagnetic_activity_rotation_relay_source_contract_failure_2026-08-12.json"
)


def test_hvgmr_source_contract_failure_is_terminal_and_sealed():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "91ecf36ecb5aa8face66519503a5bb956757e5e7a87911b307ead08ca7a0096e"
    )
    result = json.loads(ARTIFACT.read_text())
    assert result["policy_id"] == "HVGMR-24"
    assert result["probe"]["status_counts"] == {"def": 24}
    assert result["probe"]["all_rows_preserve_preregistered_nowcast_status"] is False
    assert result["decision"] == "terminal_source_contract_reject_no_repair"
    assert result["support_passed"] is False
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["candidate_incidence_opened"] is False
    assert result["btc_rows_opened"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False


def test_hvgmr_source_failure_manifest_is_hash_bound():
    result = json.loads(ARTIFACT.read_text())
    assert result["manifest_hash"] == audit.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
    assert result["source_evaluator"]["sha256"] == audit.sha(
        Path(result["source_evaluator"]["path"])
    )
