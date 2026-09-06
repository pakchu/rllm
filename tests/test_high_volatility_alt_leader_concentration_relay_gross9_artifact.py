import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_alt_leader_concentration_relay_gross9_novelty_2026-08-13.json")
EXPECTED_SHA256 = "1f5f91568a0d23cea5e9289b5938ecbf789dba54666326f65d9fc97c450a3426"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def test_gross9_artifact_is_immutable_and_passes_before_economics():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVALCR-8"
    assert payload["every_gross9_sleeve_passed"] is True
    assert all(row["passed"] for row in payload["gross9_sleeves"].values())
    assert payload["advance_to_economic_outcomes"] is True
    assert payload["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    assert payload["evidence_boundary"]["portfolio_return_or_pnl_metrics_computed"] is False
