import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_cross_alt_quarter_hour_flow_consensus_relay_gross9_novelty_2026-08-13.json")
EXPECTED_SHA256 = "03ce2fa0fbc56d05973abfbc83fb863b14388a9aeae83b000bdd334d7bd5fc98"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def test_gross9_artifact_is_immutable_and_terminal_before_economics():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVCAQF-6"
    assert payload["every_gross9_sleeve_passed"] is False
    assert payload["gross9_novelty_status"] == "failed"
    assert payload["advance_to_economic_outcomes"] is False
    assert all(
        not row["checks"]["one_to_one_6h_max_matched_share"]
        for row in payload["gross9_sleeves"].values()
    )
    assert payload["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    assert payload["evidence_boundary"]["portfolio_return_or_pnl_metrics_computed"] is False
