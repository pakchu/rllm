import hashlib
import json
from pathlib import Path


PATH = Path("results/high_volatility_oi_turnover_scarcity_continuation_source_execution_failure_2026-08-10.json")
EXPECTED_SHA256 = "81e3025397795bd5af0dea4eabb0d5f2f8758ad3b38281abfb20333838226b98"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_hvotsc_source_failure_is_immutable_and_terminal():
    assert hashlib.sha256(PATH.read_bytes()).hexdigest() == EXPECTED_SHA256
    artifact = json.loads(PATH.read_text())
    manifest_hash = artifact.pop("manifest_hash")
    assert canonical_hash(artifact) == manifest_hash
    assert artifact["policy_id"] == "HVOTSC-8"
    assert artifact["execution"]["exception"] == "RuntimeError: HVOTSC invalid OI key"
    assert artifact["source_audit"]["rows"] == 376570
    assert artifact["source_audit"]["null_observed_at_rows"] == 376570
    assert artifact["source_audit"]["null_observed_at_share"] == 1.0
    assert artifact["artifacts"]["candidate_clock_produced"] is False
    assert artifact["support_passed"] is False
    assert artifact["advance_to_gross9_novelty"] is False
    assert artifact["advance_to_economic_outcomes"] is False
    assert artifact["decision"] == "terminal_source_execution_reject_no_repair"
    assert artifact["repair_forbidden"] is True
    assert artifact["downstream_stages_opened"] is False
    assert not any(
        artifact["evidence_opened"][key]
        for key in ("postentry_return_pnl_execution_price", "funding_values", "gross9_rows")
    )
