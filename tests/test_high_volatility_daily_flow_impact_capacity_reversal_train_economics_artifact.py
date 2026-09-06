import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_daily_flow_impact_capacity_reversal_train_economics_2026-08-13.json")
EXPECTED_SHA256 = "349bf28a80bce3de19ec4dc94093c71ff71bc7b3a306addd241c5a73b0828f2a"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def test_train_failure_is_immutable_and_terminal():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVDFICR-12"
    assert payload["stage"] == "train"
    assert payload["passed"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["later_stage_outcomes_opened"] is False
    assert payload["primary"]["base"]["absolute_return_pct"] < 0
    assert payload["primary"]["base"]["mean_gross_underlying_bp"] < 0
    assert payload["primary"]["calendar_halves"]["first"]["absolute_return_pct"] < 0
    assert payload["primary"]["calendar_halves"]["second"]["absolute_return_pct"] < 0
