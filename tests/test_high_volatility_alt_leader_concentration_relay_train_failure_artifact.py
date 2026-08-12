import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_alt_leader_concentration_relay_train_economics_2026-08-13.json"
)
EXPECTED_SHA256 = "1c3906e67ca9a78a6c1e374354cd3c42afc54fba37ac8c457ee87006d4ef00ff"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_train_failure_is_immutable_and_terminal():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVALCR-8"
    assert payload["stage"] == "train"
    assert payload["passed"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["later_stage_outcomes_opened"] is False
    assert payload["primary"]["base"]["absolute_return_pct"] < 0
    assert payload["primary"]["base"]["mean_gross_underlying_bp"] < 20
    assert payload["advance_to_next_stage"] is False
    assert payload["advance_to_post_stage_volatility_audit"] is False
