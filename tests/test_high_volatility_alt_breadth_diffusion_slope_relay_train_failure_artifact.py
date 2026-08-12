import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_alt_breadth_diffusion_slope_relay_train_economics_2026-08-13.json"
)
EXPECTED_SHA256 = "6e782edc8da6cc28e2c9030180e74c5b4e140348df6aec2344a529c9bbecee86"


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
    assert payload["policy_id"] == "HVABDS-8"
    assert payload["stage"] == "train"
    assert payload["passed"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["later_stage_outcomes_opened"] is False
    assert payload["primary"]["base"]["absolute_return_pct"] < 0
    assert payload["primary"]["base"]["mean_gross_underlying_bp"] < 0
    assert payload["advance_to_next_stage"] is False
    assert payload["advance_to_post_stage_volatility_audit"] is False
