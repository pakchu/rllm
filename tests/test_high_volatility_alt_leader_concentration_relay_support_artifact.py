import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_alt_leader_concentration_relay_support_2026-08-13.json")
EXPECTED_SHA256 = "40c7d80e1968644c2e986f615def3252c181b8d98538bc3fcc3081e2bb606c52"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def test_support_artifact_is_immutable_and_passes_all_gates():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVALCR-8"
    assert payload["support_passed"] is True
    assert all(payload["support_checks"].values())
    assert {name: row["events"] for name, row in payload["support"].items()} == {
        "train": 47, "test": 79, "eval": 66, "final": 28,
    }
    assert payload["advance_to_gross9_novelty"] is True
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
