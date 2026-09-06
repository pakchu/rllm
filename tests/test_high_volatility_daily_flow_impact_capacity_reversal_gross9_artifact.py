import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_daily_flow_impact_capacity_reversal_gross9_novelty_2026-08-13.json")
EXPECTED_SHA256 = "0981bccee9b0f45a97a4f8b5933dea091b3b40f5bd04ee275f5f2abce7e943d5"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def test_gross9_artifact_is_immutable_and_passes_before_economics():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVDFICR-12"
    assert payload["every_gross9_sleeve_passed"] is True
    assert all(row["passed"] for row in payload["gross9_sleeves"].values())
    assert payload["advance_to_economic_outcomes"] is True
    assert max(
        row["metrics"]["one_to_one_6h_max_matched_share"]
        for row in payload["gross9_sleeves"].values()
    ) <= 0.35
    assert payload["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    assert payload["evidence_boundary"]["portfolio_return_or_pnl_metrics_computed"] is False
