import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_alt_breadth_diffusion_slope_relay_gross9_novelty_2026-08-13.json"
)
EXPECTED_SHA256 = "59e0ea96d227376e6e89ac2e7783d92086d65ffcece7f00e4c9d614d5781eec4"


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


def test_gross9_artifact_is_immutable_and_passes_without_economics():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVABDS-8"
    assert payload["source_support_passed"] is True
    assert payload["every_gross9_sleeve_passed"] is True
    assert all(row["passed"] for row in payload["gross9_sleeves"].values())
    assert payload["advance_to_economic_outcomes"] is True
    assert payload["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    assert payload["evidence_boundary"]["portfolio_return_or_pnl_metrics_computed"] is False
