import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_cross_alt_extreme_traversal_consensus_relay_gross9_novelty_2026-08-12.json"
)


def test_hvcatcr_novelty_artifact_passes_and_is_hash_bound():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "HVCATCR-8"
    assert report["gross9_novelty_status"] == "passed"
    assert report["every_gross9_sleeve_passed"] is True
    assert report["advance_to_economic_outcomes"] is True
    assert report["evidence_boundary"]["outcomes_opened"] is False
    assert all(item["passed"] for item in report["gross9_sleeves"].values())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    from training import export_gross9_structural_clocks as gross9

    assert report["manifest_hash"] == gross9.canonical_hash(core)
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "f361996ec06fd029dfe9dce24ff5e48d8e41e2ad11d7489264012dae35cd4747"
