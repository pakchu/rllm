import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_cross_alt_extreme_traversal_consensus_relay_support_2026-08-12.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvcatcr_source_artifact_passes_and_is_hash_bound():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "HVCATCR-8"
    assert report["support_passed"] is True
    assert report["advance_to_gross9_novelty"] is True
    assert report["advance_to_economic_outcomes"] is False
    assert report["decision"] == "pass_to_novelty"
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert {split: report["support"][split]["events"] for split in ("train", "test", "eval", "final")} == {
        "train": 65,
        "test": 119,
        "eval": 114,
        "final": 37,
    }
    assert all(report["support_checks"].values())
    assert sha256(Path(report["clock"]["path"])) == report["clock"]["sha256"]
    assert sha256(Path(report["source_manifest"]["path"])) == report["source_manifest"]["sha256"]
