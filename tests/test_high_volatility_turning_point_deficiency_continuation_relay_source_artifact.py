import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_turning_point_deficiency_continuation_relay_support_2026-08-12.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvtpdcr_source_artifact_passes_and_is_hash_bound():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "HVTPDCR-8"
    assert report["support_passed"] is True
    assert report["advance_to_gross9_novelty"] is True
    assert report["advance_to_economic_outcomes"] is False
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert all(report["support_checks"].values())
    assert {name: values["events"] for name, values in report["support"].items()} == {
        "train": 26,
        "test": 57,
        "eval": 38,
        "final": 17,
    }
    assert sha256(Path(report["clock"]["path"])) == report["clock"]["sha256"]
    assert sha256(Path(report["source_manifest"]["path"])) == report["source_manifest"]["sha256"]
    for control in report["controls"].values():
        assert sha256(Path(control["path"])) == control["sha256"]
        assert control["promotion_authorized"] is False
