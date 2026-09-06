import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_cross_alt_directional_coherence_relay_support_2026-08-12.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvcadcr_source_artifact_is_terminal_and_hash_bound():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "HVCADCR-8"
    assert report["support_passed"] is False
    assert report["advance_to_gross9_novelty"] is False
    assert report["advance_to_economic_outcomes"] is False
    assert report["decision"] == "terminal_source_support_reject"
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert all(report["support"][split]["events"] == 0 for split in ("train", "test", "eval", "final"))
    assert report["controls"]["no_coherence_gate"]["rows"] == 178
    assert report["controls"]["no_variation_gate"]["rows"] == 0
    assert sha256(Path(report["clock"]["path"])) == report["clock"]["sha256"]
    assert sha256(Path(report["source_manifest"]["path"])) == report["source_manifest"]["sha256"]
