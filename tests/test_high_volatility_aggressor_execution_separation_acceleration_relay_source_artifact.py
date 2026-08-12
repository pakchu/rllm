import hashlib
import json
from pathlib import Path

RESULT = Path("results/high_volatility_aggressor_execution_separation_acceleration_relay_support_2026-08-12.json")


def test_hvaesar_source_artifact_is_terminal_and_hash_bound():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "HVAESAR-8"
    assert report["support_passed"] is False
    assert report["advance_to_gross9_novelty"] is False
    assert report["decision"] == "terminal_source_support_reject"
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert [report["support"][s]["shorts"] for s in ("train", "test", "eval", "final")] == [1, 1, 1, 0]
    assert all(report["support_checks"][f"{s}_side_balance"] is False for s in ("train", "test", "eval", "final"))
    manifest = Path(report["source_manifest"]["path"])
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == report["source_manifest"]["sha256"]
