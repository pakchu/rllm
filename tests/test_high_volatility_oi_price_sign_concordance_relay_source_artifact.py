import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_oi_price_sign_concordance_relay_support_2026-08-12.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvoipscr_source_artifact_is_terminal_and_hash_bound():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "HVOIPSCR-8"
    assert report["support_passed"] is False
    assert report["advance_to_gross9_novelty"] is False
    assert report["advance_to_economic_outcomes"] is False
    assert report["decision"] == "terminal_source_support_reject"
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert report["support_checks"]["test_month_concentration"] is False
    assert report["support"]["test"]["events"] == 15
    assert report["support"]["test"]["max_month_share"] == 2 / 3
    assert sha256(Path(report["clock"]["path"])) == report["clock"]["sha256"]
    assert sha256(Path(report["source_manifest"]["path"])) == report["source_manifest"]["sha256"]
