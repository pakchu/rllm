import hashlib
import json
from pathlib import Path


RESULT = Path("results/bitcoin_stock_correlation_break_relay_support_2026-08-09.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_bscbr_support_artifact_is_hash_bound_terminal_rejection() -> None:
    report = json.loads(RESULT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == _canonical_hash(core)
    assert report["decision"] == "terminal_source_support_reject"
    assert report["support_passed"] is False
    assert report["advance_to_gross9_novelty"] is False
    assert report["advance_to_economic_outcomes"] is False
    failed = [name for name, passed in report["support_checks"].items() if not passed]
    assert failed == ["final_month_concentration"]


def test_bscbr_support_artifact_preserves_outcome_boundary_and_files() -> None:
    report = json.loads(RESULT.read_text())
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["gross9_rows_opened"] is False
    clock = Path(report["clock"]["path"])
    assert _sha256(clock) == report["clock"]["sha256"]
    source = report["source_manifest"]
    assert _sha256(Path(source["path"])) == source["sha256"]
    for control in report["controls"].values():
        assert control["promotion_authorized"] is False
        assert _sha256(Path(control["path"])) == control["sha256"]
