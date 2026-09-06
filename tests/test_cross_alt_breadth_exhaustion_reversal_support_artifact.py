import hashlib
import json
from pathlib import Path

from training import build_cross_alt_breadth_exhaustion_reversal_support as support


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_caber_source_rejection_is_outcome_sealed_and_terminal():
    assert sha(support.RESULT) == "9fef488a17363b9588be7569c95dca92d29e079fda93390eee2d0d1fcc4fc2f1"
    result = json.loads(support.RESULT.read_text())
    assert result["support_passed"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train", "test", "eval", "final")] == [19, 69, 71, 44]
    assert result["support"]["train"]["max_month_share"] == 10 / 19
    assert result["support_checks"]["train_month_concentration"] is False


def test_caber_artifact_hashes_and_controls_are_bound():
    result = json.loads(support.RESULT.read_text())
    assert result["manifest_hash"] == support.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    assert result["source_manifest"]["sha256"] == sha(Path(result["source_manifest"]["path"]))
    for control in result["controls"].values():
        assert control["promotion_authorized"] is False
        assert control["sha256"] == sha(Path(control["path"]))
