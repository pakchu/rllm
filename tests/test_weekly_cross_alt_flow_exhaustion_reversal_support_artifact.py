import hashlib
import json
from pathlib import Path

from training import build_weekly_cross_alt_flow_exhaustion_reversal_support as support


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_wcafer_source_rejection_is_sealed():
    assert sha(support.RESULT) == "26b32754772f9ebacf07d238ea9a1e810853782ccfa0cd1d56b3d10295cd1f5b"
    result = json.loads(support.RESULT.read_text())
    assert result["policy_id"] == "WCAFER-24"
    assert result["support_passed"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train", "test", "eval", "final")] == [6, 25, 9, 9]


def test_wcafer_hashes_and_controls_are_bound():
    result = json.loads(support.RESULT.read_text())
    assert result["manifest_hash"] == support.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    assert result["source_manifest"]["sha256"] == sha(Path(result["source_manifest"]["path"]))
    for control in result["controls"].values():
        assert control["promotion_authorized"] is False
        assert control["sha256"] == sha(Path(control["path"]))
