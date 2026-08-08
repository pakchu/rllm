import hashlib
import json
from pathlib import Path

from training import build_cross_alt_flow_acceleration_exhaustion_reversal_support as support


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cafaer_source_support_pass_is_outcome_sealed():
    assert sha(support.RESULT) == "c5567d724dc07aae7d737ce2f2ff44c9e9730857136a2853829ae9ebb454d8a4"
    result = json.loads(support.RESULT.read_text())
    assert result["policy_id"] == "CAFAER-12"
    assert result["support_passed"] is True
    assert result["decision"] == "pass_to_novelty"
    assert result["advance_to_gross9_novelty"] is True
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train", "test", "eval", "final")] == [35, 109, 57, 60]


def test_cafaer_support_hashes_bind_frozen_files():
    result = json.loads(support.RESULT.read_text())
    assert result["manifest_hash"] == support.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    assert result["source_manifest"]["sha256"] == sha(Path(result["source_manifest"]["path"]))
    for control in result["controls"].values():
        assert control["promotion_authorized"] is False
        assert control["sha256"] == sha(Path(control["path"]))
