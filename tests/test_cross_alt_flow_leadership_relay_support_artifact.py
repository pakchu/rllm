import hashlib
import json
from pathlib import Path

from training import build_cross_alt_flow_leadership_relay_support as support


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_caflr_source_support_pass_is_outcome_sealed():
    assert sha(support.RESULT) == "2c50171cd9c853b2994d2b9ea4fe561efb933dcc00ebf9d0079b76f40ba1968a"
    result = json.loads(support.RESULT.read_text())
    assert result["policy_id"] == "CAFLR-6"
    assert result["support_passed"] is True
    assert result["decision"] == "pass_to_novelty"
    assert result["advance_to_gross9_novelty"] is True
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train", "test", "eval", "final")] == [185, 377, 370, 191]


def test_caflr_support_hashes_bind_frozen_files():
    result = json.loads(support.RESULT.read_text())
    assert result["manifest_hash"] == support.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    assert result["source_manifest"]["sha256"] == sha(Path(result["source_manifest"]["path"]))
    for control in result["controls"].values():
        assert control["promotion_authorized"] is False
        assert control["sha256"] == sha(Path(control["path"]))
