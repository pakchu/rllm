import hashlib
import json
from pathlib import Path

from training import build_high_volatility_dollar_breadth_relay_support as support


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvdbr_source_failure_is_terminal_and_outcome_sealed():
    result = json.loads(support.RESULT.read_text())
    assert result["policy_id"] == "HVDBR-12"
    assert result["support_passed"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train", "test", "eval", "final")] == [3, 15, 19, 10]
    assert result["support"]["train"]["shorts"] == 0


def test_hvdbr_artifact_hashes_bind_frozen_files():
    result = json.loads(support.RESULT.read_text())
    assert result["manifest_hash"] == support.chash({key: value for key, value in result.items() if key != "manifest_hash"})
    assert result["preregistration"]["sha256"] == sha(Path(result["preregistration"]["path"]))
    for item in result["source_manifests"].values():
        assert item["sha256"] == sha(Path(item["path"]))
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    for item in result["controls"].values():
        assert item["promotion_authorized"] is False
        assert item["sha256"] == sha(Path(item["path"]))
