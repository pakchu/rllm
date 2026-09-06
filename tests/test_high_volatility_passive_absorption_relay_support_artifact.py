import hashlib
import json
from pathlib import Path

from training import build_high_volatility_passive_absorption_relay_support as support


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvpar_source_failure_is_terminal_and_outcome_sealed():
    result = json.loads(support.RESULT.read_text())
    assert result["policy_id"] == "HVPAR-6"
    assert result["support_passed"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train", "test", "eval", "final")] == [1, 1, 2, 1]
    assert all(result["support"][stage]["shorts"] == 0 for stage in ("train", "test", "eval", "final"))


def test_hvpar_artifact_hashes_bind_frozen_files():
    result = json.loads(support.RESULT.read_text())
    assert result["manifest_hash"] == support.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
    for key in ("preregistration", "source_manifest"):
        item = result[key]
        assert item["sha256"] == sha(Path(item["path"]))
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    for item in result["controls"].values():
        assert item["promotion_authorized"] is False
        assert item["sha256"] == sha(Path(item["path"]))
