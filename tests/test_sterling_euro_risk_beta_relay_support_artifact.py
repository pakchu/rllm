import hashlib
import json
from pathlib import Path

from training import build_sterling_euro_risk_beta_relay_support as support


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_serbr_source_support_pass_is_outcome_sealed():
    result = json.loads(support.RESULT.read_text())
    assert sha(support.RESULT) == "23816135a263e6cb4568bc357eb67d6725e9c63963bbea62d85033f71690daef"
    assert result["policy_id"] == "SERBR-12"
    assert result["support_passed"] is True
    assert result["decision"] == "pass_to_novelty"
    assert result["advance_to_gross9_novelty"] is True
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train", "test", "eval", "final")] == [20, 107, 61, 57]


def test_serbr_support_hashes_bind_frozen_files():
    result = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == support.chash(core)
    for key in ("preregistration", "source_manifest"):
        item = result[key]
        assert item["sha256"] == sha(Path(item["path"]))
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    for item in result["controls"].values():
        assert item["promotion_authorized"] is False
        assert item["sha256"] == sha(Path(item["path"]))
