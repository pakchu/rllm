import hashlib
import json
from pathlib import Path

from training import build_high_volatility_hikkake_pattern_relay_support as support


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_support_pass_is_outcome_sealed() -> None:
    result = json.loads(support.RESULT.read_text())
    assert result["policy_id"] == "HVHIKKAKE-C3-8"
    assert result["support_passed"] is True
    assert result["decision"] == "pass_to_novelty"
    assert result["advance_to_gross9_novelty"] is True
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["funding_values_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train", "test", "eval", "final")] == [41, 97, 78, 37]


def test_source_support_hashes_bind_frozen_artifacts() -> None:
    result = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == support.canonical_hash(core)
    for key in ("preregistration", "source_manifest"):
        item = result[key]
        assert item["sha256"] == sha(Path(item["path"]))
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    assert sha(support.RESULT) == "d348276dd539ee7ce9aab1bfe7198dc0801c9d97b209bf5fb03aec7632af0f9a"
    for item in result["controls"].values():
        assert item["promotion_authorized"] is False
        assert item["sha256"] == sha(Path(item["path"]))
