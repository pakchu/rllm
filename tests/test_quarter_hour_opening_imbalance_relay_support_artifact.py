import hashlib
import json
from pathlib import Path

from training import build_quarter_hour_opening_imbalance_relay_support as support


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qhoir_source_support_pass_is_outcome_sealed():
    result = json.loads(support.RESULT.read_text())
    assert result["policy_id"] == "QHOIR-8"
    assert result["support_passed"] is True
    assert result["decision"] == "pass_to_novelty"
    assert result["advance_to_gross9_novelty"] is True
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train", "test", "eval", "final")] == [234, 455, 434, 201]


def test_qhoir_support_hashes_bind_frozen_files():
    result = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == support.canonical_hash(core)
    for key in ("preregistration", "source_manifest"):
        record = result[key]
        assert record["sha256"] == sha(Path(record["path"]))
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    for record in result["controls"].values():
        assert record["promotion_authorized"] is False
        assert record["sha256"] == sha(Path(record["path"]))
