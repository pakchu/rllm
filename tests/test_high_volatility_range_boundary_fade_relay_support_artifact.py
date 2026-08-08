import hashlib
import json
from pathlib import Path

from training import build_high_volatility_range_boundary_fade_relay_support as support


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvrbfr_source_failure_is_terminal_and_outcome_sealed():
    result = json.loads(support.RESULT.read_text())
    assert result["policy_id"] == "HVRBFR-2"
    assert result["support_passed"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert all(values["events"] == 0 for values in result["support"].values())


def test_hvrbfr_artifact_hashes_bind_frozen_files():
    result = json.loads(support.RESULT.read_text())
    assert result["manifest_hash"] == support.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
    for key in ("preregistration", "source_manifest"):
        record = result[key]
        assert record["sha256"] == sha(Path(record["path"]))
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    for record in result["controls"].values():
        assert record["promotion_authorized"] is False
        assert record["sha256"] == sha(Path(record["path"]))
