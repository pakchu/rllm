import hashlib
import json
from pathlib import Path

from training import (
    build_high_volatility_relative_daily_volume_continuation_relay_support as support,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvrdv_source_pass_is_frozen_and_downstream_remains_sealed():
    result = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == support.canonical_hash(core)
    assert result["policy_id"] == "HVRDV-8"
    assert result["support_passed"] is True
    assert result["advance_to_gross9_novelty"] is True
    assert result["advance_to_economic_outcomes"] is False
    assert result["decision"] == "pass_to_novelty"
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["funding_values_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [
        result["support"][stage]["events"]
        for stage in ("train", "test", "eval", "final")
    ] == [57, 107, 107, 47]
    assert all(result["support_checks"].values())
    assert result["clock"]["rows"] == 318
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    assert sha(support.RESULT) == (
        "0c0e0036d012017e074a4f1c4886461ae5aec7fcfe61cf533159f85dad4c2ad3"
    )
