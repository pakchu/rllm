import hashlib
import json
from pathlib import Path

from training import build_high_volatility_path_analog_relay_support as support


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvpar_source_support_pass_is_oos_outcome_sealed() -> None:
    result = json.loads(support.RESULT.read_text())
    assert result["policy_id"] == "HVPAR-8"
    assert result["support_passed"] is True
    assert result["advance_to_gross9_novelty"] is True
    assert result["advance_to_economic_outcomes"] is False
    assert result["oos_postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train", "test", "eval", "final")] == [44, 88, 101, 37]


def test_hvpar_support_hashes_bind_frozen_inputs_and_clocks() -> None:
    result = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == support.canonical_hash(core)
    for key in ("preregistration", "source_manifest", "model_freeze"):
        assert result[key]["sha256"] == sha(Path(result[key]["path"]))
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    assert set(result["controls"]) == set(support.CONTROLS)
    for item in result["controls"].values():
        assert item["promotion_authorized"] is False
        assert item["sha256"] == sha(Path(item["path"]))

