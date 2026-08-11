import hashlib
import json
from pathlib import Path

from training import build_high_volatility_air_pollution_penalty_rotation_relay_support as support


ARTIFACT = Path("results/high_volatility_air_pollution_penalty_rotation_relay_support_2026-08-12.json")


def test_hvappr_terminal_source_support_is_sealed():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "d86b5d8b8f2401b842b7d429a6ec2a376e375190a6abc4d8efae9b58f73036e7"
    )
    result = json.loads(ARTIFACT.read_text())
    assert result["policy_id"] == "HVAPPR-24"
    assert result["support"]["final"]["events"] == 29
    assert result["support"]["final"]["max_month_share"] > 0.45
    assert result["support_checks"]["final_month_concentration"] is False
    assert result["support_passed"] is False
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["decision"] == "terminal_source_support_reject"


def test_hvappr_support_manifest_and_outputs_are_hash_bound():
    result = json.loads(ARTIFACT.read_text())
    assert result["manifest_hash"] == support.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
    manifest_path = Path(result["source_manifest"]["path"])
    assert support.sha(manifest_path) == result["source_manifest"]["sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["outputs"]["aqi"]["missing_days"] == 15
    for output in manifest["outputs"].values():
        assert support.sha(Path(output["path"])) == output["sha256"]
