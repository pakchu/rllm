import hashlib
import json
from pathlib import Path

from training import build_high_volatility_fan_token_result_rotation_relay_support as support


ARTIFACT = Path("results/high_volatility_fan_token_result_rotation_relay_support_2026-08-12.json")


def test_hvftrr_terminal_source_support_is_sealed():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "25b929334d85abe42f1d7604a00283f66ce9526933120e5562531e026f0c0ea0"
    )
    result = json.loads(ARTIFACT.read_text())
    assert result["policy_id"] == "HVFTRR-12"
    assert result["support"]["train"]["events"] == 6
    assert result["support"]["train"]["max_month_share"] == 0.5
    assert result["support"]["final"]["max_month_share"] > 0.45
    assert result["support_checks"]["train_minimum_events"] is False
    assert result["support_checks"]["train_month_concentration"] is False
    assert result["support_checks"]["final_month_concentration"] is False
    assert result["support_passed"] is False
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False


def test_hvftrr_support_manifest_and_outputs_are_hash_bound():
    result = json.loads(ARTIFACT.read_text())
    assert result["manifest_hash"] == support.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
    manifest_path = Path(result["source_manifest"]["path"]); assert support.sha(manifest_path) == result["source_manifest"]["sha256"]
    manifest = json.loads(manifest_path.read_text())
    for output in manifest["outputs"].values(): assert support.sha(Path(output["path"])) == output["sha256"]
